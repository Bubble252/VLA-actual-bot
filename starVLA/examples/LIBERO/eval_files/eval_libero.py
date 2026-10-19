import dataclasses
import json
import logging
import math
import multiprocessing
import os
import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

import imageio
import numpy as np
import tqdm
import tyro
from libero.libero import benchmark, get_libero_path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


def _binarize_gripper_open(open_val: Union[np.ndarray, float]) -> np.ndarray:
    arr = np.asarray(open_val, dtype=np.float32).reshape(-1)
    v = float(arr[0])
    bin_val = 1.0 - 2.0 * (v > 0.5)
    return np.asarray([bin_val], dtype=np.float32)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 10093
    # Comma-separated policy server ports for load balancing, e.g. "5694,5695,5696"
    policy_ports: str = ""
    resize_size: Tuple[int, int] = (224, 224)

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = "libero_goal"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    num_parallel_workers: int = 1  # Number of parallel episode rollouts (processes)
    max_tasks: int = -1  # If > 0, evaluate only the first N tasks for quick debugging

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "experiments/libero/logs"  # Path to save videos
    save_videos: bool = True

    seed: int = 7  # Random Seed (for reproducibility)

    pretrained_path: str = ""
    # Dataset key for un-normalization. None = auto (only if model has single dataset stats).
    unnorm_key: Optional[str] = None

    post_process_action: bool = True

    job_name: str = "test"

    # Optional comma-separated GPU ids for EGL rendering in parallel workers, e.g. "0,1,2,3"
    render_gpu_ids: str = ""


def _parse_policy_ports(args: Args) -> List[int]:
    if args.policy_ports.strip():
        return [int(p.strip()) for p in args.policy_ports.split(",") if p.strip()]
    return [args.port]


def _parse_render_gpu_ids(args: Args) -> List[int]:
    if not args.render_gpu_ids.strip():
        return []
    return [int(g.strip()) for g in args.render_gpu_ids.split(",") if g.strip()]


def _max_steps_for_suite(task_suite_name: str) -> int:
    base = task_suite_name
    for prefix in (
        "libero_spatial",
        "libero_object",
        "libero_goal",
        "libero_10",
        "libero_90",
    ):
        if task_suite_name == prefix or task_suite_name.startswith(prefix + "_"):
            base = prefix
            break
    if base == "libero_spatial":
        return 220
    if base == "libero_object":
        return 280
    if base == "libero_goal":
        return 300
    if base == "libero_10":
        return 520
    if base == "libero_90":
        return 400
    raise ValueError(f"Unknown task suite: {task_suite_name}")


def _configure_mujoco_render(render_gpu_id: Optional[int]) -> None:
    """Must run before importing robosuite / OffScreenRenderEnv."""
    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    if render_gpu_id is not None:
        os.environ["MUJOCO_EGL_DEVICE_ID"] = str(render_gpu_id)


def _get_libero_env(task, resolution, seed, render_gpu_id: Optional[int] = None):
    """Initializes and returns the LIBERO environment, along with the task description."""
    _configure_mujoco_render(render_gpu_id)
    from libero.libero.envs import OffScreenRenderEnv

    task_description = task.language
    task_bddl_file = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _run_single_episode(payload: dict) -> dict:
    """Run one LIBERO episode in an isolated process (env + websocket client)."""
    from examples.LIBERO.eval_files.model2libero_interface import ModelClient

    args_dict = payload["args"]
    args = Args(**args_dict)
    task_id = payload["task_id"]
    episode_idx = payload["episode_idx"]
    policy_port = payload["policy_port"]
    max_steps = payload["max_steps"]
    render_gpu_id: Optional[int] = payload.get("render_gpu_id")

    _configure_mujoco_render(render_gpu_id)
    logging.basicConfig(level=logging.INFO, force=True)
    logging.info(
        f"Episode start: task={task_id} ep={episode_idx} port={policy_port} "
        f"egl_gpu={render_gpu_id if render_gpu_id is not None else 'default'}"
    )

    np.random.seed(args.seed + task_id * 1000 + episode_idx)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    task = task_suite.get_task(task_id)
    initial_states = task_suite.get_task_init_states(task_id)

    success = False
    failure_reason = "unknown"
    error_message: Optional[str] = None
    step = 0
    replay_images = []
    env = None
    client_model = None
    task_description = str(task.language)
    try:
        env, task_description = _get_libero_env(
            task, LIBERO_ENV_RESOLUTION, args.seed + episode_idx, render_gpu_id=render_gpu_id
        )
        client_model = ModelClient(
            unnorm_key=args.unnorm_key,
            host=args.host,
            port=policy_port,
            image_size=list(args.resize_size),
        )
        client_model.reset(task_description=task_description)
        env.reset()
        obs = env.set_init_state(initial_states[episode_idx])

        t = 0
        done = False
        while t < max_steps + args.num_steps_wait:
            if t < args.num_steps_wait:
                obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                t += 1
                continue

            img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
            replay_images.append(img)

            state = np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            )

            example_dict = {
                "image": [img, wrist_img],
                "lang": str(task_description),
            }

            response = client_model.step(example=example_dict, step=step)
            raw_action = response["raw_action"]

            world_vector_delta = np.asarray(
                raw_action.get("world_vector"), dtype=np.float32
            ).reshape(-1)
            rotation_delta = np.asarray(
                raw_action.get("rotation_delta"), dtype=np.float32
            ).reshape(-1)
            open_gripper = np.asarray(
                raw_action.get("open_gripper"), dtype=np.float32
            ).reshape(-1)
            gripper = _binarize_gripper_open(open_gripper)

            if not (
                world_vector_delta.size == 3
                and rotation_delta.size == 3
                and open_gripper.size == 1
            ):
                raise ValueError(
                    f"Invalid action sizes: world_vector={world_vector_delta.shape}, "
                    f"rotation_delta={rotation_delta.shape}, gripper={gripper.shape}"
                )

            delta_action = np.concatenate(
                [world_vector_delta, rotation_delta, gripper], axis=0
            )
            obs, reward, done, info = env.step(delta_action.tolist())
            if done:
                success = True
                failure_reason = "success"
                break
            t += 1
            step += 1

        if not success:
            failure_reason = "max_steps_exceeded"
    except Exception as exc:
        failure_reason = f"exception:{type(exc).__name__}"
        error_message = str(exc)
        logging.exception(
            f"Episode exception: task={task_id} ep={episode_idx} port={policy_port}"
        )
    finally:
        try:
            if client_model is not None:
                client_model.client.close()
        except Exception:
            pass
        try:
            if env is not None:
                env.close()
        except Exception:
            pass

    video_path = None
    if args.save_videos and replay_images:
        suffix = "success" if success else "failure"
        task_segment = task_description.replace(" ", "_")
        video_path = (
            pathlib.Path(args.video_out_path)
            / f"rollout_task{task_id}_{task_segment}_episode{episode_idx}_{suffix}.mp4"
        )
        imageio.mimwrite(
            video_path,
            [np.asarray(x) for x in replay_images],
            fps=10,
        )

    logging.info(
        f"Episode done: task={task_id} ep={episode_idx} success={success} "
        f"reason={failure_reason} steps={step}"
    )
    return {
        "task_id": task_id,
        "episode_idx": episode_idx,
        "task_description": task_description,
        "success": success,
        "failure_reason": failure_reason,
        "error_message": error_message,
        "steps": step,
        "video_path": str(video_path) if video_path is not None else None,
        "policy_port": policy_port,
    }


def _build_work_items(args: Args, num_tasks: int, policy_ports: List[int]) -> List[dict]:
    max_steps = _max_steps_for_suite(args.task_suite_name)
    args_dict = dataclasses.asdict(args)
    render_gpu_ids = _parse_render_gpu_ids(args)
    work_items = []
    work_index = 0
    for task_id in range(num_tasks):
        for episode_idx in range(args.num_trials_per_task):
            render_gpu_id = None
            if render_gpu_ids:
                render_gpu_id = render_gpu_ids[work_index % len(render_gpu_ids)]
            work_items.append(
                {
                    "task_id": task_id,
                    "episode_idx": episode_idx,
                    "policy_port": policy_ports[work_index % len(policy_ports)],
                    "max_steps": max_steps,
                    "render_gpu_id": render_gpu_id,
                    "args": args_dict,
                }
            )
            work_index += 1
    return work_items


def _ensure_libero_config() -> None:
    """Verify LIBERO paths from config.yaml; warn if init_states is missing."""
    init_states = get_libero_path("init_states")
    if not os.path.isdir(init_states):
        libero_root = os.environ.get("LIBERO_HOME", "")
        hint = (
            f"init_states not found: {init_states}\n"
            f"Fix LIBERO/libero/config.yaml or run:\n"
            f"  export LIBERO_CONFIG_PATH=${{LIBERO_HOME}}/libero\n"
            f"  python -c \"from libero.libero import set_libero_default_path; "
            f"set_libero_default_path('{libero_root}/libero/libero')\""
        )
        raise FileNotFoundError(hint)


def eval_libero(args: Args) -> None:
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Arguments: {json.dumps(dataclasses.asdict(args), indent=4)}")

    _ensure_libero_config()
    np.random.seed(args.seed)

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    n_eval_tasks = (
        num_tasks_in_suite
        if args.max_tasks <= 0
        else min(args.max_tasks, num_tasks_in_suite)
    )
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    max_steps = _max_steps_for_suite(args.task_suite_name)
    policy_ports = _parse_policy_ports(args)

    work_items = _build_work_items(args, n_eval_tasks, policy_ports)
    total_episodes = len(work_items)
    render_gpu_ids = _parse_render_gpu_ids(args)
    if args.num_parallel_workers > 1:
        if len(render_gpu_ids) < args.num_parallel_workers:
            raise ValueError(
                f"num_parallel_workers={args.num_parallel_workers} requires at least that many "
                f"render_gpu_ids (got {render_gpu_ids or 'none'}). "
                f"Each worker needs its own GPU for EGL offscreen rendering, "
                f"or set num_parallel_workers=1."
            )
    logging.info(
        f"Evaluating {total_episodes} episodes "
        f"({n_eval_tasks} tasks x {args.num_trials_per_task} trials) "
        f"with {args.num_parallel_workers} parallel worker(s), "
        f"policy port(s): {policy_ports}"
    )

    results: List[dict] = []
    if args.num_parallel_workers <= 1:
        for item in tqdm.tqdm(work_items, desc="Episodes"):
            results.append(_run_single_episode(item))
    else:
        mp_ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=args.num_parallel_workers, mp_context=mp_ctx
        ) as executor:
            futures = {
                executor.submit(_run_single_episode, item): item for item in work_items
            }
            with tqdm.tqdm(total=total_episodes, desc="Episodes") as pbar:
                for future in as_completed(futures):
                    results.append(future.result())
                    pbar.update(1)
                    done = len(results)
                    succ = sum(1 for r in results if r["success"])
                    logging.info(
                        f"Progress: {done}/{total_episodes} episodes, "
                        f"success rate so far: {succ}/{done} ({100.0 * succ / done:.1f}%)"
                    )

    total_successes = sum(1 for r in results if r["success"])
    task_stats: Dict[int, dict] = {}
    for r in results:
        tid = r["task_id"]
        if tid not in task_stats:
            task_stats[tid] = {"episodes": 0, "successes": 0, "description": r["task_description"]}
        task_stats[tid]["episodes"] += 1
        task_stats[tid]["successes"] += int(r["success"])

    for task_id in sorted(task_stats.keys()):
        stats = task_stats[task_id]
        rate = stats["successes"] / stats["episodes"] if stats["episodes"] else 0.0
        logging.info(
            f"Task {task_id} ({stats['description']}): "
            f"{stats['successes']}/{stats['episodes']} = {rate * 100:.1f}%"
        )

    overall_rate = total_successes / total_episodes if total_episodes else 0.0
    logging.info(
        f"Total success rate: {total_successes}/{total_episodes} = {overall_rate * 100:.1f}%"
    )
    failure_reason_stats: Dict[str, int] = {}
    for r in results:
        reason = r.get("failure_reason", "unknown")
        failure_reason_stats[reason] = failure_reason_stats.get(reason, 0) + 1
    logging.info(f"Failure reason stats: {json.dumps(failure_reason_stats, ensure_ascii=False)}")
    logging.info(f"Max steps per episode (suite limit): {max_steps}")


def start_debugpy_once():
    import debugpy

    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10092))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10092 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    tyro.cli(eval_libero)
