# 迁移与 GitHub

## 从哪台机器打包的

本仓内容从下列路径同步（2026-10 批次）：

| 角色 | 原路径 |
|---|---|
| client + IK 修复 | `ARX_X5_clean(7.12eval)/lewm_x5_grasp/` |
| CAN / ROS2 源码 | `ARX_X5_clean(7.12eval)/ARX_CAN`、`ROS2/X5_ws/src` |
| 部署脚本 | `...server46.../deploy_arxx5_ljn_pick_stage2_90k/` |
| RTC / tool | `...server46.../RTC`、`scripts/tool` |
| ModelClient | `~/project/starVLA` |

**故意未纳入**：`server46/.../src/lewm_x5_grasp/eef_to_joint_ik.py`（另一套实现、无 rotvec 修复）。

---

## 推到 GitHub（私有仓推荐）

勿 push 到上游 `ARXroboticsX/ARX_X5`。建议新建私有仓：

```bash
cd /path/to/arx_x5_ljn_deploy

# 若尚未 init（本仓初始化时一般已做过）
git status

# 创建并推送（需已登录 gh）
gh repo create arx_x5_ljn_deploy --private --source=. --remote=origin --push

# 或手动
# git remote add origin git@github.com:<you>/arx_x5_ljn_deploy.git
# git push -u origin main
```

新机：

```bash
git clone git@github.com:<you>/arx_x5_ljn_deploy.git
cd arx_x5_ljn_deploy
# 然后按根目录 README「新机最短路径」编译 ROS2 并启动
```

---

## 换机后只需改什么

| 情况 | 操作 |
|---|---|
| 仓放在任意路径 | **通常不用改**（`deploy/env.sh` 用相对仓根） |
| 101 主机/端口变了 | 改 `deploy/env.sh` 的 `SERVER_HOST` / `SERVER_PORT` / `POLICY_PORT` |
| SSH 密钥 | 配置本机 `~/.ssh`，不要把私钥提交进仓 |
| GPU 卡号 | `GPU_IDS`（默认 1，避开 0813 的 0） |

---

## 从开发机同步更新到本仓

在开发机改完 `7.12eval` 的 client 后：

```bash
BUNDLE=~/project/arx_x5_ljn_deploy
EVAL=~/project/ARX_X5_clean\(7.12eval\)

rsync -a --delete \
  --exclude '__pycache__' --exclude 'data/' --exclude 'videos/' \
  "$EVAL/lewm_x5_grasp/" "$BUNDLE/eval_root/lewm_x5_grasp/"

cd "$BUNDLE" && git status && git add -A && git commit -m "sync client from 7.12eval"
```

部署脚本同步同理，指向 `deploy/` 即可。

---

## 体积说明

| 内容 | 是否进仓 |
|---|---|
| client + deploy 脚本 + RTC + starVLA | 是（约数 MB～数十 MB） |
| ROS2 **源码** | 是 |
| ROS2 build/install | **否**（`.gitignore`） |
| rollouts / avi | **否** |
| 101 ckpt / VLM | **否**（仍在 101） |
