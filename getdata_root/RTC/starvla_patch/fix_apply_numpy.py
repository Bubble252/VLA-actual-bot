#!/usr/bin/env python3
from pathlib import Path

p = Path("/root/code/Starvla/deployment/model_server/policy_norm_processor.py")
text = p.read_text()
old = "            data[full_key] = torch.as_tensor(slice_, dtype=torch.float32)\n"
# Only replace inside apply_actions: find apply_actions and replace first tensor assign there
idx = text.find("def apply_actions(")
if idx < 0:
    raise SystemExit("apply_actions missing")
# within apply_actions until next def
end = text.find("\n    def ", idx + 1)
if end < 0:
    end = len(text)
block = text[idx:end]
if "np.asarray(slice_" in block:
    print("already fixed")
else:
    if old not in block:
        raise SystemExit("tensor assign not found in apply_actions")
    block2 = block.replace(
        old,
        "            # transform.apply expects numpy arrays\n"
        "            data[full_key] = np.asarray(slice_, dtype=np.float32)\n",
        1,
    )
    text = text[:idx] + block2 + text[end:]
    p.write_text(text)
    print("fixed")
