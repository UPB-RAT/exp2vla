import omni.usd
from pxr import Usd, UsdGeom
import random

# ACTIVE STAGE
stage = omni.usd.get_context().get_stage()

# File paths
MULTI = "/home/rat-laptop/Documents/isaac_env/blender/multicorridor/multicorridor.usdc"
CIRCLE = "/home/rat-laptop/Documents/isaac_env/blender/circle_gate/circle_gate.usdc"
SQUARE = "/home/rat-laptop/Documents/isaac_env/blender/sq_gate/gate.usdc"

# Corridor dimensions
CORRIDOR_W = 25.0     # width  (x-axis)
CORRIDOR_L = 15.0    # length (y-axis)

# Minimal safe spacing
X_STEP = CORRIDOR_W + 0.5      # = 8.5
Y_STEP = -(CORRIDOR_L + 0.5)   # = -15.5  (arka sıra biraz geride)

# Placement regions
regions = [
    ((1,3), (-1.5,-0.4), (0,0)),
    ((0.2,-1.25), (-1.5,-0.4), (-7.5,7.5)),
    ((-2.5,-1.8), (0.1,0.1), (-95,-85)),
    ((0.2,-1.25), (0.4,1.3), (-7.5,7.5)),
    ((1,3), (0.4,1.3), (0,0)),
]

def pick_gate():
    return CIRCLE if random.random() < 0.5 else SQUARE


# WORLD ROOT
stage.DefinePrim("/World", "Xform")


# ============= 6 ENVIRONMENT DİP DİBE =============
for env_id in range(1):

    row = env_id // 3      # 0: öndeki 3, 1: arkadaki 3
    col = env_id % 3       # 0,1,2

    X_OFFSET = col * X_STEP
    Y_OFFSET = row * Y_STEP

    env_prim = stage.DefinePrim(f"/World/Env_{env_id+1}", "Xform")
    UsdGeom.Xformable(env_prim).AddTranslateOp().Set((X_OFFSET, Y_OFFSET, 0))

    # corridor
    corr = stage.DefinePrim(f"/World/Env_{env_id+1}/Multicorridor", "Xform")
    corr.GetReferences().AddReference(MULTI)

    # 5 gates
    for i, reg in enumerate(regions):
        (xr, yr, yaw_rng) = reg
        x = random.uniform(*xr)
        y = random.uniform(*yr)
        yaw = random.uniform(*yaw_rng) + 90

        gate_file = pick_gate()
        prim_path = f"/World/Env_{env_id+1}/Gate_{i+1}"
        prim = stage.DefinePrim(prim_path, "Xform")
        prim.GetReferences().AddReference(gate_file)

        xf = UsdGeom.Xformable(prim)
        xf.AddTranslateOp().Set((x, y, 0))
        xf.AddRotateZOp().Set(yaw)

    print(f"Env {env_id+1} → Pos ({X_OFFSET:.2f}, {Y_OFFSET:.2f})")

print("✔ 6 compact environments placed!")
