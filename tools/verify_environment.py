# -*- coding: utf-8 -*-
"""SAC (Smart App Control) 关闭后的环境验证脚本。

每个测试在独立子进程中运行——本机曾出现原生 DLL 被拦截时进程静默退出
(exit 127, 无回溯)，单进程脚本会在第一个失败处无声死亡、丢失后续结果。

用法（项目根目录 PowerShell）：
    .\.venv312\python.exe tools\verify_environment.py

把全部输出原样贴回对话即可。
"""
import subprocess
import sys

TESTS = [
    ("numpy GEMM (matmul 512x512)",
     "import numpy as np;"
     "a=np.random.default_rng(0).standard_normal((512,512));"
     "b=np.random.default_rng(1).standard_normal((512,512));"
     "c=a@b;"
     "assert np.isfinite(c).all() and abs(c[0,0])>0;"
     "print('numpy',np.__version__,'dot ok')"),

    ("numpy LAPACK (eigh/lstsq)",
     "import numpy as np;"
     "rng=np.random.default_rng(0);"
     "m=rng.standard_normal((128,128));m=m+m.T;"
     "w=np.linalg.eigvalsh(m);"
     "x=np.linalg.lstsq(rng.standard_normal((50,10)),rng.standard_normal(50),rcond=None)[0];"
     "assert np.isfinite(w).all() and np.isfinite(x).all();"
     "print('eigh ok, lstsq ok')"),

    ("pybullet import + DIRECT step",
     "import pybullet as p;"
     "cid=p.connect(p.DIRECT);"
     "p.setGravity(0,0,-9.8);"
     "p.loadURDF('plane.urdf') if False else None;"
     "[p.stepSimulation() for _ in range(10)];"
     "p.disconnect();"
     "print('pybullet',p.__version__ if hasattr(p,'__version__') else 'ok')"),

    ("matplotlib Agg render to PNG",
     "import matplotlib;"
     "matplotlib.use('Agg');"
     "import matplotlib.pyplot as plt, tempfile, os;"
     "fig,ax=plt.subplots();ax.plot([0,1,2],[0,1,0]);ax.set_title('probe');"
     "fp=os.path.join(tempfile.gettempdir(),'mpl_probe.png');"
     "fig.savefig(fp,dpi=80);"
     "assert os.path.getsize(fp)>3000;"
     "print('matplotlib',matplotlib.__version__,'png',os.path.getsize(fp),'bytes')"),

    ("mujoco import + step",
     "import mujoco, numpy as np;"
     "m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><freejoint/>"
     "<geom size=\".1\" mass=\"1\"/></body></worldbody></mujoco>');"
     "d=mujoco.MjData(m);"
     "[mujoco.mj_step(m,d) for _ in range(10)];"
     "print('mujoco',mujoco.__version__,'ok')"),
]


def main():
    print("=" * 64)
    print("interpreter:", sys.executable)
    print("python:", sys.version.split()[0])
    print("=" * 64)
    fails = 0
    for name, code in TESTS:
        r = subprocess.run([sys.executable, "-c", code],
                           capture_output=True, text=True, timeout=180)
        if r.returncode == 0:
            detail = r.stdout.strip().replace("\n", " | ")
            print(f"[PASS] {name}  --  {detail}")
        else:
            fails += 1
            err = (r.stderr.strip().splitlines() or ["<no traceback>"])[-1]
            print(f"[FAIL] {name}  --  exit={r.returncode}  {err}")
    print("=" * 64)
    print(f"SUMMARY: {len(TESTS) - fails}/{len(TESTS)} passed, {fails} failed")
    print("=" * 64)


if __name__ == "__main__":
    main()
