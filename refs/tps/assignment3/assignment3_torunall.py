"""we run the exact assignment3 pipeline for all i and save outputs here"""

from __future__ import annotations

from pathlib import Path  # we import path


def main() -> None:
    assignment3_dir = Path(__file__).resolve().parents[3]  # we locate refs/tps/assignment3
    assignment3_path = assignment3_dir / "Assignment 3.py"  # we point to the working pipeline script

    out_dir = Path(__file__).resolve().parent / "data"  # we save logs/results here
    figures_dir = out_dir / "figures"  # we save figures here
    out_dir.mkdir(parents=True, exist_ok=True)  # we ensure folder exists
    figures_dir.mkdir(parents=True, exist_ok=True)  # we ensure folder exists

    code = assignment3_path.read_text(encoding="utf-8")  # we read the working code
    code = code.replace(
        'figures_dir = os.path.join(data_dir, "data", "figures")',
        f'figures_dir = r"{figures_dir.as_posix()}"',
        1,
    )  # we redirect figures output
    code = code.replace(
        'output_dir = os.path.join(data_dir, "data")',
        f'output_dir = r"{out_dir.as_posix()}"',
        1,
    )  # we redirect results/logs output

    glb: dict[str, object] = {"__file__": str(assignment3_path), "__name__": "__main__"}  # we execute as script
    exec(compile(code, str(assignment3_path), "exec"), glb, glb)  # we run exact pipeline


if __name__ == "__main__":
    main()  # we run
