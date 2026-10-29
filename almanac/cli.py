"""Command-line orchestration. Only this layer chooses run configuration."""

import argparse
from pathlib import Path
import re
import sys

from .topo import TopoContext

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_utc_offset(value: str) -> int:
    """Parse a signed whole hour or HH:MM fixed offset into integer minutes."""
    match = re.fullmatch(r"([+-]?)(\d{1,2})(?::([0-5]\d))?", value)
    if not match:
        raise argparse.ArgumentTypeError("时区格式应为 +8、+08:00 或 -05:30")
    sign, hours, minutes = match.groups()
    offset = (int(hours) * 60 + int(minutes or 0)) * (-1 if sign == "-" else 1)
    if not -720 <= offset <= 840:
        raise argparse.ArgumentTypeError("固定时区必须在 UTC-12:00 到 UTC+14:00 之间")
    return offset


def build_parser() -> argparse.ArgumentParser:
    """Build CLI and GUI entry points without starting any computation."""
    parser = argparse.ArgumentParser(description="固定时区、标准经度的全年行星年历")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("gui", help="打开轻量图形界面")
    for command, help_text in [("all", "生成新数据集和完整 PNG/PDF"),
                               ("generate", "只生成新的星历数据集")]:
        child = commands.add_parser(command, help=help_text)
        child.add_argument("--year", type=int, required=True)
        child.add_argument("--utc-offset", type=parse_utc_offset, required=True,
                           help="例如 --utc-offset=+08:00；西时区使用 --utc-offset=-05:00")
        child.add_argument("--latitude", type=float, required=True,
                           help="北纬为正，南纬为负，单位：度")
        child.add_argument("--elevation", type=float, default=0.0, help="海拔（米），默认 0")
        child.add_argument("--data-root", type=Path, default=PROJECT_ROOT / "data" / "runs")
        if command == "all":
            child.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
            child.add_argument("--no-stars", action="store_true", help="不绘制恒星曲线和标注")
    render = commands.add_parser("render", help="从指定数据集重新生成完整 PNG/PDF")
    render.add_argument("--run-dir", type=Path, required=True)
    render.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "output")
    render.add_argument("--overwrite", action="store_true", help="允许覆盖该数据集已有图像")
    render.add_argument("--no-stars", action="store_true", help="不绘制恒星曲线和标注")
    return parser


def main(argv=None) -> int:
    """Execute a requested action, returning 0 on success or 2 on input/data errors."""
    args = build_parser().parse_args(argv)
    if args.command == "gui":
        from .gui import launch_gui
        return launch_gui()
    from .dataset import generate_dataset, load_dataset

    try:
        if args.command == "render":
            dataset = load_dataset(args.run_dir)
        else:
            topo = TopoContext(args.year, args.utc_offset, args.latitude, args.elevation)
            dataset = generate_dataset(topo, args.data_root,
                                       progress=lambda message: print(message, flush=True))
        print(f"数据集：{dataset.run_dir}", flush=True)
        if args.command != "generate":
            from .render import RenderOptions, render_almanac
            options = RenderOptions(overwrite=getattr(args, "overwrite", False),
                                    stars=not args.no_stars)
            outputs = render_almanac(dataset, args.output_root, options)
            for kind, path in outputs.items():
                print(f"{kind.upper()}：{path}", flush=True)
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
