"""Generate and load self-contained, provenance-checked almanac datasets."""
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
from uuid import uuid4

import numpy as np
from astropy.table import Table

from . import __version__
from ._vendor import astronomy
from .calculations import (
    SOLAR_EXACT_EXTREME_DEGREES, SOLAR_SAMPLE_MINUTES, TABLE_COLUMNS,
    compute_ephemeris, compute_eot, compute_sun_alts,
)
from .stars import star_manifest
from .topo import TopoContext

SCHEMA_VERSION = 1
FILE_NAMES = {
    "rise": "rise.fits", "set": "set.fits", "transit": "transit.fits",
    "twilight": "twilight.fits", "EOT": "EOT.npy", "sun_alts": "sun_alts.npy",
}
ARRAY_UNITS = {"EOT": "hour", "sun_alts": "degree"}
TIME_BASIS = "fixed UTC offset; local noon to following local noon"


@dataclass(frozen=True)
class EphemerisDataset:
    """One complete generated run; topo is reconstructed from its manifest."""

    topo: TopoContext
    run_dir: Path
    manifest: dict
    tables: dict[str, Table]
    arrays: dict[str, np.ndarray]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _number_slug(value) -> str:
    return format(float(value), ".12g").replace("-", "m").replace(".", "p")


def _create_run_dir(topo: TopoContext, data_root: Path) -> tuple[Path, str]:
    data_root.mkdir(parents=True, exist_ok=True)
    offset = topo.utc_offset_minutes
    sign = "p" if offset >= 0 else "m"
    hours, minutes = divmod(abs(offset), 60)
    lat_sign = "N" if topo.latitude_deg >= 0 else "S"
    lon = topo.observer.longitude
    lon_sign = "E" if lon >= 0 else "W"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    readable = (
        f"{topo.year}_UTC{sign}{hours:02d}{minutes:02d}_"
        f"{lat_sign}{_number_slug(abs(topo.latitude_deg))}_"
        f"{lon_sign}{_number_slug(abs(lon))}_H{_number_slug(topo.elevation_m)}m_{stamp}"
    )
    while True:
        run_id = f"{readable}_{uuid4().hex[:8]}"
        run_dir = data_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_dir.resolve(), run_id


def _derived(topo: TopoContext) -> dict:
    return {
        "longitude_deg": topo.observer.longitude,
        "days_in_year": topo.days_in_year,
        "first_date": topo.dates[0].isoformat(),
        "last_date": topo.dates[-1].isoformat(),
        "origin_utc": str(topo.origin_utc),
        "end_utc_exclusive": str(topo.end_utc),
        "time_basis": TIME_BASIS,
        "missing_event": -999.0,
        "sample_interval_minutes": 1,
    }


def _algorithm() -> dict:
    return {
        "version": __version__,
        "astronomy_sha256": _sha256(Path(astronomy.__file__)),
        "solar_altitude": {
            "exact_sample_interval_minutes": SOLAR_SAMPLE_MINUTES,
            "output_interval_minutes": 1,
            "interpolation": "cubic_spline_sine_altitude_not_a_knot",
            "exact_above_absolute_altitude_degrees": SOLAR_EXACT_EXTREME_DEGREES,
            "include_final_minute": True,
        },
    }


def _supported_algorithm(record) -> bool:
    """Accept the current sampler and the explicitly known original algorithm.

    Original 1.0.0 datasets remain readable without rewriting their provenance.
    Both versions require the same fixed engine; unknown versions/settings fail.
    """
    current = _algorithm()
    original = {"version": "1.0.0", "astronomy_sha256": current["astronomy_sha256"]}
    return record == current or record == original


def _dependencies() -> dict:
    versions = {"python": platform.python_version()}
    for package in ("numpy", "astropy", "matplotlib", "scipy", "pillow"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _validate_payload(topo, tables, arrays):
    if set(tables) != set(TABLE_COLUMNS):
        raise ValueError("星历表集合不完整")
    if set(arrays) != set(ARRAY_UNITS):
        raise ValueError("数组集合不完整")
    expected_days = np.arange(1, topo.days_in_year + 1, dtype=float)
    for key, expected_columns in TABLE_COLUMNS.items():
        table = tables[key]
        if table.colnames != list(expected_columns) or len(table) != topo.days_in_year:
            raise ValueError(f"{key} 表的列或天数与配置不一致")
        if not np.array_equal(np.asarray(table["Day"]), expected_days):
            raise ValueError(f"{key} 表的 Day 必须是完整连续的一基日期索引")
        for name in expected_columns:
            values = np.asarray(table[name])
            if values.dtype.kind not in "fiu" or not np.isfinite(values).all():
                raise ValueError(f"{key}.{name} 必须为有限数值，缺失事件使用 -999")
            if np.ma.isMaskedArray(table[name]) and np.ma.getmaskarray(table[name]).any():
                raise ValueError(f"{key}.{name} 包含未定义的掩码值")
    for key, shape in (("EOT", (topo.days_in_year,)),
                       ("sun_alts", (topo.days_in_year, 1440))):
        array = arrays[key]
        if array.shape != shape or array.dtype.kind not in "fiu":
            raise ValueError(f"{key} 数组形状或类型与配置不一致")
        if not np.isfinite(array).all():
            raise ValueError(f"{key} 数组包含非有限数值")


def _file_record(key: str, path: Path, tables, arrays) -> dict:
    record = {"name": path.name, "sha256": _sha256(path)}
    if key in TABLE_COLUMNS:
        table = tables[key]
        record.update({
            "format": "fits", "shape": [len(table), len(table.colnames)],
            "columns": table.colnames,
            "units": {name: "day_index" if name == "Day" else "hour"
                      for name in table.colnames},
        })
    else:
        record.update({"format": "npy", "shape": list(arrays[key].shape),
                       "units": ARRAY_UNITS[key]})
    return record


def generate_dataset(
    topo: TopoContext, data_root: str | Path,
    progress: Callable[[str], None] | None = None,
) -> EphemerisDataset:
    """Create a fresh readable run directory; never reuse or overwrite a run.

    A failed generation leaves an incomplete manifest for diagnosis. Only a
    fully validated run is atomically marked complete and accepted by loading.
    """
    if not isinstance(topo, TopoContext):
        raise TypeError("topo 必须是 TopoContext")
    run_dir, run_id = _create_run_dir(topo, Path(data_root))
    manifest = {
        "schema_version": SCHEMA_VERSION, "status": "incomplete", "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "topo": topo.to_dict(), "derived": _derived(topo),
        "stars": star_manifest(), "algorithm": _algorithm(),
        "dependencies": _dependencies(), "files": {},
    }
    _write_json_atomic(run_dir / "manifest.json", manifest)
    if progress is not None:
        progress(f"计算星历：{run_id}")
    tables = compute_ephemeris(topo)
    if progress is not None:
        progress("计算均时差")
    eot = compute_eot(topo)
    arrays = {"EOT": eot, "sun_alts": compute_sun_alts(topo, progress=progress)}
    _validate_payload(topo, tables, arrays)
    for key, name in FILE_NAMES.items():
        path = run_dir / name
        if key in tables:
            tables[key].write(path, format="fits", overwrite=False)
        else:
            with path.open("xb") as stream:
                np.save(stream, arrays[key], allow_pickle=False)
        manifest["files"][key] = _file_record(key, path, tables, arrays)
    manifest["status"] = "complete"
    _write_json_atomic(run_dir / "manifest.json", manifest)
    if progress is not None:
        progress(f"星历数据已保存：{run_dir}")
    return EphemerisDataset(topo, run_dir, manifest, tables, arrays)


def _read_manifest(run_dir: Path) -> dict:
    path = run_dir / "manifest.json"
    if not path.is_file():
        raise ValueError(f"缺少 manifest.json，不能推测旧数据的配置：{run_dir}")
    if path.resolve().parent != run_dir:
        raise ValueError("manifest.json 不能指向数据目录外")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("manifest.json 不是有效的 UTF-8 JSON") from exc
    required = {"schema_version", "status", "run_id", "created_at_utc", "topo",
                "derived", "stars", "algorithm", "dependencies", "files"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("manifest 字段不完整或含不支持的字段")
    if type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("不支持的 manifest schema 版本")
    if manifest["status"] != "complete":
        raise ValueError("数据生成尚未完成，不能加载")
    if not isinstance(manifest["run_id"], str) or not manifest["run_id"]:
        raise ValueError("manifest 的 run_id 无效")
    try:
        created = datetime.fromisoformat(manifest["created_at_utc"])
        if created.tzinfo is None or created.utcoffset().total_seconds() != 0:
            raise ValueError("创建时间必须为 UTC")
    except (TypeError, ValueError) as exc:
        raise ValueError("manifest 的 created_at_utc 无效") from exc
    dependencies = manifest["dependencies"]
    if not isinstance(dependencies, dict) or not dependencies or not all(
        isinstance(k, str) and isinstance(v, str) and v for k, v in dependencies.items()
    ):
        raise ValueError("manifest 的依赖版本记录无效")
    return manifest


def load_dataset(run_dir: str | Path) -> EphemerisDataset:
    """Load only a complete, internally consistent dataset with verified files.

    Reject missing manifests, changed file hashes, escaped paths, mixed dates,
    incompatible schemas/algorithms and inconsistent array or table dimensions.
    """
    run_dir = Path(run_dir).resolve()
    manifest = _read_manifest(run_dir)
    try:
        topo = TopoContext.from_dict(manifest["topo"])
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError("manifest 的 TopoContext 配置无效") from exc
    if manifest["derived"] != _derived(topo):
        raise ValueError("manifest 派生时间或观测位置与配置不一致")
    if manifest["stars"] != star_manifest():
        raise ValueError("manifest 恒星编号或坐标与当前星表不一致")
    if not _supported_algorithm(manifest["algorithm"]):
        raise ValueError("数据算法版本或天文引擎与当前实现不一致")
    files = manifest["files"]
    if not isinstance(files, dict) or set(files) != set(FILE_NAMES):
        raise ValueError("manifest 文件列表不完整")
    tables, arrays = {}, {}
    for key, expected_name in FILE_NAMES.items():
        record = files[key]
        if not isinstance(record, dict) or record.get("name") != expected_name:
            raise ValueError(f"{key} 文件名不符合数据集协议")
        path = run_dir / expected_name
        if path.resolve().parent != run_dir:
            raise ValueError(f"{key} 文件路径越出数据目录")
        if not path.is_file():
            raise ValueError(f"数据文件缺失：{expected_name}")
        if record.get("sha256") != _sha256(path):
            raise ValueError(f"数据文件校验失败：{expected_name}")
        try:
            if key in TABLE_COLUMNS:
                tables[key] = Table.read(path, format="fits")
            else:
                arrays[key] = np.load(path, allow_pickle=False)
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f"不能读取数据文件：{expected_name}") from exc
        if record != _file_record(key, path, tables, arrays):
            raise ValueError(f"{expected_name} 的形状、单位或列记录不一致")
    _validate_payload(topo, tables, arrays)
    return EphemerisDataset(topo, run_dir, manifest, tables, arrays)
