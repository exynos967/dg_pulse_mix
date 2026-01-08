#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import re
import sys
from pathlib import Path
from typing import Any
from urllib.parse import unquote


SECTION_TIME_MAP: list[float] = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1,
    1.1,
    1.2,
    1.3,
    1.4,
    1.5,
    1.6,
    1.7,
    1.8,
    1.9,
    2,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3,
    3.1,
    3.2,
    3.3,
    3.4,
    3.5,
    3.6,
    3.7,
    3.8,
    3.9,
    4,
    4.1,
    4.2,
    4.3,
    4.4,
    4.5,
    4.6,
    4.7,
    4.8,
    4.9,
    5,
    5.2,
    5.4,
    5.6,
    5.8,
    6,
    6.2,
    6.4,
    6.6,
    6.8,
    7,
    7.2,
    7.4,
    7.6,
    7.8,
    8,
    8.5,
    9,
    9.5,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    23.4,
    26.6,
    30,
    33.4,
    36.6,
    40,
    45,
    50,
    55,
    60,
    70,
    80,
    90,
    100,
    120,
    140,
    160,
    180,
    200,
    250,
    300,
]


OLD_SECTION_TIME_MAP: list[float] = [
    0.1,
    0.1,
    0.1,
    0.1,
    0.1,
    0.2,
    0.2,
    0.2,
    0.3,
    0.3,
    0.3,
    0.4,
    0.4,
    0.5,
    0.5,
    0.6,
    0.6,
    0.7,
    0.7,
    0.8,
    0.9,
    0.9,
    1,
    1.1,
    1.1,
    1.2,
    1.3,
    1.3,
    1.4,
    1.5,
    1.6,
    1.6,
    1.7,
    1.8,
    1.9,
    2.0,
    2.1,
    2.1,
    2.2,
    2.3,
    2.4,
    2.5,
    2.6,
    2.7,
    2.8,
    2.9,
    3.0,
    3.1,
    3.2,
    3.3,
    3.4,
    3.5,
    3.6,
    3.7,
    3.8,
    3.9,
    4.1,
    4.2,
    4.3,
    4.4,
    4.5,
    4.6,
    4.7,
    4.9,
    5.0,
    5.1,
    5.2,
    5.4,
    5.5,
    5.6,
    5.7,
    5.9,
    6.0,
    6.1,
    6.3,
    6.4,
    6.5,
    6.7,
    6.8,
    6.9,
    7.1,
    7.2,
    7.4,
    7.5,
    7.6,
    7.8,
    7.9,
    8.1,
    8.2,
    8.4,
    8.5,
    8.7,
    8.8,
    9.0,
    9.1,
    9.3,
    9.4,
    9.6,
    9.7,
    9.9,
    10.0,
]


DUNGEONLAB_PREFIX = "Dungeonlab+pulse:"

_FLOAT_PREFIX_RE = re.compile(r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
_INT_PREFIX_RE = re.compile(r"^\s*([+-]?\d+)")


class PulseParseError(ValueError):
    pass


def _parse_float_js(value: str, *, field: str) -> float:
    match = _FLOAT_PREFIX_RE.match(value)
    if not match:
        raise PulseParseError(f"{field} 不是合法数字: {value!r}")
    return float(match.group(1))


def _parse_int_js(value: str, *, field: str) -> int:
    match = _INT_PREFIX_RE.match(value)
    if not match:
        raise PulseParseError(f"{field} 不是合法整数: {value!r}")
    return int(match.group(1))


def _normalize_pulse_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith(DUNGEONLAB_PREFIX):
        return text

    # 1) URL-encoded content (common when copied from query params)
    decoded = unquote(text)
    if decoded.startswith(DUNGEONLAB_PREFIX):
        return decoded

    # 2) base64-encoded content
    b64_compact = "".join(text.split())
    if len(b64_compact) >= 16:
        try:
            decoded_bytes = base64.b64decode(b64_compact, validate=True)
        except Exception:
            decoded_bytes = b""
        if decoded_bytes:
            try:
                decoded_text = decoded_bytes.decode("utf-8", errors="strict").strip()
            except UnicodeDecodeError:
                decoded_text = ""
            if decoded_text.startswith(DUNGEONLAB_PREFIX):
                return decoded_text

    raise PulseParseError("不支持的 .pulse 内容：未识别到 Dungeonlab+pulse: 前缀")


def _closest_old_section_time_index(real_duration_value: float) -> int:
    closest_index = 0
    min_diff = float("inf")
    for idx, value in enumerate(OLD_SECTION_TIME_MAP):
        diff = abs(value - real_duration_value)
        if diff < min_diff:
            min_diff = diff
            closest_index = idx
    return closest_index


def _number_like_js(value: float) -> float | int:
    if value.is_integer():
        return int(value)
    return value


def _default_points_json_string() -> str:
    return json.dumps(
        [
            {"anchor": 0, "x": 0, "y": 0.0},
            {"anchor": 0, "x": 1, "y": 0.0},
        ],
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _evenly_sample(
    points: list[tuple[int, float]], target: int
) -> list[tuple[int, float]]:
    if target <= 0:
        return []
    if len(points) <= target:
        return points
    if target == 1:
        return [points[0]]

    last_index = len(points) - 1
    sampled: list[tuple[int, float]] = []
    prev = -1
    for i in range(target):
        idx = round(i * last_index / (target - 1))
        if idx <= prev:
            idx = prev + 1
        if idx > last_index:
            idx = last_index
        sampled.append(points[idx])
        prev = idx
        if prev == last_index:
            break

    if len(sampled) < target:
        sampled.extend([points[-1]] * (target - len(sampled)))
    return sampled


def _ensure_min_points(
    points: list[tuple[int, float]],
    *,
    minimum: int,
) -> list[tuple[int, float]]:
    if minimum <= 0:
        return points
    if not points:
        return [(0, 0.0)] * minimum
    if len(points) >= minimum:
        return points
    return points + [points[-1]] * (minimum - len(points))


def _parse_points_json_string(
    metadata_string: str,
    *,
    section_index: int,
    max_points: int,
    min_points: int,
) -> tuple[int, str]:
    if not metadata_string:
        return 2, _default_points_json_string()

    raw_segments = [seg for seg in metadata_string.split(",") if seg]
    parsed: list[tuple[int, float]] = []
    for idx, segment in enumerate(raw_segments):
        if segment.count("-") != 1:
            continue
        intensity_raw, anchor_raw = segment.split("-", 1)
        intensity = _parse_float_js(
            intensity_raw, field=f"intensity{section_index}[{idx}]"
        )
        anchor = _parse_int_js(anchor_raw, field=f"anchor{section_index}[{idx}]")
        y = (intensity / 100) * 20
        if y < 0:
            y = 0.0
        elif y > 20:
            y = 20.0
        parsed.append((anchor, y))

    if max_points > 0:
        parsed = _evenly_sample(parsed, max_points)
    parsed = _ensure_min_points(parsed, minimum=min_points)

    points_json: list[dict[str, Any]] = []
    for x, (anchor, y) in enumerate(parsed):
        points_json.append({"anchor": anchor, "x": x, "y": y})

    return len(points_json), json.dumps(
        points_json, ensure_ascii=False, separators=(",", ":")
    )


def parse_dungeonlab_pulse(
    raw_pulse_text: str,
    *,
    name: str | None,
    max_sections: int = 3,
    max_points_per_section: int = 30,
    min_points_per_section: int = 2,
) -> dict[str, Any]:
    text = _normalize_pulse_text(raw_pulse_text)
    data_string = text[len(DUNGEONLAB_PREFIX) :]

    wave_name = name or "名称"
    wave_name_en = name or "Name"

    l_value: float | int = 0
    zy_value = 16

    if "=" in data_string:
        header, data_string = data_string.split("=", 1)
        header_values = header.split(",")
        if len(header_values) >= 3:
            l_value = _number_like_js(_parse_float_js(header_values[0], field="L"))
            zy_value = _parse_int_js(header_values[2], field="ZY")

    section_count = 3
    parse_limit = min(max_sections, section_count)

    default_a = 0
    default_b = 20
    default_j = 0
    default_pc = 1
    default_points = _default_points_json_string()

    a_values: list[float | int] = [default_a] * section_count
    b_values: list[float | int] = [default_b] * section_count
    j_values: list[int] = [default_j] * section_count
    pc_values: list[int] = [default_pc] * section_count
    c_values: list[int] = [2] * section_count
    points_values: list[str] = [default_points] * section_count
    jie_values: list[int] = [0] * section_count

    sections = data_string.split("+section+")
    parsed_sections = 0

    for section_data in sections:
        if parsed_sections >= parse_limit:
            break

        if "/" not in section_data:
            continue

        params_part, metadata_string = section_data.split("/", 1)
        params = params_part.split(",")
        if len(params) < 5:
            continue

        frequency_a = _parse_float_js(params[0], field=f"A{parsed_sections}")
        frequency_b = _parse_float_js(params[1], field=f"B{parsed_sections}")
        duration_index = _parse_int_js(
            params[2], field=f"durationIndex{parsed_sections}"
        )
        mode = _parse_int_js(params[3], field=f"PC{parsed_sections}")
        switch_value = _parse_int_js(params[4], field=f"JIE{parsed_sections}")

        if 0 <= duration_index < len(SECTION_TIME_MAP):
            real_duration_value = SECTION_TIME_MAP[duration_index]
        else:
            real_duration_value = 10

        if real_duration_value > 10:
            real_duration_value = 10

        closest_index = _closest_old_section_time_index(real_duration_value)

        a_values[parsed_sections] = _number_like_js(frequency_a)
        b_values[parsed_sections] = _number_like_js(frequency_b)
        j_values[parsed_sections] = closest_index
        pc_values[parsed_sections] = mode
        if parsed_sections > 0:
            jie_values[parsed_sections] = switch_value

        c_value, points_value = _parse_points_json_string(
            metadata_string,
            section_index=parsed_sections,
            max_points=max_points_per_section,
            min_points=min_points_per_section,
        )
        c_values[parsed_sections] = c_value
        points_values[parsed_sections] = points_value

        parsed_sections += 1

    ordered: dict[str, Any] = {}
    for i in range(section_count):
        ordered[f"A{i}"] = a_values[i]
    for i in range(section_count):
        ordered[f"B{i}"] = b_values[i]
    for i in range(section_count):
        ordered[f"C{i}"] = c_values[i]
    for i in range(section_count):
        ordered[f"J{i}"] = j_values[i]
    if section_count >= 2:
        ordered["JIE1"] = jie_values[1]
    if section_count >= 3:
        ordered["JIE2"] = jie_values[2]

    ordered["L"] = l_value
    for i in range(section_count):
        ordered[f"PC{i}"] = pc_values[i]
    ordered["ZY"] = zy_value
    ordered["classic"] = 1
    ordered["defaultName"] = 0
    for i in range(section_count):
        ordered[f"points{i + 1}"] = points_values[i]
    ordered["waveName"] = wave_name
    ordered["waveNameEn"] = wave_name_en

    return ordered


def _timestamp_utc_yyyymmddhhmmss() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%d%H%M%S")


def _default_output_filename() -> str:
    return f"波形聚合-{_timestamp_utc_yyyymmddhhmmss()}.txt"


def _output_paths_for_chunks(
    *,
    output: str | None,
    chunk_count: int,
) -> list[Path]:
    if chunk_count <= 0:
        raise ValueError("chunk_count must be > 0")

    if output is None:
        base = Path(_default_output_filename())
        stem = base.stem
        suffix = base.suffix or ".txt"
        if chunk_count == 1:
            return [base]
        return [Path(f"{stem}-{i:03d}{suffix}") for i in range(1, chunk_count + 1)]

    output_path = Path(output)
    if output_path.exists() and output_path.is_dir():
        output_dir = output_path
        output_dir.mkdir(parents=True, exist_ok=True)
        base = Path(_default_output_filename())
        stem = base.stem
        suffix = base.suffix or ".txt"
        if chunk_count == 1:
            return [output_dir / f"{stem}{suffix}"]
        return [
            output_dir / f"{stem}-{i:03d}{suffix}" for i in range(1, chunk_count + 1)
        ]

    if chunk_count == 1:
        return [output_path]

    suffix = output_path.suffix or ".txt"
    stem = output_path.stem if output_path.suffix else output_path.name
    parent = output_path.parent if output_path.parent != Path("") else Path(".")
    return [parent / f"{stem}-{i:03d}{suffix}" for i in range(1, chunk_count + 1)]


def _iter_pulse_files(paths: list[Path], *, recursive: bool) -> list[Path]:
    found: dict[Path, None] = {}
    for path in paths:
        if path.is_file():
            found[path] = None
            continue
        if path.is_dir():
            iterator = path.rglob("*.pulse") if recursive else path.glob("*.pulse")
            for file_path in sorted(iterator):
                if file_path.is_file():
                    found[file_path] = None
            continue
        raise FileNotFoundError(str(path))
    return list(found.keys())


def _name_from_path(path: Path, *, mode: str) -> str:
    if mode == "first-dot":
        first = path.name.split(".", 1)[0]
        return first or path.stem
    if mode == "stem":
        return path.stem
    raise ValueError(f"unknown name mode: {mode!r}")


def _read_text(path: Path, *, encoding: str) -> str:
    return path.read_text(encoding=encoding)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="批量导入 DG-LAB .pulse 并导出波形聚合 .txt（JSON 数组）"
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="输入 .pulse 文件或目录（目录默认读取其下的 *.pulse；不填则默认读取 pulse/ 目录）",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出 .txt 路径或目录（默认：波形聚合-<UTC时间戳>.txt）",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="递归扫描输入目录",
    )
    parser.add_argument(
        "--skip-invalid",
        action="store_true",
        help="遇到解析失败的文件时跳过（默认：直接报错退出）",
    )
    parser.add_argument(
        "--max-sections",
        type=int,
        default=3,
        help="读取每条波形最多的小节数（默认：3；导出结构固定为 3 小节）",
    )
    parser.add_argument(
        "--max-points-per-section",
        type=int,
        default=30,
        help="每小节最多点数（默认：30；>0 时会降采样到该数量，0 表示不限制）",
    )
    parser.add_argument(
        "--min-points-per-section",
        type=int,
        default=2,
        help="每小节最少点数（默认：2；不足会用末点补齐）",
    )
    parser.add_argument(
        "--name-mode",
        choices=("first-dot", "stem"),
        default="first-dot",
        help="波形名称来源：first-dot=与网页一致（取第一个点之前），stem=去掉最后一个扩展名",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8-sig",
        help="输入文件编码（默认：utf-8-sig，可自动去 BOM）",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="输出格式化 JSON（默认：压缩一行，与网页一致）",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=200,
        help="切片输出：每个 .txt 最多包含的波形数量（默认：200；<=0 表示不切片）",
    )

    args = parser.parse_args(argv)

    output_arg = args.output
    if output_arg is None:
        output_dir = Path(__file__).resolve().parent / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_arg = str(output_dir)

    input_args = args.inputs if args.inputs else ["pulse"]
    input_paths = [Path(p).expanduser() for p in input_args]
    if not args.inputs and not input_paths[0].exists():
        raise SystemExit(
            "默认输入目录 pulse/ 不存在；请创建 pulse/ 并放入 .pulse 文件，或在命令行指定输入路径"
        )
    pulse_files = _iter_pulse_files(input_paths, recursive=args.recursive)
    if not pulse_files:
        raise SystemExit("未找到任何 .pulse 文件")

    waves: list[dict[str, Any]] = []
    errors: list[str] = []

    for pulse_path in pulse_files:
        try:
            raw_text = _read_text(pulse_path, encoding=args.encoding)
            name = _name_from_path(pulse_path, mode=args.name_mode)
            wave = parse_dungeonlab_pulse(
                raw_text,
                name=name,
                max_sections=args.max_sections,
                max_points_per_section=args.max_points_per_section,
                min_points_per_section=args.min_points_per_section,
            )
            waves.append(wave)
        except Exception as exc:
            message = f"{pulse_path}: {exc}"
            if args.skip_invalid:
                errors.append(message)
                continue
            raise SystemExit(message) from exc

    chunk_size = args.chunk_size
    if chunk_size is None or chunk_size <= 0:
        chunks = [waves]
    else:
        chunks = [waves[i : i + chunk_size] for i in range(0, len(waves), chunk_size)]

    output_paths = _output_paths_for_chunks(
        output=output_arg,
        chunk_count=len(chunks),
    )

    for chunk, out_path in zip(chunks, output_paths, strict=True):
        if args.pretty:
            json_text = json.dumps(chunk, ensure_ascii=False, indent=2)
        else:
            json_text = json.dumps(chunk, ensure_ascii=False, separators=(",", ":"))
        out_path.write_text(json_text, encoding="utf-8")

    if len(output_paths) == 1:
        print(f"已导出 {len(waves)} 条波形到: {output_paths[0]}")
    else:
        print(
            f"已导出 {len(waves)} 条波形到 {len(output_paths)} 个文件（每个最多 {chunk_size} 条）："
        )
        for out_path in output_paths:
            print(f"- {out_path}")
    if errors:
        print(f"跳过 {len(errors)} 个文件（--skip-invalid）:", file=sys.stderr)
        for line in errors:
            print(f"- {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
