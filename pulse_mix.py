#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import math
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


def _js_round(value: float) -> int:
    # JS Math.round: round half towards +∞ (can be modeled as floor(x + 0.5))
    return int(math.floor(value + 0.5))


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


def parse_dungeonlab_pulse(
    raw_pulse_text: str,
    *,
    name: str | None,
    max_sections: int = 3,
) -> dict[str, Any]:
    text = _normalize_pulse_text(raw_pulse_text)
    data_string = text[len(DUNGEONLAB_PREFIX) :]

    result: dict[str, Any] = {
        "classic": 1,
        "defaultName": 0,
        "waveName": name or "名称",
        "waveNameEn": name or "Name",
        "L": 0,
        "ZY": 16,
    }

    if "=" in data_string:
        header, data_string = data_string.split("=", 1)
        header_values = header.split(",")
        if len(header_values) >= 3:
            result["L"] = _number_like_js(_parse_float_js(header_values[0], field="L"))
            result["ZY"] = _parse_int_js(header_values[2], field="ZY")

    sections = data_string.split("+section+")
    section_index = 0

    for section_data in sections:
        if section_index >= max_sections:
            break

        if "/" not in section_data:
            continue

        params_part, metadata_string = section_data.split("/", 1)
        params = params_part.split(",")
        if len(params) < 5:
            continue

        frequency_a = _parse_float_js(params[0], field=f"A{section_index}")
        frequency_b = _parse_float_js(params[1], field=f"B{section_index}")
        duration_index = _parse_int_js(params[2], field=f"durationIndex{section_index}")
        mode = _parse_int_js(params[3], field=f"PC{section_index}")
        switch_value = _parse_int_js(params[4], field=f"JIE{section_index}")

        if 0 <= duration_index < len(SECTION_TIME_MAP):
            real_duration_value = SECTION_TIME_MAP[duration_index]
        else:
            real_duration_value = 10

        if real_duration_value > 10:
            real_duration_value = 10

        closest_index = _closest_old_section_time_index(real_duration_value)

        result[f"A{section_index}"] = _number_like_js(frequency_a)
        result[f"B{section_index}"] = _number_like_js(frequency_b)
        result[f"J{section_index}"] = closest_index
        result[f"PC{section_index}"] = mode
        if section_index > 0:
            result[f"JIE{section_index}"] = switch_value

        if metadata_string:
            pulses = metadata_string.split(",")
            result[f"C{section_index}"] = len(pulses)

            metadata_json: list[dict[str, Any]] = []
            for idx, pulse_str in enumerate(pulses):
                if pulse_str.count("-") != 1:
                    continue
                intensity_raw, anchor_type = pulse_str.split("-", 1)
                intensity = _parse_float_js(
                    intensity_raw, field=f"intensity{section_index}[{idx}]"
                )
                mapped_intensity = _js_round((intensity / 100) * 20)
                metadata_json.append(
                    {
                        "anchor": anchor_type,
                        "x": idx,
                        "y": mapped_intensity,
                    }
                )

            result[f"points{section_index + 1}"] = json.dumps(
                metadata_json, ensure_ascii=False, separators=(",", ":")
            )
        else:
            result[f"C{section_index}"] = 0
            result[f"points{section_index + 1}"] = "[]"

        section_index += 1

    return result


def _timestamp_utc_yyyymmddhhmmss() -> str:
    return _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S")


def _default_output_filename() -> str:
    return f"波形聚合-{_timestamp_utc_yyyymmddhhmmss()}.txt"


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
        help="输出 .txt 路径（默认：波形聚合-<UTC时间戳>.txt）",
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
        help="每条波形最多保留的小节数（默认：3，与网页一致）",
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

    args = parser.parse_args(argv)

    input_args = args.inputs if args.inputs else ["pulse"]
    input_paths = [Path(p).expanduser() for p in input_args]
    if not args.inputs and not input_paths[0].exists():
        raise SystemExit(
            "默认输入目录 pulse/ 不存在；请创建 pulse/ 并放入 .pulse 文件，或在命令行指定输入路径"
        )
    pulse_files = _iter_pulse_files(input_paths, recursive=args.recursive)
    if not pulse_files:
        raise SystemExit("未找到任何 .pulse 文件")

    output_path = Path(args.output) if args.output else Path(_default_output_filename())
    if output_path.exists() and output_path.is_dir():
        output_path = output_path / _default_output_filename()

    waves: list[dict[str, Any]] = []
    errors: list[str] = []

    for pulse_path in pulse_files:
        try:
            raw_text = _read_text(pulse_path, encoding=args.encoding)
            name = _name_from_path(pulse_path, mode=args.name_mode)
            wave = parse_dungeonlab_pulse(
                raw_text, name=name, max_sections=args.max_sections
            )
            waves.append(wave)
        except Exception as exc:
            message = f"{pulse_path}: {exc}"
            if args.skip_invalid:
                errors.append(message)
                continue
            raise SystemExit(message) from exc

    if args.pretty:
        json_text = json.dumps(waves, ensure_ascii=False, indent=2)
    else:
        json_text = json.dumps(waves, ensure_ascii=False, separators=(",", ":"))

    output_path.write_text(json_text + "\n", encoding="utf-8")

    print(f"已导出 {len(waves)} 条波形到: {output_path}")
    if errors:
        print(f"跳过 {len(errors)} 个文件（--skip-invalid）:", file=sys.stderr)
        for line in errors:
            print(f"- {line}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
