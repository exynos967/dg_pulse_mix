# dg_pulse_mix

把 `dg_bbs/mix.html`（网页端“DG-LAB 波形聚合”）的导出逻辑移植成 Python 脚本，方便批量导入多个 `.pulse` 并导出聚合 `.txt` 文件（内容为 JSON 数组）。

注：dglab3.0一个app只支持200个自定义波形，如需更多请移步到dglab4，dglab4可直接导入包含pulse波形的zip文件

## 使用方法

- 默认读取 `pulse/` 目录（不填参数）：
  - `python3 pulse_mix.py`
- 默认输出到程序目录的 `output/` 文件夹：
  - `python3 pulse_mix.py`（会在 `output/` 下生成一个或多个 `.txt`）
- 按郊狼 App 导入上限切片导出（App 最多 200 条）：
  - `python3 pulse_mix.py --chunk-size 200 -o out_dir`
- 聚合多个文件/目录（目录默认读取其下的 `*.pulse`）：
  - `python3 pulse_mix.py a.pulse b.pulse some_dir`
- 递归扫描目录：
  - `python3 pulse_mix.py -r some_dir`
- 指定输出文件名：
  - `python3 pulse_mix.py -r some_dir -o 波形聚合.txt`
- 跳过解析失败的文件：
  - `python3 pulse_mix.py -r some_dir --skip-invalid`

更多参数说明：`python3 pulse_mix.py -h`

## 说明

- `.pulse` 内容需要能解析出 `Dungeonlab+pulse:` 前缀（脚本会额外尝试一次 URL 解码 / base64 解码）。
- 输出文件默认名为 `波形聚合-<UTC时间戳>.txt`，结构对齐可正常导入的波形集合 `.txt`（固定 3 小节、`points1~3` 为 JSON 字符串）。
- 单个 `.pulse` 若某小节点数过多，默认会降采样到 30 个点（更贴近 App 可导入的集合文件规模）。
