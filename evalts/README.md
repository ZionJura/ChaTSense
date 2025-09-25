# EvalTS: 时间序列评估工具集 | EvalTS: Time Series Evaluation Toolkit

## 项目简介 | Project Introduction

EvalTS 是一个用于时间序列任务的评估工具集，支持多种任务类型的数据生成与评测，适用于科研与工业场景。

EvalTS is a toolkit for evaluating time series tasks, supporting data generation and evaluation for various task types. It is suitable for both research and industrial applications.

## 主要功能 | Main Features

- 多任务类型支持（如异常检测、趋势变化、极值检测等）
- 任务数据自动生成与评测
- 易于扩展与集成

- Support for multiple task types (e.g., anomaly detection, trend change, extreme value detection, etc.)
- Automatic task data generation and evaluation
- Easy to extend and integrate

## 快速开始 | Quick Start

1. 安装依赖 | Install dependencies
```bash
pip install -r requirements.txt
```
2. 运行示例 | Run example
```bash
python gen_meta.py
```

## 文件结构 | File Structure

- `gen_meta.py`：元数据生成脚本 | Metadata generation script
- `gen_qa.py` / `gen_qa.ipynb`：问答数据生成脚本 | QA data generation script
- `new_task/`：各类任务数据 | Various task data

## 贡献 | Contributing

欢迎提交 issue 和 pull request，共同完善本项目。

Contributions are welcome! Please submit issues and pull requests to help improve this project.

## 许可证 | License

MIT License
