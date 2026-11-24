# TCM-THGNet · 01-mapping 数据标准化脚本

基于大语言模型（DeepSeek，OpenAI 兼容接口）的**中医多队列临床数据标准化工具**。
通过精心设计的提示词（Prompt），把医院原始数据中的非标准名称自动映射到统一的标准名称，
并输出可审计的映射表，供下游 TCM-THGNet 知识图谱 / 多组学研究直接使用。

## 它能做什么

把多个队列（CKD / RA / MN / LC / BC）原始数据中的**自由文本名称**标准化为**受控术语**：

| 脚本 | 标准化对象 | 主要输出字段 |
|------|-----------|-------------|
| `03_standardize_Disease.py` | 疾病诊断 | 原始诊断、标准名称、类型、疾病系统、是否候选共病 |
| `03_standardize_Drug.py` | 西药/药品 | 原始名称、标准药品名、分类等 |
| `03_standardize_Lab.py` | 实验室检验项目 | 原始名称、通用名、英文简称、分类（LOINC 风格） |
| `03_standardize_Syndrome.py` | 中医证候 | 原始名称、标准证候（291 词表口径）、英文简称 |
| `03_standardize_TCM.py` | 中药 | 原始名称、标准中药名、炮制品、性味归经、功效分类 |

## 工作原理

每个脚本采用「**强制规则 + LLM 标准化 + 后处理校验**」的混合方案：

1. **强制规则预处理**：已知映射写死在本地规则表，命中后直接跳过 API（省钱、稳定）。
2. **LLM 批量标准化**：未命中规则的名称按批次（如每批 15/20 条）组装成提示词，调用
   DeepSeek 大模型输出结构化 JSON（标准名 + 分类 + 简称等）。
3. **后处理校验**：扫描「同一通用名 → 多分类/多简称」等冲突，标记需要人工复核的条目。
4. **统一缓存**：标准化结果跨队列共享并落盘，保证同一原始名称在各队列结果一致，
   重复运行零 API 成本。

## 目录结构

```
TCM-THGNet-01maping/
├── 03_standardize_Disease.py    # 疾病诊断标准化
├── 03_standardize_Drug.py       # 药品标准化
├── 03_standardize_Lab.py        # 检验项目标准化
├── 03_standardize_Syndrome.py   # 中医证候标准化
├── 03_standardize_TCM.py        # 中药标准化
├── requirements.txt             # 依赖清单
├── .env.example                 # 环境变量示例（API Key 配置）
└── README.md
```

## 快速开始

### 1. 安装依赖

```powershell
pip install -r requirements.txt
```

### 2. 配置 API Key（从环境变量读取，切勿写回源码）

```powershell
# PowerShell
$env:DEEPSEEK_API_KEY = "你的 DeepSeek API Key"
```

> 本仓库已把所有硬编码的 API Key 移除，统一改为 `os.environ.get("DEEPSEEK_API_KEY")`。
> 如未设置该环境变量，脚本会因缺少密钥而报错，这是预期行为。

### 3. 运行

```powershell
python 03_standardize_Disease.py
python 03_standardize_Drug.py
python 03_standardize_Lab.py
python 03_standardize_Syndrome.py
python 03_standardize_TCM.py
```

## 配置说明

每个脚本顶部的「配置区」需要按你的本机路径修改：

- `client`：API 客户端，`base_url` 默认 `https://api.deepseek.com`，
  `MODEL_NAME` 为 `deepseek-v4-pro` / `deepseek-v4-flash`。
- `COHORTS` / `QUEUE_CONFIG`：各队列的输入 CSV 与输出映射表（xlsx）路径，
  当前是作者本机的绝对路径（`D:\python\TCM-THGNet\...`），请改成你自己的路径。
- `TEST_MODE` / `USE_CACHE` / `BATCH_SIZE` / `SLEEP_SECONDS` 等运行参数可按需调整。

## 输出

结果写入各队列的映射表（`.xlsx`）与全局标准映射表（`.csv`），例如：

- `map/*_Disease_map.xlsx`、`map/*_Lab_map.xlsx` …
- `map/Disease_global_standard_map.csv`、`map/Lab_global_standard_map.csv` …
- 需要人工复核的条目会单独输出（如 `Lab_manual_review.xlsx`）。

## 注意事项

- **API Key 安全**：请只通过环境变量注入密钥，不要把真实 Key 提交到版本库。
- **数据隐私**：输入数据若含患者信息，上传/分享前请先脱敏。
- **路径**：脚本内为 Windows 绝对路径，跨平台或换机运行需修改配置区。
- **成本**：脚本内置缓存与批处理，重复名称不会重复调用 API，可显著降低成本。
