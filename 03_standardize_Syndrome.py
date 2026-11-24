import pandas as pd
import json
import time
import re
import os
from pathlib import Path
from openai import OpenAI


# =========================
# 1. 配置区
# =========================

client = OpenAI(
    api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
    base_url="https://api.deepseek.com")
MODEL_NAME ="deepseek-v4-flash"
# ---- 多队列配置（每个队列独立输入、输出，共享缓存） ----

COHORTS = [
    {
        "name": "CKD",
        "input": r"D:\python\TCM-THGNet\CKD\Syndrome_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\CKD_Syndrome_map.xlsx",
    },
    {
        "name": "RA",
        "input": r"D:\python\TCM-THGNet\RA\Syndrome_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\RA_Syndrome_map.xlsx",
    },
    {
        "name": "MN",
        "input": r"D:\python\TCM-THGNet\MN\Syndrome_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\MN_Syndrome_map.xlsx",
    },
    {
        "name": "LC",
        "input": r"D:\python\TCM-THGNet\LC\Syndrome_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\LC_Syndrome_map.xlsx",
    },
    {
        "name": "BC",
        "input": r"D:\python\TCM-THGNet\BC\Syndrome_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\BC_Syndrome_map.xlsx",
    },
]

CACHE_DIR = Path(r"D:\python\A硕士毕业_RA\cache")
# v2：标准证候口径从「细粒度」升级为「291 词表」，且非中医项输出标准化名称+英文简称；
# 旧缓存（细粒度/非中医证候/N/A）与新口径不兼容，故改用新文件名强制重建。
CACHE_FILE = CACHE_DIR / "Syndrome_map_cache_v2.json"


# =========================
# 2. 测试开关
# =========================

TEST_MODE = False
TEST_N = 20

KEEP_FREQ = False


# =========================
# 3. 缓存开关
# =========================

USE_CACHE = True
SAVE_CACHE = True
CLEAR_CACHE_BEFORE_RUN = False

SLEEP_SECONDS = 0.5


# =========================
# 3.5 标准证候词表（最终证候口径，291）
# =========================

# 最终证候口径清单（291 归并组，列：归并名/成员数/成员）。
# prompt 会把「归并名」作为标准证候词表，让 LLM 直接映射到 291 口径。
CANONICAL_GROUPS_FILE = Path(
    r"D:\python\TCM-THGNet\01maping\map\Syndrome_synonym_groups.csv"
)


# =========================
# 4. 输出字段
# =========================

OUTPUT_COLUMNS = [
    "原始证候",
    "标准证候",
    "简称",
    "分类"
]

REQUIRED_CACHE_FIELDS = [
    "标准证候",
    "简称",
    "分类"
]


# =========================
# 5. 读取文件
# =========================

def read_table_safely(file_path):
    file_path = Path(file_path)
    suffix = file_path.suffix.lower()

    if suffix in [".xlsx", ".xls"]:
        return pd.read_excel(file_path)

    if suffix == ".csv":
        for enc in ["utf-8-sig", "utf-8", "gbk"]:
            try:
                return pd.read_csv(file_path, encoding=enc)
            except UnicodeDecodeError:
                continue
        raise ValueError("无法识别 CSV 编码，请检查文件编码。")

    raise ValueError(f"不支持的文件格式：{suffix}，请使用 CSV 或 Excel。")


# =========================
# 6. 缓存函数
# =========================

def clean_cache_key(name):
    if pd.isna(name):
        return ""

    name = str(name).strip()
    name = name.replace(" ", "")
    name = name.replace("　", "")
    name = re.sub(r"[，,。；;：:！!？?、\[\]【】（）()《》<>\"']", "", name)
    return name


def load_cache(cache_file):
    cache_path = Path(cache_file)

    if CLEAR_CACHE_BEFORE_RUN:
        print("已开启 CLEAR_CACHE_BEFORE_RUN，本次运行不读取旧缓存。")
        return {}

    if not cache_path.exists():
        print("未发现缓存文件，将创建新缓存。")
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            print("缓存文件格式异常，将重新创建缓存。")
            return {}

        print(f"已读取缓存，共 {len(cache)} 条记录。")
        return cache

    except Exception as e:
        print(f"读取缓存失败，将重新创建缓存：{e}")
        return {}


def save_cache(cache, cache_file):
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # 先写入同目录临时文件，再原子替换，避免程序中断时损坏整个缓存。
    temp_path = cache_path.with_name(
        f".{cache_path.name}.{os.getpid()}.tmp"
    )
    try:
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, cache_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


ALLOWED_CATEGORIES = {"证候", "疾病", "症状", "其他"}


def is_cache_item_complete(item):
    if not isinstance(item, dict):
        return False

    for field in REQUIRED_CACHE_FIELDS:
        value = str(item.get(field, "")).strip()
        if value == "" or value.lower() in ["nan", "none", "null"]:
            return False

    std = str(item.get("标准证候", "")).strip()
    abbr = str(item.get("简称", "")).strip()
    category = str(item.get("分类", "")).strip()

    if std == "不确定" or category not in ALLOWED_CATEGORIES:
        return False

    # 证候类：简称必须是英文简称（不能是 N/A）。
    # 疾病/症状/其他：标准名称 + 英文简称 均已在上方校验非空即可。
    if category == "证候" and abbr.upper() == "N/A":
        return False

    return True


# =========================
# 7. JSON 解析函数
# =========================

def extract_json(content):
    if not isinstance(content, str):
        raise ValueError("API 返回内容不是字符串")

    content = content.strip()

    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if match:
        content = match.group(1).strip()

    match = re.search(r"\{.*\}", content, re.S)
    if match:
        content = match.group(0).strip()

    return json.loads(content)


def extract_json_array(content):
    """
    解析 API 返回的 JSON 数组。
    支持 ```json ... ``` 包裹、纯数组、或数组嵌在文本中。
    """
    if not isinstance(content, str):
        raise ValueError("API 返回内容不是字符串")

    content = content.strip()

    # 优先解析 Markdown 代码块中的内容。
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S | re.I)
    if match:
        content = match.group(1).strip()

    try:
        data = json.loads(content)
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass

    # 若数组嵌在说明文字中，使用 JSONDecoder 定位第一个合法数组，
    # 避免贪婪正则把多个方括号之间的文本错误拼接起来。
    decoder = json.JSONDecoder()
    for index, char in enumerate(content):
        if char != "[":
            continue
        try:
            data, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(data, list):
            return data

    raise ValueError("API 返回内容中未找到合法 JSON 数组")


def validate_batch_results(data, names):
    """严格校验批量 API 结果，并按输入序号重建结果顺序。"""
    if not isinstance(data, list):
        raise ValueError("API 返回结果不是 JSON 数组")

    if len(data) != len(names):
        raise ValueError(
            f"API 返回数量不一致：期望 {len(names)} 条，实际 {len(data)} 条"
        )

    required_fields = {"序号", "原始证候", "标准证候", "简称", "分类"}
    indexed_results = {}

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("API 返回数组中存在非对象元素")

        missing_fields = required_fields - set(item)
        if missing_fields:
            raise ValueError(
                f"API 返回元素缺少字段：{', '.join(sorted(missing_fields))}"
            )

        sequence_value = item.get("序号")
        if isinstance(sequence_value, bool):
            raise ValueError(f"无效序号：{sequence_value}")
        if isinstance(sequence_value, float) and not sequence_value.is_integer():
            raise ValueError(f"无效序号：{sequence_value}")
        if isinstance(sequence_value, str) and not sequence_value.strip().isdigit():
            raise ValueError(f"无效序号：{sequence_value}")
        try:
            sequence = int(sequence_value)
        except (TypeError, ValueError) as e:
            raise ValueError(f"无效序号：{sequence_value}") from e

        if sequence < 1 or sequence > len(names):
            raise ValueError(f"序号超出范围：{sequence}")
        if sequence in indexed_results:
            raise ValueError(f"API 返回重复序号：{sequence}")

        expected_raw = str(names[sequence - 1]).strip()
        returned_raw = str(item.get("原始证候", "")).strip()
        if returned_raw != expected_raw:
            raise ValueError(
                f"序号 {sequence} 的原始证候不一致："
                f"期望 '{expected_raw}'，实际 '{returned_raw}'"
            )

        std = str(item.get("标准证候", "")).strip()
        abbr = str(item.get("简称", "")).strip()
        category = str(item.get("分类", "")).strip()

        if not std or not abbr or not category:
            raise ValueError(f"序号 {sequence} 存在空字段")
        if category not in ALLOWED_CATEGORIES:
            raise ValueError(f"序号 {sequence} 的分类无效：{category}")

        if category == "证候" and abbr.upper() == "N/A":
            raise ValueError(
                f"序号 {sequence} 的证候简称不能为 N/A"
            )

        indexed_results[sequence] = {
            "原始证候": expected_raw,
            "标准证候": std,
            "简称": abbr,
            "分类": category
        }

    expected_sequences = set(range(1, len(names) + 1))
    if set(indexed_results) != expected_sequences:
        raise ValueError("API 返回序号不完整")

    return [indexed_results[i] for i in range(1, len(names) + 1)]


# =========================
# 8. 文本清洗（已精简：API 直接返回结果，不做额外清洗）
# =========================


# ===================================================================
# 8.5 A 层：强制规则预检
# ===================================================================

def normalize_for_force_map(name):
    """
    清洗名称用于强制规则匹配。
    比 clean_cache_key 更激进：去括号及括号内容，提高命中率。
    例如："湿热痹阻(活动期)" -> "湿热痹阻"
          "肝肾亏虚（缓解期）" -> "肝肾亏虚"
    """
    if pd.isna(name):
        return ""
    name = str(name).strip()
    # 去空格（全角+半角）
    name = name.replace(" ", "").replace("　", "")
    # 去括号及括号内内容
    name = re.sub(r"[（(][^）)]*[）)]", "", name)
    # 统一全半角括号（处理残留单边括号）
    name = name.replace("（", "").replace("）", "").replace("(", "").replace(")", "")
    # 去多余标点
    name = name.strip("，,。；;：:！!？?、")
    return name


# ---- 确定性非证候清单（标准名称 -> (分类, 英文简称)） ----
# 症状、疾病、治法和无意义文本只做完整词匹配，避免“口干”“无”等
# 子串把“气阴两虚伴口干”“表实无汗证”等复合证候整体误判。
FORCE_NON_TCM_TERMS = {
    # 症状
    "水肿": ("症状", "Edema"),
    "胸痛": ("症状", "Chest Pain"),
    "头痛": ("症状", "Headache"),
    "腹痛": ("症状", "Abdominal Pain"),
    "腰痛": ("症状", "Low Back Pain"),
    "关节痛": ("症状", "Arthralgia"),
    "关节肿胀": ("症状", "Joint Swelling"),
    "发热": ("症状", "Fever"),
    "咳嗽": ("症状", "Cough"),
    "咳痰": ("症状", "Expectoration"),
    "气喘": ("症状", "Dyspnea"),
    "乏力": ("症状", "Fatigue"),
    "纳差": ("症状", "Poor Appetite"),
    "失眠": ("症状", "Insomnia"),
    "晨僵": ("症状", "Morning Stiffness"),
    "麻木": ("症状", "Numbness"),
    "怕冷": ("症状", "Aversion to Cold"),
    "盗汗": ("症状", "Night Sweating"),
    "自汗": ("症状", "Spontaneous Sweating"),
    "口干": ("症状", "Dry Mouth"),
    "口苦": ("症状", "Bitter Taste"),
    # 疾病
    "高血压": ("疾病", "Hypertension"),
    "糖尿病": ("疾病", "Diabetes Mellitus"),
    "冠心病": ("疾病", "Coronary Heart Disease"),
    "类风湿关节炎": ("疾病", "Rheumatoid Arthritis"),
    "类风湿": ("疾病", "Rheumatoid Arthritis"),
    "系统性红斑狼疮": ("疾病", "Systemic Lupus Erythematosus"),
    "干燥综合征": ("疾病", "Sjögren's Syndrome"),
    "强直性脊柱炎": ("疾病", "Ankylosing Spondylitis"),
    "骨关节炎": ("疾病", "Osteoarthritis"),
    "痛风": ("疾病", "Gout"),
    "慢性肾炎": ("疾病", "Chronic Nephritis"),
    "肾病综合征": ("疾病", "Nephrotic Syndrome"),
    "慢性肾衰竭": ("疾病", "Chronic Renal Failure"),
    # 治法
    "益气养阴": ("其他", "Supplement Qi and Nourish Yin"),
    "活血化瘀": ("其他", "Activate Blood and Resolve Stasis"),
    "清热解毒": ("其他", "Clear Heat and Resolve Toxins"),
    "祛风除湿": ("其他", "Expel Wind and Remove Dampness"),
    "温阳散寒": ("其他", "Warm Yang and Disperse Cold"),
    "健脾利湿": ("其他", "Invigorate Spleen and Drain Dampness"),
    "疏肝理气": ("其他", "Soothe Liver and Regulate Qi"),
    "补肾壮骨": ("其他", "Tonify Kidney and Strengthen Bones"),
    "化痰通络": ("其他", "Resolve Phlegm and Unblock Collaterals"),
    "清热利湿": ("其他", "Clear Heat and Drain Dampness"),
    # 无意义文本
    "无": ("其他", "N/A"),
    "未见": ("其他", "N/A"),
    "未见病人": ("其他", "N/A"),
    "未见患者": ("其他", "N/A"),
    "未查": ("其他", "N/A"),
    "不详": ("其他", "N/A"),
    "缺失": ("其他", "N/A"),
    "正常": ("其他", "Normal"),
}

# 检验指标允许子串匹配（关键词 -> 英文简称），以识别“肌酐升高”“尿蛋白阳性”等写法。
FORCE_NON_TCM_INDICATOR_KEYWORDS = {
    "肌酐": "Creatinine Abnormality",
    "尿素氮": "BUN Abnormality",
    "尿酸": "Uric Acid Abnormality",
    "尿蛋白": "Proteinuria",
    "血沉": "ESR Abnormality",
    "C反应蛋白": "CRP Abnormality",
    "类风湿因子": "Rheumatoid Factor Abnormality",
    "抗CCP": "Anti-CCP Abnormality",
    "血红蛋白": "Hemoglobin Abnormality",
    "白细胞": "WBC Abnormality",
    "血小板": "Platelet Abnormality",
    "转氨酶": "Transaminase Abnormality",
    "胆红素": "Bilirubin Abnormality",
    "血糖": "Blood Glucose Abnormality",
    "糖化血红蛋白": "HbA1c Abnormality",
}

# ---- 人工覆盖：LLM 越界输出（不在 291 词表）的确定性纠正 ----
# 键为原始名（精确匹配），优先于 force map 与其余规则。
FORCE_OVERRIDE = {
    "心阴虚证": {"标准证候": "阴虚证", "简称": "Yin Deficiency Pattern", "分类": "证候"},
    "正虚毒炽": {"标准证候": "毒证", "简称": "Toxin Pattern", "分类": "证候"},
    "正虚毒留证": {"标准证候": "毒证", "简称": "Toxin Pattern", "分类": "证候"},
    "肝郁证": {"标准证候": "气郁证", "简称": "Qi Depression Pattern", "分类": "证候"},
    "肝郁(证)": {"标准证候": "气郁证", "简称": "Qi Depression Pattern", "分类": "证候"},
    # 同义合并（同病/同症状，标准名归一，消除同简称多标准）：痹=痹病、尿血=血尿
    "痹": {"标准证候": "痹病", "简称": "Bi Disease", "分类": "疾病"},
    "尿血": {"标准证候": "血尿", "简称": "Hematuria", "分类": "症状"},
}

# ---- 简称唯一化：同一英文简称对应多个标准证候时，强制一一对应 ----
# 键 = 标准证候，值 = 唯一英文简称（仅列出需要改的；未列出的保持原样）。
ABBR_OVERRIDE = {
    "瘀证": "Stasis Pattern",
    "气滞瘀证": "Qi Stagnation and Stasis Pattern",
    "水饮证": "Fluid-Rheum Retention Pattern",
}

# ---- 已知精确映射表 ----
# 初始化时从已有缓存加载，运行时动态更新
FORCE_EXACT_MAP = {}   # key: 规范化名称, value: {"标准证候": ..., "简称": ..., "分类": ...}


def build_force_map_from_cache(cache):
    """
    从已有缓存中提取所有完整确认的映射，作为 A 层精确匹配规则。
    只有字段完整、语义一致且不是"不确定"的记录才进入 force map。
    """
    force_map = {}
    ambiguous_keys = set()
    for cache_key, item in cache.items():
        if not is_cache_item_complete(item):
            continue
        std = str(item.get("标准证候", "")).strip()
        if std in ["不确定", ""]:
            continue
        source_name = str(item.get("原始证候", cache_key)).strip() or cache_key
        force_key = clean_cache_key(normalize_for_force_map(source_name))
        if not force_key or force_key in ambiguous_keys:
            continue

        record = {
            "标准证候": std,
            "简称": str(item.get("简称", "")).strip(),
            "分类": str(item.get("分类", "")).strip()
        }

        # 多个原始写法清洗成同一个 key 时，仅保留完全一致的映射；
        # 若映射冲突，则回退到原始 cache key 查询，不做强制覆盖。
        if force_key in force_map and force_map[force_key] != record:
            ambiguous_keys.add(force_key)
            force_map.pop(force_key, None)
            continue
        force_map[force_key] = record

    print(
        f"从缓存构建 force_map，共 {len(force_map)} 条无冲突映射，"
        f"跳过 {len(ambiguous_keys)} 个冲突 key。"
    )
    return force_map


def check_force_rules(raw_name):
    """
    A 层：强制规则检查。
    命中 → 直接返回结果（不调 API）。
    未命中 → 返回 None，进入 B 层。

    匹配优先级：
    ⓪ 精确匹配 FORCE_OVERRIDE（人工覆盖，越界纠正）
    ① 精确匹配 FORCE_EXACT_MAP（经 normalize_for_force_map 清洗后）
    ② 完整词命中确定性非证候清单 → 按疾病/症状/其他分类
    ③ 检验指标关键词命中 → 标准化名称 / 英文简称 / 其他
    """
    if pd.isna(raw_name) or str(raw_name).strip() == "":
        return {"原始证候": "", "标准证候": "", "简称": "", "分类": ""}

    raw_name = str(raw_name).strip()

    # ⓪ 人工覆盖（越界纠正，最高优先级）
    if raw_name in FORCE_OVERRIDE:
        record = FORCE_OVERRIDE[raw_name]
        result = {
            "原始证候": raw_name,
            "标准证候": record["标准证候"],
            "简称": record["简称"],
            "分类": record["分类"]
        }
        print(f"  A 层命中（人工覆盖）：{raw_name} -> {result['标准证候']}")
        return result

    norm_name = normalize_for_force_map(raw_name)

    # ① 精确匹配 force map
    force_key = clean_cache_key(norm_name)
    if force_key in FORCE_EXACT_MAP:
        record = FORCE_EXACT_MAP[force_key]
        result = {
            "原始证候": raw_name,
            "标准证候": record["标准证候"],
            "简称": record.get("简称", ""),
            "分类": record.get("分类", "")
        }
        print(f"  A 层命中（精确）：{raw_name} -> {result['标准证候']}")
        return result

    # ② 症状、疾病、治法及无意义文本：只做完整词匹配。
    if norm_name in FORCE_NON_TCM_TERMS:
        category, abbr = FORCE_NON_TCM_TERMS[norm_name]
        result = {
            "原始证候": raw_name,
            "标准证候": norm_name,
            "简称": abbr,
            "分类": category
        }
        print(f"  A 层命中（非证候）：{raw_name} -> {norm_name}（{category}）")
        return result

    # ③ 检验指标允许子串匹配。
    for keyword, abbr in FORCE_NON_TCM_INDICATOR_KEYWORDS.items():
        if keyword in norm_name:
            result = {
                "原始证候": raw_name,
                "标准证候": norm_name,
                "简称": abbr,
                "分类": "其他"
            }
            print(
                f"  A 层命中（检验指标）：{raw_name} -> {norm_name}"
                f"（关键词: {keyword}）"
            )
            return result

    # 未命中，进入 B 层
    return None


# ===================================================================
# 9. B 层：批量 API 调用
# ===================================================================

# 批量大小
BATCH_SIZE = 20

# 指数退避重试参数
MAX_RETRIES = 3
BASE_RETRY_DELAY = 2   # 秒


def build_batch_prompt(names):
    """
    构造批量 prompt：一步完成「证候精简（映射到 291 词表）+ 英文简称 + 非中医标准化」。
    标准证候直接取 291 词表中的归并名，不再单独做后处理归并。
    """
    names_list = "\n".join([f"{i+1}. {n}" for i, n in enumerate(names)])
    canonical_names = get_canonical_names()
    canonical_list = "、".join(canonical_names)

    prompt = f"""
你是一名严谨的中医证候术语标准化专家。

请对以下 {len(names)} 个原始证候名称逐一标准化：把同义/兼夹/病位扩展写法归并到标准证候词表，并给出英文简称。

严格返回 JSON 数组，不要使用 Markdown，不要输出解释文字。

返回格式（每个元素5个字段）：
[
  {{"序号": 1, "原始证候": "xxx", "标准证候": "xxx", "简称": "xxx", "分类": "证候"}},
  ...
]

====================
待标准化名称列表：
====================
{names_list}

====================
一、标准证候词表（最终口径，共 {len(canonical_names)} 个）
====================

「标准证候」必须从以下词表中选取，不得自创新证候名：

{canonical_list}

====================
二、证候精简（归并）规则
====================

1. 同义写法、病历简写、错别字，统一为词表中最规范的一个。
   例：肝肾不足/肝肾两虚 -> 肝肾亏虚证；痰淤 -> 痰瘀。

2. 「脏腑+虚 + 兼夹病邪」归并为「脏腑+虚」：
   例：脾肾气虚血瘀/脾肾气虚湿浊 -> 脾肾气虚证。

3. 「复合虚 + 兼夹」归并为「复合虚」：
   例：气阴两虚夹湿热 -> 气阴两虚证。

4. 病位/状态修饰词（下注、痹阻、阻络、内蕴、犯肺、上扰、蒙窍、夹湿、兼夹等）去掉，只留病性核心：
   例：脾胃湿热/湿热下注/湿热痹阻 -> 湿热证。

5. 瘀/血瘀为核心病机，词表中含瘀的复合证候必须保留瘀：
   例：湿热瘀阻 -> 湿热瘀证；气虚血瘀 -> 气虚血瘀证；痰瘀阻络 -> 痰瘀证。

6. 六经证、经典证（太阳/阳明/少阳/太阴/少阴/厥阴、肝阳上亢、心肾不交、脱证、亡阳等）保留原名。

7. 若原始名称已是词表中的证候，保持不变。

====================
三、"简称"字段规则
====================

「简称」为英文规范简称，采用 WHO 中医术语国际标准（IST）或公认学术译法。

命名规则：
- 使用英文名词短语，首字母大写
- 病性在前，病位在后
- 多个病机用 and 连接

示例：
- 湿热证 -> Dampness-Heat Pattern
- 湿热瘀证 -> Dampness-Heat and Blood Stasis
- 气虚血瘀证 -> Qi Deficiency and Blood Stasis
- 气阴两虚证 -> Dual Deficiency of Qi and Yin
- 肝肾亏虚证 -> Liver and Kidney Deficiency
- 肝阳上亢证 -> Ascendant Hyperactivity of Liver Yang
- 心肾不交证 -> Heart-Kidney Disharmony

====================
四、非中医证候处理
====================

若输入是症状、体征、疾病、检验指标、药物、治法或无意义文本，不要强行为证候：

- 「标准证候」填标准化名称（清理括号/错别字后的规范名，如"高血压"、"水肿"、"肌酐升高"、"益气养阴"；无意义文本填"无"/"未见"/"不详"等本身）。
- 「简称」填对应英文（如"高血压" -> Hypertension；"水肿" -> Edema；"肌酐升高" -> Elevated Creatinine；"益气养阴" -> Supplement Qi and Nourish Yin；无意义文本填 N/A）。
- 「分类」填 疾病/症状/其他。

====================
五、"分类"字段规则
====================

「分类」只能从以下四个值中选一个（不允许其他值）：

1. 证候：输入是中医证候名称
2. 疾病：输入是西医或中医疾病诊断
3. 症状：输入是症状、体征
4. 其他：输入是检验指标、药物、治法、无意义文本等

====================
六、输出要求
====================

1. 只返回合法 JSON 数组。
2. 「标准证候」：证候类必须取自词表；非中医类填标准化名称。
3. 「简称」：证候类填 IST 英文简称；非中医类填对应英文；不得为空。
4. 数组长度必须等于 {len(names)}，每个元素必须包含"序号"、"原始证候"、"标准证候"、"简称"、"分类"五个字段。
"""
    return prompt


def call_batch_api(names, cache):
    """
    批量调用 API，带指数退避重试。
    返回：dict {raw_name: result_dict}
    """
    prompt = build_batch_prompt(names)

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model= MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "你是一名严谨的中医证候术语标准化专家。"
                            "你的任务是根据国家标准、行业规范、权威教材和临床常用名称，"
                            "将原始证候名称映射为标准通用证候名称。"
                            "不能机械补字，不能编造证候。只返回合法 JSON 数组。"
                        )
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False
            )

            content = response.choices[0].message.content
            data = extract_json_array(content)
            validated_items = validate_batch_results(data, names)

            # 只有整批结果全部通过校验后，才允许写入缓存。
            results = {}
            pending_cache = {}
            for result in validated_items:
                raw = result["原始证候"]
                results[raw] = result

                cache_key = clean_cache_key(raw)
                comparable = {
                    "标准证候": result["标准证候"],
                    "简称": result["简称"],
                    "分类": result["分类"]
                }
                # 同批内 / 与已有缓存 的 key 冲突：不再中断整批。
                # 保留首个结果写入缓存，冲突结果仍进入 results 交由 C 层投票统一。
                if (
                    cache_key in pending_cache
                    and pending_cache[cache_key]["comparable"] != comparable
                ):
                    print(
                        f"⚠ 批次内缓存 key '{cache_key}' 映射冲突，"
                        f"保留首个结果，交由 C 层投票。"
                    )
                    continue

                existing_item = cache.get(cache_key)
                if is_cache_item_complete(existing_item):
                    existing_comparable = {
                        "标准证候": str(existing_item.get("标准证候", "")).strip(),
                        "简称": str(existing_item.get("简称", "")).strip(),
                        "分类": str(existing_item.get("分类", "")).strip()
                    }
                    if existing_comparable != comparable:
                        print(
                            f"⚠ 缓存 key '{cache_key}' 新旧映射冲突，"
                            f"保留现有缓存，新结果交由 C 层投票。"
                        )
                        continue
                pending_cache[cache_key] = {
                    "result": result,
                    "comparable": comparable
                }

            if SAVE_CACHE:
                for cache_key, pending in pending_cache.items():
                    cache[cache_key] = pending["result"]
                save_cache(cache, CACHE_FILE)

            print(
                f"    批次 API 调用并校验成功"
                f"（{len(results)}/{len(names)} 条结果）"
            )
            return results

        except Exception as e:
            delay = BASE_RETRY_DELAY ** attempt   # 2, 4, 8 秒
            print(f"    批次 API 失败（第 {attempt}/{MAX_RETRIES} 次）：{e}")
            if attempt < MAX_RETRIES:
                print(f"    等待 {delay}s 后重试...")
                time.sleep(delay)
            else:
                raise RuntimeError(
                    "批次 API 连续失败，已停止运行；"
                    "未生成包含“不确定”占位值的正式结果。"
                ) from e


def get_batch_syndrome_standards(names, cache):
    """
    B 层入口：对所有 A 层未命中的名称做批量 API 调用。
    先检查缓存，只对未缓存的调 API。
    返回：dict {raw_name: result_dict}
    """
    results = {}
    uncached = []

    for name in names:
        if pd.isna(name) or str(name).strip() == "":
            continue

        raw = str(name).strip()
        cache_key = clean_cache_key(raw)

        # 查缓存
        if USE_CACHE and cache_key in cache:
            item = cache[cache_key]
            if is_cache_item_complete(item):
                results[raw] = {
                    "原始证候": raw,
                    "标准证候": item.get("标准证候", ""),
                    "简称": item.get("简称", ""),
                    "分类": item.get("分类", "")
                }
                print(f"    缓存命中：{raw} -> {item.get('标准证候', '')}")
                continue

        uncached.append(raw)

    # 对未缓存的调用批量 API
    if uncached:
        total_batches = (len(uncached) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"  {len(names)} 条中，{len(uncached)} 条未命中缓存，分 {total_batches} 批调 API...")
        for i in range(0, len(uncached), BATCH_SIZE):
            batch = uncached[i:i + BATCH_SIZE]
            print(f"  发送批次 {i // BATCH_SIZE + 1}/{total_batches}（{len(batch)} 条）...")
            batch_results = call_batch_api(batch, cache)
            results.update(batch_results)
            time.sleep(SLEEP_SECONDS)

    return results


# ===================================================================
# 9.5 C 层：自愈校验
# ===================================================================

def self_healing_verify(all_results, cache, raw_weights=None):
    """
    C 层：保守检查同一 clean_cache_key 下的映射一致性。

    - 已确认结果可覆盖同 key 下的空值或“不确定”；
    - 多个已确认结果冲突时，只在频数权重形成严格多数时自动修正；
    - 平票或没有严格多数时只报告，不自动覆盖；
    - 简称冲突只报告，避免用可能错误的英文翻译反向污染中文证候。
    """
    print("\n" + "=" * 50)
    print("C 层：保守一致性校验")
    print("=" * 50)

    raw_weights = raw_weights or {}
    changes_made = 0
    conflicts_found = 0
    unresolved_conflicts = 0

    def get_weight(raw_name):
        try:
            weight = float(raw_weights.get(raw_name, 1.0))
        except (TypeError, ValueError):
            return 1.0
        if pd.isna(weight) or weight <= 0 or weight == float("inf"):
            return 1.0
        return weight

    # key -> [(index, raw_name, std_name, weight)]
    key_to_results = {}
    for index, result in enumerate(all_results):
        raw = str(result.get("原始证候", "")).strip()
        std = str(result.get("标准证候", "")).strip()
        key = clean_cache_key(raw)
        if not key:
            continue
        key_to_results.setdefault(key, []).append(
            (index, raw, std, get_weight(raw))
        )

    for key, entries in key_to_results.items():
        confirmed_entries = [
            entry for entry in entries if entry[2] not in ["", "不确定"]
        ]
        uncertain_entries = [
            entry for entry in entries if entry[2] in ["", "不确定"]
        ]

        if not confirmed_entries:
            continue

        std_weights = {}
        for _, _, std, weight in confirmed_entries:
            std_weights[std] = std_weights.get(std, 0.0) + weight

        if len(std_weights) == 1:
            selected_std = next(iter(std_weights))
            # 即使标准证候相同，也检查简称和分类是否一致。
            needs_correction = len(entries) > 1 or bool(uncertain_entries)
        else:
            conflicts_found += 1
            total_weight = sum(std_weights.values())
            selected_std, selected_weight = max(
                std_weights.items(), key=lambda item: item[1]
            )

            print(f"\nKey 冲突：key='{key}'")
            for std, weight in sorted(
                std_weights.items(), key=lambda item: item[1], reverse=True
            ):
                print(f"  {std}: 权重 {weight:g}")

            # 必须超过全部已确认权重的一半，平票时不做任何自动修改。
            if selected_weight <= total_weight / 2:
                unresolved_conflicts += 1
                print("  未形成严格多数，已保留原结果并标记人工复核。")
                continue
            needs_correction = True

        if not needs_correction:
            continue

        canonical_candidates = [
            entry for entry in confirmed_entries if entry[2] == selected_std
        ]
        canonical_index, _, _, _ = max(
            canonical_candidates, key=lambda entry: entry[3]
        )
        canonical = all_results[canonical_index]
        canonical_fields = {
            "标准证候": str(canonical.get("标准证候", "")).strip(),
            "简称": str(canonical.get("简称", "")).strip(),
            "分类": str(canonical.get("分类", "")).strip()
        }

        if not is_cache_item_complete(canonical_fields):
            unresolved_conflicts += 1
            print(f"  key='{key}' 的多数结果字段不完整，跳过自动修正。")
            continue

        for index, raw, std, _ in entries:
            current = all_results[index]
            current_fields = {
                "标准证候": str(current.get("标准证候", "")).strip(),
                "简称": str(current.get("简称", "")).strip(),
                "分类": str(current.get("分类", "")).strip()
            }
            if current_fields == canonical_fields:
                continue

            all_results[index] = {
                "原始证候": raw,
                **canonical_fields
            }
            print(
                f"  修正：'{raw}' : '{std}' -> "
                f"'{canonical_fields['标准证候']}'（简称、分类同步）"
            )
            changes_made += 1

            if SAVE_CACHE:
                cache_key = clean_cache_key(raw)
                cache[cache_key] = all_results[index].copy()

    if SAVE_CACHE and changes_made > 0:
        save_cache(cache, CACHE_FILE)

    print("\n--- 简称一致性检查（仅报告） ---")
    abbr_to_entries = {}
    for result in all_results:
        abbr = str(result.get("简称", "")).strip()
        if not abbr or abbr.upper() == "N/A":
            continue
        abbr_to_entries.setdefault(abbr, []).append(
            (
                str(result.get("原始证候", "")).strip(),
                str(result.get("标准证候", "")).strip()
            )
        )

    abbr_conflicts = 0
    for abbr, entries in abbr_to_entries.items():
        standards = sorted({std for _, std in entries if std})
        if len(standards) <= 1:
            continue
        abbr_conflicts += 1
        print(
            f"  简称 '{abbr}' 对应多个标准证候："
            f"{' | '.join(standards)}；未自动修改。"
        )

    print(
        f"\n一致性校验完成：Key 冲突 {conflicts_found} 处，"
        f"未决冲突 {unresolved_conflicts} 处，简称冲突 {abbr_conflicts} 处，"
        f"自动修正 {changes_made} 条。"
    )

    # ---- 逆向简称一致性：同标准证候 → 不同简称/分类（自动修正） ----
    print("\n--- 标准证候简称一致性检查（自动修正） ---")
    std_to_entries = {}   # 标准证候 -> [{index, raw, 简称, 分类, weight}]
    for i, result in enumerate(all_results):
        std = str(result.get("标准证候", "")).strip()
        if std in ["", "不确定"]:
            continue
        abbr = str(result.get("简称", "")).strip()
        cat = str(result.get("分类", "")).strip()
        raw = str(result.get("原始证候", "")).strip()
        if not abbr:
            continue
        std_to_entries.setdefault(std, []).append({
            "index": i, "raw": raw,
            "简称": abbr, "分类": cat,
            "weight": get_weight(raw)
        })

    std_abbr_conflicts = 0
    std_abbr_changes = 0
    for std, entries in std_to_entries.items():
        if len(entries) < 2:
            continue

        # 简称多数投票
        abbr_weights = {}
        for e in entries:
            abbr_weights[e["简称"]] = abbr_weights.get(e["简称"], 0.0) + e["weight"]
        if len(abbr_weights) <= 1:
            # 简称一致，但检查分类
            pass
        else:
            total_w = sum(abbr_weights.values())
            majority_abbr, maj_abbr_w = max(abbr_weights.items(), key=lambda x: x[1])
            if maj_abbr_w <= total_w / 2:
                # 平票，不自动修
                pass
            else:
                std_abbr_conflicts += 1
                print(f"\n标准证候 '{std}' 对应 {len(abbr_weights)} 种简称：")
                for a, w in abbr_weights.items():
                    marker = " ← 多数" if a == majority_abbr else " ← 修正"
                    print(f"  {a}: 权重 {w:g}{marker}")
                for e in entries:
                    if e["简称"] != majority_abbr:
                        old_abbr = all_results[e["index"]]["简称"]
                        all_results[e["index"]]["简称"] = majority_abbr
                        print(f"  修正：'{e['raw']}' : 简称 '{old_abbr}' -> '{majority_abbr}'")
                        std_abbr_changes += 1

        # 分类多数投票（同一标准证候下分类也应一致）
        cat_weights = {}
        for e in entries:
            cat_weights[e["分类"]] = cat_weights.get(e["分类"], 0.0) + e["weight"]
        if len(cat_weights) > 1:
            total_w = sum(cat_weights.values())
            majority_cat, maj_cat_w = max(cat_weights.items(), key=lambda x: x[1])
            if maj_cat_w > total_w / 2:
                for e in entries:
                    if e["分类"] != majority_cat and e["分类"]:
                        old_cat = all_results[e["index"]]["分类"]
                        all_results[e["index"]]["分类"] = majority_cat
                        print(f"  修正：'{e['raw']}' : 分类 '{old_cat}' -> '{majority_cat}'")
                        std_abbr_changes += 1

    if SAVE_CACHE and std_abbr_changes > 0:
        for i, result in enumerate(all_results):
            key = clean_cache_key(str(result.get("原始证候", "")))
            if key:
                cache[key] = result.copy()
        save_cache(cache, CACHE_FILE)

    if std_abbr_conflicts > 0 or std_abbr_changes > 0:
        print(
            f"\n标准证候简称一致性：发现 {std_abbr_conflicts} 处简称不一致，"
            f"自动修正 {std_abbr_changes} 条。"
        )
    else:
        print("  未发现简称不一致。")
    return all_results


# =========================
# 10. 自动识别输入列
# =========================

def detect_columns(df):
    name_col = None
    freq_col = None

    for col in df.columns:
        col_str = str(col).strip()
        col_lower = col_str.lower()

        if name_col is None and (
            "原始证候" in col_str
            or "证候" in col_str
            or "证型" in col_str
            or "中医证" in col_str
            or "辨证" in col_str
            or "名称" in col_str
            or "name" in col_lower
        ):
            name_col = col

        if freq_col is None and (
            "频数" in col_str
            or "次数" in col_str
            or "数量" in col_str
            or "freq" in col_lower
            or "frequency" in col_lower
            or "count" in col_lower
        ):
            freq_col = col

    if name_col is None:
        raise ValueError(
            "未找到证候名称列，请确保列名包含：原始证候、证候、证型、中医证、辨证、名称 或 name"
        )

    return name_col, freq_col


# =========================
# 10.5 标准证候词表（291）
# =========================

_CANONICAL_NAMES_CACHE = None


def get_canonical_names():
    """
    读取最终证候口径清单（Syndrome_synonym_groups.csv，291 组），
    返回去重后的归并名列表（供 prompt 嵌入 + 输出校验）。
    """
    global _CANONICAL_NAMES_CACHE
    if _CANONICAL_NAMES_CACHE is not None:
        return _CANONICAL_NAMES_CACHE

    groups_path = Path(CANONICAL_GROUPS_FILE)
    if not groups_path.exists():
        print(f"⚠ 未找到归并清单 {groups_path}，prompt 将不含词表约束。")
        _CANONICAL_NAMES_CACHE = []
        return _CANONICAL_NAMES_CACHE

    df = pd.read_csv(groups_path, encoding="utf-8-sig")
    _CANONICAL_NAMES_CACHE = sorted(
        {str(c).strip() for c in df["归并名"] if str(c).strip()}
    )
    print(f"已加载标准证候词表：{len(_CANONICAL_NAMES_CACHE)} 个。")
    return _CANONICAL_NAMES_CACHE


# =========================
# 11. 主处理函数
# =========================

def process_all_cohorts():
    """
    多队列合并模式：
    1. 读取所有队列的输入文件
    2. 提取所有"名称" → 去重 → 唯一名称集合
    3. 统一标准化引擎（A → B → C）
    4. 生成全局 standard_map
    5. 每个队列用同一张 standard_map 输出各自的 Excel
    """

    if client.api_key is None or str(client.api_key).strip() == "":
        raise ValueError(
            "未检测到 API Key，请检查配置区中的 client 设置。"
        )

    # =====================
    # Step 1：读取所有队列，提取名称
    # =====================
    print("=" * 60)
    print("Step 1：读取所有队列输入文件")
    print("=" * 60)

    cohort_data = []   # [(name, df, name_col, freq_col, output)]

    global_unique_names = set()
    global_all_names = []   # 保持顺序用于去重
    name_weights = {}       # 原始证候 -> 各队列频数之和

    for cohort in COHORTS:
        name = cohort["name"]
        input_file = cohort["input"]

        print(f"\n[{name}] 读取：{input_file}")

        if not Path(input_file).exists():
            print(f"  ⚠ 文件不存在，跳过队列 [{name}]。")
            continue

        df = read_table_safely(input_file)
        name_col, freq_col = detect_columns(df)

        print(f"  [{name}] 证候名称列：{name_col}，行数：{len(df)}")

        if freq_col:
            print(f"  [{name}] 频数列：{freq_col}")

        if TEST_MODE:
            df = df.head(TEST_N).copy()
            print(f"  [{name}] 测试模式，仅取前 {TEST_N} 行。")

        cohort_data.append((name, df, name_col, freq_col, cohort["output"]))

        # 收集名称
        for idx, row in df.iterrows():
            raw_name = row[name_col]
            if pd.isna(raw_name) or str(raw_name).strip() == "":
                continue
            raw_name = str(raw_name).strip()

            weight = 1.0
            if freq_col is not None:
                numeric_weight = pd.to_numeric(row[freq_col], errors="coerce")
                if (
                    pd.notna(numeric_weight)
                    and float(numeric_weight) > 0
                    and float(numeric_weight) != float("inf")
                ):
                    weight = float(numeric_weight)
            name_weights[raw_name] = name_weights.get(raw_name, 0.0) + weight

            if raw_name not in global_unique_names:
                global_unique_names.add(raw_name)
                global_all_names.append(raw_name)

    if not cohort_data:
        raise FileNotFoundError("所有队列输入文件均不存在，程序已停止。")
    if not global_all_names:
        raise ValueError("输入文件中没有可处理的证候名称，程序已停止。")

    print(f"\n{'='*60}")
    print(f"各队列读取完毕。")
    print(f"全局唯一证候名称数：{len(global_all_names)}")
    print(f"{'='*60}")

    # =====================
    # Step 2：加载缓存 & 构建 A 层规则
    # =====================
    print("\n" + "=" * 60)
    print("Step 2：加载缓存 & 构建 A 层规则")
    print("=" * 60)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache(CACHE_FILE)

    global FORCE_EXACT_MAP
    FORCE_EXACT_MAP = build_force_map_from_cache(cache)

    get_canonical_names()   # 预热 291 词表（供 prompt 嵌入与输出校验）

    # =====================
    # Step 3：A 层 — 强制规则预检
    # =====================
    print("\n" + "=" * 60)
    print("Step 3：A 层 — 强制规则预检")
    print("=" * 60)

    a_hit = {}        # raw_name -> result (A 层命中)
    a_miss = []       # raw_name (A 层未命中，需进 B 层)

    for raw_name in global_all_names:
        result = check_force_rules(raw_name)
        if result is not None:
            a_hit[raw_name] = result
        else:
            a_miss.append(raw_name)

    print(f"\nA 层结果：命中 {len(a_hit)} 条，未命中 {len(a_miss)} 条。")

    # =====================
    # Step 4：B 层 — 批量 API 调用
    # =====================
    print("\n" + "=" * 60)
    print("Step 4：B 层 — 批量 API 调用")
    print("=" * 60)

    b_results = {}    # raw_name -> result
    b_cache_hits = sum(
        1
        for raw_name in a_miss
        if USE_CACHE
        and is_cache_item_complete(cache.get(clean_cache_key(raw_name)))
    )
    b_api_required = len(a_miss) - b_cache_hits

    if a_miss:
        print(
            f"共 {len(a_miss)} 条进入 B 层："
            f"缓存命中 {b_cache_hits} 条，需调用 API {b_api_required} 条。"
        )
        b_results = get_batch_syndrome_standards(a_miss, cache)
    else:
        print("B 层无待处理项。")

    # ---- 合并 A + B 结果 ----
    all_results = []
    for raw_name in global_all_names:
        if raw_name in a_hit:
            all_results.append(a_hit[raw_name])
        elif raw_name in b_results:
            all_results.append(b_results[raw_name])
        else:
            # fallback（不应出现）
            all_results.append({
                "原始证候": raw_name,
                "标准证候": "不确定",
                "简称": "",
                "分类": ""
            })

    # =====================
    # Step 5：C 层 — 自愈校验
    # =====================
    all_results = self_healing_verify(all_results, cache, name_weights)

    # ---- 简称唯一化：对同简称多标准证的证候强制一一对应 ----
    abbr_override_count = 0
    for result in all_results:
        std = str(result.get("标准证候", "")).strip()
        if std in ABBR_OVERRIDE and result.get("简称", "") != ABBR_OVERRIDE[std]:
            result["简称"] = ABBR_OVERRIDE[std]
            abbr_override_count += 1
            if SAVE_CACHE:
                key = clean_cache_key(str(result.get("原始证候", "")))
                if key:
                    cache[key] = result.copy()
    if SAVE_CACHE and abbr_override_count > 0:
        save_cache(cache, CACHE_FILE)
    if abbr_override_count:
        print(
            f"\n简称唯一化：{len(ABBR_OVERRIDE)} 个标准证候强制一一对应，"
            f"修正 {abbr_override_count} 条。"
        )

    invalid_results = [
        str(result.get("原始证候", "")).strip()
        for result in all_results
        if not is_cache_item_complete(result)
    ]
    if invalid_results:
        preview = " | ".join(invalid_results[:10])
        raise RuntimeError(
            f"发现 {len(invalid_results)} 条不完整或字段不一致的结果，"
            f"已停止输出。示例：{preview}"
        )

    # ---- 构建全局 standard_map ----
    standard_map = {}   # raw_name -> {"标准证候": ..., "简称": ..., "分类": ...}
    canonical_names = set(get_canonical_names())
    off_vocab = set()
    for r in all_results:
        raw = str(r.get("原始证候", "")).strip()
        std = r.get("标准证候", "不确定")
        category = r.get("分类", "")
        # 证候类标准名应落在 291 词表内；越界者仅提示，人工复核。
        if category == "证候" and canonical_names and std not in canonical_names:
            off_vocab.add(std)
        standard_map[raw] = {
            "标准证候": std,
            "简称": r.get("简称", ""),
            "分类": category
        }

    if off_vocab:
        print(
            f"\n⚠ 有 {len(off_vocab)} 个证候标准名不在 291 词表内，"
            f"请人工复核。示例：{' | '.join(sorted(off_vocab)[:10])}"
        )

    # ---- 保存全局映射表 ----
    map_df = pd.DataFrame([
        {
            "原始证候": k,
            "标准证候": v["标准证候"],
            "简称": v["简称"],
            "分类": v["分类"]
        }
        for k, v in standard_map.items()
    ])
    map_file = Path(r"D:\python\TCM-THGNet\01maping\map\Syndrome_global_standard_map.csv")
    map_file.parent.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(map_file, index=False, encoding="utf-8-sig")
    print(f"\n全局映射表已保存：{map_file}（{len(map_df)} 条）")

    # =====================
    # Step 6：为每个队列输出
    # =====================
    print("\n" + "=" * 60)
    print("Step 6：各队列输出")
    print("=" * 60)

    for name, df, name_col, freq_col, output_file in cohort_data:
        print(f"\n[{name}] 生成输出...")

        result_rows = []
        for idx, row in df.iterrows():
            raw_value = row[name_col]
            if pd.isna(raw_value) or str(raw_value).strip() == "":
                result_rows.append({
                    "原始证候": "",
                    "标准证候": "",
                    "简称": "",
                    "分类": ""
                })
                continue

            raw_name = str(raw_value).strip()
            if raw_name not in standard_map:
                raise RuntimeError(
                    f"队列 [{name}] 的证候 '{raw_name}' 缺少标准化结果"
                )
            mapping = standard_map[raw_name]

            result_row = {
                "原始证候": raw_name,
                "标准证候": mapping["标准证候"],
                "简称": mapping["简称"],
                "分类": mapping["分类"]
            }

            if KEEP_FREQ and freq_col is not None:
                result_row[freq_col] = row[freq_col]

            result_rows.append(result_row)

        result_df = pd.DataFrame(result_rows)

        if KEEP_FREQ and freq_col is not None:
            final_columns = OUTPUT_COLUMNS + [freq_col]
        else:
            final_columns = OUTPUT_COLUMNS

        result_df = result_df.reindex(columns=final_columns)

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()

        if suffix == ".xlsx":
            result_df.to_excel(output_path, index=False)
        elif suffix == ".csv":
            result_df.to_csv(output_path, index=False, encoding="utf-8-sig")
        else:
            raise ValueError("输出文件只支持 .xlsx 或 .csv")

        print(f"  [{name}] 输出完成：{output_path}（{len(result_df)} 行）")

    # =====================
    # 终报告
    # =====================
    print("\n" + "=" * 60)
    print("全流程完成！")
    print("=" * 60)
    print(f"  队列数：{len(cohort_data)}")
    print(f"  唯一证候名称：{len(global_all_names)}")
    print(f"  A 层命中：{len(a_hit)}")
    print(f"  B 层缓存命中：{b_cache_hits}")
    print(f"  B 层 API 处理：{b_api_required}")
    print(f"  标准证候口径：{len(get_canonical_names())} 词表（291 归并）")
    print(f"  缓存文件：{CACHE_FILE}")
    for name, _, _, _, output_file in cohort_data:
        print(f"  [{name}] 输出：{output_file}")
    print("=" * 60)


# =========================
# 12. 运行入口
# =========================

if __name__ == "__main__":
    process_all_cohorts()
