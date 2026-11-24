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
GLOBAL_MAP_FILE = Path(r"D:\python\TCM-THGNet\01maping\map\TCM_global_standard_map.csv")

COHORTS = [
    {
        "name": "CKD",
        "input": r"D:\python\TCM-THGNet\CKD\TCM_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\CKD_TCM_map.xlsx",
    },
    {
        "name": "RA",
        "input": r"D:\python\TCM-THGNet\RA\TCM_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\RA_TCM_map.xlsx",
    },
    {
        "name": "MN",
        "input": r"D:\python\TCM-THGNet\MN\TCM_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\MN_TCM_map.xlsx",
    },
    {
        "name": "LC",
        "input": r"D:\python\TCM-THGNet\LC\TCM_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\LC_TCM_map.xlsx",
    },
    {
        "name": "BC",
        "input": r"D:\python\TCM-THGNet\BC\TCM_only.csv",
        "output": r"D:\python\TCM-THGNet\01maping\map\BC_TCM_map.xlsx",
    },
]

CACHE_FILE = r"D:\python\A硕士毕业_RA\cache\TCM_standard_cache.json"

# =========================
# 2. 测试开关
# =========================

TEST_MODE = False
# True：只测试前 TEST_N 行
# False：全量运行

TEST_N = 20

# 是否保留原来的频数列
KEEP_FREQ = False
# False：输出你示例中的10列表格
# True：在最后额外保留频数列

# =========================
# 3. 缓存开关
# =========================

USE_CACHE = True
# True：优先使用缓存
# False：不读取缓存，每次都重新调用API

SAVE_CACHE = True
# True：保存本次查询结果到缓存
# False：不保存缓存

CLEAR_CACHE_BEFORE_RUN = False
# True：运行前清空旧缓存
# False：保留旧缓存

SLEEP_SECONDS = 0.5


# =========================
# 4. 目标输出字段
# =========================

OUTPUT_COLUMNS = [
    "原始中药名",
    "标准中药名",
    "是否炮制品",
    "炮制类型",
    "基础药材名",
    "性",
    "味",
    "归经",
    "功效类别",
    "治疗属性"
]

# 缓存完整性检查字段
# 炮制类型可以为空，因为非炮制品本来就没有炮制类型
REQUIRED_CACHE_FIELDS = [
    "标准中药名",
    "是否炮制品",
    "基础药材名",
    "性",
    "味",
    "归经",
    "功效类别",
    "治疗属性"
]


# =========================
# 5. 读取 CSV
# =========================

def read_csv_safely(file_path):
    """
    自动尝试常见编码读取 CSV
    """
    for enc in ["utf-8-sig", "utf-8", "gbk"]:
        try:
            return pd.read_csv(file_path, encoding=enc)
        except UnicodeDecodeError:
            continue

    raise ValueError("无法识别 CSV 编码，请检查文件编码格式。")


# =========================
# 5.1 人工裁决表
# =========================

def get_manual_override_file():
    """人工裁决表与全局映射表放在同一目录，不增加或修改配置区。"""
    return GLOBAL_MAP_FILE.parent / "TCM_manual_override.csv"


def load_manual_overrides():
    """
    读取人工裁决表。

    优先级：人工裁决表 > 缓存 > API。
    “排除”和“待拆分”会保留原始名称，但标准化字段为空，不再调用 API。
    """
    override_path = get_manual_override_file()
    if not override_path.exists():
        print(f"未发现人工裁决表，将继续使用缓存和 API：{override_path}")
        return {}, override_path

    df = read_csv_safely(override_path)
    required_columns = {
        "原始中药名", "处理动作", "启用", "审核状态",
        *OUTPUT_COLUMNS[1:],
    }
    missing_columns = sorted(required_columns - set(df.columns))
    if missing_columns:
        raise ValueError(
            f"人工裁决表缺少字段：{missing_columns}；文件：{override_path}"
        )

    def cell_text(row, column):
        value = row.get(column, "")
        if pd.isna(value):
            return ""
        return str(value).strip()

    overrides = {}
    action_counts = {}
    pending_review = []

    for row_number, row in df.iterrows():
        enabled = cell_text(row, "启用").lower()
        if enabled not in {"是", "true", "1", "yes", "y"}:
            continue

        raw_name = cell_text(row, "原始中药名")
        cache_key = clean_cache_key(raw_name)
        if not cache_key:
            raise ValueError(f"人工裁决表第 {row_number + 2} 行缺少原始中药名")
        if cache_key in overrides:
            raise ValueError(f"人工裁决表存在重复原始中药名：{raw_name}")

        action = cell_text(row, "处理动作")
        if action not in {"覆盖", "排除", "待拆分"}:
            raise ValueError(
                f"人工裁决表第 {row_number + 2} 行处理动作非法：{action!r}"
            )

        if action == "覆盖":
            result = {column: cell_text(row, column) for column in OUTPUT_COLUMNS}
            result["原始中药名"] = raw_name
            validate_result(result)
        else:
            result = {column: "" for column in OUTPUT_COLUMNS}
            result["原始中药名"] = raw_name

        review_status = cell_text(row, "审核状态")
        if "待" in review_status:
            pending_review.append(raw_name)

        overrides[cache_key] = {
            "result": result,
            "action": action,
            "review_status": review_status,
            "reason": cell_text(row, "裁决依据"),
        }
        action_counts[action] = action_counts.get(action, 0) + 1

    print(
        f"已读取人工裁决表，共启用 {len(overrides)} 条："
        + "，".join(f"{action} {count} 条" for action, count in sorted(action_counts.items()))
    )
    if pending_review:
        print(
            f"[提示] 其中 {len(pending_review)} 条仍需人工复核："
            + "、".join(pending_review)
        )

    return overrides, override_path


# =========================
# 6. 缓存函数
# =========================

def clean_cache_key(name):
    """
    清洗缓存键，避免空格造成重复查询
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()
    name = name.replace(" ", "")
    name = name.replace("　", "")
    return name


def load_cache(cache_file):
    """
    读取缓存
    """
    cache_path = Path(cache_file)

    if CLEAR_CACHE_BEFORE_RUN:
        # 真实清空缓存内容，避免“本轮跳过读取、磁盘旧缓存仍保留”的歧义。
        save_cache({}, cache_path)
        print("已开启 CLEAR_CACHE_BEFORE_RUN，旧缓存已清空。")
        return {}

    if not cache_path.exists():
        print("未发现缓存文件，将创建新缓存。")
        return {}

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            cache = json.load(f)

        if not isinstance(cache, dict):
            raise ValueError("缓存根对象必须是 JSON object")

        print(f"已读取缓存，共 {len(cache)} 条记录。")
        return cache

    except Exception as e:
        raise RuntimeError(
            f"读取缓存失败。为避免覆盖原缓存，程序已停止：{cache_path}；原因：{e}"
        ) from e


def save_cache(cache, cache_file):
    """
    保存缓存
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")

    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())

    # 同盘原子替换：写入中断时保留原缓存，不产生半截 JSON。
    os.replace(temp_path, cache_path)


def is_cache_item_complete(item):
    """
    判断缓存记录是否完整
    """
    try:
        validate_result(item)
    except (TypeError, ValueError):
        return False

    return True


# =========================
# 7. JSON 解析函数
# =========================

def extract_json(content):
    """
    兼容模型返回纯 JSON 或 ```json ... ``` 的情况
    """
    if not isinstance(content, str):
        raise ValueError("API 返回内容不是字符串")

    content = content.strip()

    # 提取 Markdown 代码块中的 JSON
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if match:
        content = match.group(1).strip()

    # 优先解析纯 JSON；若模型附带少量说明，则定位第一个合法 JSON 对象。
    try:
        data = json.loads(content)
    except json.JSONDecodeError as first_error:
        decoder = json.JSONDecoder()
        data = None
        for start, char in enumerate(content):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(content[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                data = candidate
                break

        if data is None:
            raise ValueError(f"API 返回内容中未找到合法 JSON 对象：{first_error}") from first_error

    if not isinstance(data, dict):
        raise ValueError("API 返回的 JSON 根对象必须是 object")

    return data


def normalize_result(data, raw_name):
    """
    规范 API 返回结果，保证字段齐全
    """
    if not isinstance(data, dict):
        raise ValueError("标准化结果必须是字典")

    result = {}

    result["原始中药名"] = raw_name
    result["标准中药名"] = str(data.get("标准中药名", raw_name)).strip()
    result["是否炮制品"] = str(data.get("是否炮制品", "")).strip()
    result["炮制类型"] = str(data.get("炮制类型", "")).strip()
    result["基础药材名"] = str(data.get("基础药材名", result["标准中药名"])).strip()
    result["性"] = str(data.get("性", "")).strip()
    result["味"] = str(data.get("味", "")).strip()
    result["归经"] = str(data.get("归经", "")).strip()
    result["功效类别"] = str(data.get("功效类别", "")).strip()
    result["治疗属性"] = str(data.get("治疗属性", "")).strip()

    # 基础修正：非炮制品的炮制类型应为空
    if result["是否炮制品"] == "否":
        result["炮制类型"] = ""

    validate_result(result)
    return result


def validate_result(result):
    """严格校验 API 结果，防止无效内容进入缓存和正式输出。"""
    if not isinstance(result, dict):
        raise ValueError("标准化结果必须是字典")

    missing = []
    for field in REQUIRED_CACHE_FIELDS:
        value = str(result.get(field, "")).strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            missing.append(field)

    if missing:
        raise ValueError(f"API 结果缺少必填字段：{', '.join(missing)}")

    processed = str(result.get("是否炮制品", "")).strip()
    if processed not in {"是", "否", "不确定"}:
        raise ValueError(f"是否炮制品取值非法：{processed!r}")

    process_type = str(result.get("炮制类型", "")).strip()
    if processed == "是" and not process_type:
        raise ValueError("炮制品必须填写炮制类型")
    if processed == "否" and process_type:
        raise ValueError("非炮制品的炮制类型必须为空")

    treatment = str(result.get("治疗属性", "")).strip()
    if treatment != "不确定" and not treatment.endswith("类中药"):
        raise ValueError(f"治疗属性格式非法，应以“类中药”结尾：{treatment!r}")


# =========================
# 7.1 功效标签与治疗一级分类标准化
# =========================

# 功效类别保留为多标签，但只输出下列受控词。规则按此顺序同时决定最终词序。
EFFECT_TAG_RULES = [
    ("解表", r"解表|发汗|解肌|发表|宣发郁热"),
    ("散寒", r"散寒|祛寒|风寒"),
    ("疏散风热", r"疏散风热|散风清热|疏风散热|疏散退热"),
    ("透疹", r"透疹"),
    ("通鼻窍", r"通鼻窍|通窍散风寒|散风通窍"),
    ("清热", r"清热|清火|清心|清肺|清肝|凉心|凉肝|泄热"),
    ("泻火", r"泻火|降火|肝胆火"),
    ("解毒", r"解毒|攻毒|托毒"),
    ("凉血", r"凉血"),
    ("燥湿", r"燥湿"),
    ("利湿", r"利湿|清利湿热|清湿热|去湿|除湿"),
    ("解暑", r"解暑|消暑|清暑|祛暑|发表解暑"),
    ("退虚热", r"退虚热|清虚热|除骨蒸|退热除蒸|除疳热"),
    ("生津", r"生津"),
    ("止渴", r"止渴"),
    ("除烦", r"除烦"),
    ("明目", r"明目|退翳|消翳|聪耳"),
    ("清利头目", r"清利头目"),
    ("利咽", r"利咽|开音"),
    ("泻下", r"泻下|攻积"),
    ("润肠通便", r"润肠|滑肠|润燥通便|通便|健胃通便"),
    ("通利二便", r"通利二便"),
    ("逐水", r"逐水|泻水"),
    ("祛风", r"祛风(?!湿|除湿|除痹)|疏风|散风(?!寒|清热)"),
    ("祛风湿", r"祛风湿|祛风除湿|祛风除痹|祛湿通痹|除痹|通痹|祛寒湿"),
    ("通络", r"通络|通经络|活络|通利关节|利关节"),
    ("强筋骨", r"强筋骨|强筋健骨|壮筋骨|强骨|强腰膝"),
    ("化湿", r"化湿|芳香化湿"),
    ("健脾", r"健脾|补脾|醒脾|和中健脾"),
    ("升阳", r"升阳|升举阳气|升发清阳"),
    ("和胃", r"和胃|养胃"),
    ("利水", r"利水|行水|助阳化气"),
    ("利尿通淋", r"利尿|通淋|利小便|化石"),
    ("退黄", r"退黄|利胆"),
    ("温中", r"温中|温里|温胃|暖宫"),
    ("温肺", r"温肺"),
    ("补阳", r"补火助阳|补肾阳|补肾助阳|补肾壮阳|温肾阳|温肾助阳|温肾壮阳|壮阳|温补肾阳|温阳|助阳"),
    ("回阳救逆", r"回阳救逆|回阳通脉|引火归源"),
    ("理气", r"理气|行气|通气|顺气|利气|破气|下气|宽中|宽胸"),
    ("降脂化浊", r"化浊降脂|降脂化浊"),
    ("降逆止呕", r"降逆|止呕|止呃"),
    ("疏肝解郁", r"疏肝|舒肝|解郁|开郁|散郁"),
    ("消食", r"消食|健胃|开胃"),
    ("消积", r"消积|化积|导滞|化滞|消导|消疳|去积"),
    ("杀虫", r"杀虫|驱虫|灭虱|安蛔"),
    ("止血", r"止血|摄血|止崩漏|固崩"),
    ("活血化瘀", r"活血|化瘀|祛瘀|散瘀|逐瘀|破血|消瘀|散血|行血"),
    ("通经调经", r"通经|调经|调冲任"),
    ("续筋接骨", r"续筋接骨|续折伤"),
    ("消肿", r"消肿|消胀"),
    ("散结", r"散结|软坚|消癥|散痞|消痞"),
    ("化痰", r"化痰|祛痰|消痰|除痰|豁痰|痰饮|坠痰|祛风痰"),
    ("止咳", r"止咳|镇咳"),
    ("平喘", r"平喘|定喘|纳气"),
    ("润肺", r"润肺|清肺润燥|养阴清肺"),
    ("安神", r"安神|宁心|补心|交通心肾"),
    ("定惊止痉", r"定惊|镇惊|止痉|镇痉|解痉|定志"),
    ("平肝潜阳", r"平肝|潜阳|平抑肝阳"),
    ("开窍醒神", r"开窍|醒神|醒脑"),
    ("补气", r"补气|益气|大补元气|补中益气|补中缓肝|补虚|补益心脾"),
    ("益肺", r"益肺"),
    ("补血", r"补血|养血"),
    ("滋阴", r"滋阴|养阴|补阴|潜阳补阴"),
    ("补肝肾", r"补肝肾|补益肝肾|滋补肝肾|滋肾|益肾"),
    ("益精血", r"益精|补精|精血|填髓"),
    ("收涩", r"收涩|收敛|固涩|敛阴|敛肺|涩肠|涩精|固脱"),
    ("固表止汗", r"固表|止汗|敛汗"),
    ("固精缩尿", r"固精|涩精|缩尿"),
    ("止泻", r"止泻|止痢"),
    ("止带", r"止带"),
    ("安胎", r"安胎"),
    ("生肌敛疮", r"生肌|敛疮|疗疮|收湿"),
    ("排脓消痈", r"排脓|消痈|消疮|痈肿|疮毒"),
    ("乌发生发", r"生发乌发|乌须发|乌须明目"),
    ("其他作用", r"抗癌|降血压|引血下行|通窍$|解酒"),
    ("止痛", r"止痛|定痛|止痹痛"),
    ("止痒", r"止痒"),
    ("通乳", r"通乳|下乳|回乳"),
    ("截疟", r"截疟"),
    ("调和诸药", r"调和诸药|缓和药性"),
]

EFFECT_TAG_ORDER = {tag: index for index, (tag, _) in enumerate(EFFECT_TAG_RULES)}
EFFECT_TAG_PATTERNS = [
    (tag, re.compile(pattern)) for tag, pattern in EFFECT_TAG_RULES
]

# 固定一级治疗分类。一个中药只保留一个一级分类，用于统计建模和图谱节点。
PRIMARY_TREATMENT_CATEGORIES = [
    "解表类中药", "清热类中药", "泻下类中药", "祛风湿类中药",
    "化湿类中药", "利水渗湿类中药", "温里类中药", "理气类中药",
    "消食类中药", "驱虫类中药", "止血类中药", "活血化瘀类中药",
    "化痰止咳平喘类中药", "安神类中药", "平肝息风类中药",
    "开窍类中药", "补气类中药", "补阳类中药", "补血类中药",
    "补阴类中药", "补益类中药", "收涩类中药", "其他类中药",
    "不确定",
]

PRIMARY_TREATMENT_RULES = [
    ("解表类中药", r"^(解表|疏散风热|祛风散寒|祛风解表|散风|通鼻窍)"),
    ("清热类中药", r"^(清热|清暑|清心|清肝|凉血|泻火|解毒|和解退热|解暑生津)"),
    ("泻下类中药", r"^(泻下|泻水|逐水|润肠通便)"),
    ("祛风湿类中药", r"^(祛风湿|祛风除湿|祛风通络|祛湿通|祛湿除痹|舒筋)"),
    ("祛风湿类中药", r"^祛风(止痛|活血|杀虫|清热|解毒)"),
    ("化湿类中药", r"^(化湿|健脾祛湿|祛湿健脾|健脾化湿|行气化湿|理气化湿)"),
    ("利水渗湿类中药", r"^(利水|利尿|利湿|清利湿热|泻肺利水|润肠利水)"),
    ("温里类中药", r"^(温里|温中|温阳散寒|温阳祛寒|散寒|温化|温肺|温经)"),
    ("理气类中药", r"^(理气|行气|降逆|疏肝|通阳行气|利咽和胃)"),
    ("消食类中药", r"^(消食|健脾消食|消导)"),
    ("消食类中药", r"^解酒"),
    ("驱虫类中药", r"^(驱虫|攻毒杀虫|消导杀虫)"),
    ("止血类中药", r"止血"),
    ("活血化瘀类中药", r"^(活血|化瘀|破血|降脂化瘀)"),
    ("活血化瘀类中药", r"^(补肝肾通经|消肿通经)"),
    ("化痰止咳平喘类中药", r"^(化痰|止咳|润肺止咳|祛风化痰|泻肺平喘)"),
    ("安神类中药", r"^(安神|清心安神|镇心安神|重镇安神)"),
    ("平肝息风类中药", r"^(平肝|息风|祛风定惊)"),
    ("开窍类中药", r"^开窍"),
    ("补气类中药", r"^(补气|益气|固表止汗)"),
    ("补阳类中药", r"^(补肾壮阳|补肾阳|补肾助阳|补肾强骨|补肾润肠|温补肾阳|温肾|补阳|温阳强骨|温补肝肾|补肝肾强筋骨|祛湿强骨)"),
    ("补血类中药", r"^(补血|养血)"),
    ("补阴类中药", r"^(滋阴|养阴)"),
    ("补益类中药", r"^(补益|滋补|补虚|补肝肾)"),
    ("收涩类中药", r"^(收|固|涩|敛|温肾固涩|补肾固涩|温补固涩|补脾固肾)"),
    ("其他类中药", r"^(明目|升阳透疹|散结消肿|消肿排脓)"),
]

PRIMARY_TREATMENT_PATTERNS = [
    (category, re.compile(pattern))
    for category, pattern in PRIMARY_TREATMENT_RULES
]


def clean_label_text(value):
    """统一空白、标点和“类中药”后缀，供标签规则匹配。"""
    text = "" if pd.isna(value) else str(value).strip()
    text = re.sub(r"\s+", "", text)
    text = text.replace("，", "、").replace(",", "、")
    text = text.replace("；", "、").replace(";", "、")
    return text


def normalize_effect_labels(value):
    """
    将自由文本功效拆成受控多标签并统一顺序。

    返回：(标准功效字符串, 未识别的原始功效短语列表)
    """
    text = clean_label_text(value)
    if not text:
        return "", []
    if text == "不确定":
        return "不确定", []

    source_phrases = [part for part in text.split("、") if part]
    matched_tags = set()
    unmatched_phrases = []

    for phrase in source_phrases:
        if phrase in EFFECT_TAG_ORDER or phrase == "其他":
            phrase_tags = {phrase}
        else:
            phrase_tags = {
                tag for tag, pattern in EFFECT_TAG_PATTERNS if pattern.search(phrase)
            }
        if phrase_tags:
            matched_tags.update(phrase_tags)
        else:
            unmatched_phrases.append(phrase)

    if unmatched_phrases:
        matched_tags.add("其他")

    ordered_tags = sorted(
        matched_tags,
        key=lambda tag: (EFFECT_TAG_ORDER.get(tag, len(EFFECT_TAG_ORDER)), tag),
    )
    return "、".join(ordered_tags), unmatched_phrases


def map_primary_treatment(value):
    """把自由文本治疗属性映射为固定、互斥的一级分类。"""
    text = clean_label_text(value)
    if not text:
        return "", "空值"
    if text == "不确定":
        return "不确定", "原值不确定"
    if text in PRIMARY_TREATMENT_CATEGORIES:
        return text, "已是受控一级分类"

    core = re.sub(r"类中药$", "", text)
    for category, pattern in PRIMARY_TREATMENT_PATTERNS:
        if pattern.search(core):
            return category, "规则映射"
    return "其他类中药", "未命中一级分类规则"


def standardize_label_fields(result):
    """标准化单条结果中的功效多标签和治疗一级分类，并返回审计记录。"""
    standardized = result.copy()
    original_effect = clean_label_text(result.get("功效类别", ""))
    original_treatment = clean_label_text(result.get("治疗属性", ""))

    normalized_effect, unmatched_phrases = normalize_effect_labels(original_effect)
    primary_treatment, treatment_status = map_primary_treatment(original_treatment)

    standardized["功效类别"] = normalized_effect
    standardized["治疗属性"] = primary_treatment

    # 排除/待拆分记录允许标准化字段为空；其余记录必须落入受控范围。
    if primary_treatment and primary_treatment not in PRIMARY_TREATMENT_CATEGORIES:
        raise ValueError(f"治疗一级分类不在受控词表中：{primary_treatment}")

    audit = {
        "原始中药名": standardized.get("原始中药名", ""),
        "标准中药名": standardized.get("标准中药名", ""),
        "原功效类别": original_effect,
        "标准功效多标签": normalized_effect,
        "功效未识别短语": "、".join(unmatched_phrases),
        "原治疗属性": original_treatment,
        "治疗一级分类": primary_treatment,
        "治疗映射状态": treatment_status,
    }
    return standardized, audit


def save_label_standardization_audit(audit_rows, output_path):
    """保存标签转换审计表，保留转换前后取值和未映射提示。"""
    columns = [
        "原始中药名", "标准中药名", "原功效类别", "标准功效多标签",
        "功效未识别短语", "原治疗属性", "治疗一级分类", "治疗映射状态",
    ]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(audit_rows, columns=columns).to_csv(
        temp_path, index=False, encoding="utf-8-sig"
    )
    os.replace(temp_path, output_path)
    print(f"标签标准化审计表已保存 → {output_path}（{len(audit_rows)} 条）")


# =========================
# 8. 调用 API 获取中药属性
# =========================

def get_tcm_property_info(name, cache):
    """
    获取中药标准化属性：
    1. 优先查缓存
    2. 缓存不完整则重新调用 API
    3. API 成功后写入缓存
    """
    if pd.isna(name) or str(name).strip() == "":
        return {
            "原始中药名": "",
            "标准中药名": "",
            "是否炮制品": "",
            "炮制类型": "",
            "基础药材名": "",
            "性": "",
            "味": "",
            "归经": "",
            "功效类别": "",
            "治疗属性": ""
        }, False, None

    raw_name = str(name).strip()
    cache_key = clean_cache_key(raw_name)

    # ---------- 1. 读取缓存 ----------
    if USE_CACHE and cache_key in cache:
        item = cache[cache_key]

        if is_cache_item_complete(item):
            print(f"使用缓存：{raw_name} -> {item.get('标准中药名', '')}")

            result = {col: item.get(col, "") for col in OUTPUT_COLUMNS}
            result["原始中药名"] = raw_name
            return result, False, None

        print(f"缓存中“{raw_name}”字段不完整，将重新调用 API。")

    # ---------- 2. 调用 API ----------
    prompt = f"""
你是一名严谨的中药学、中药炮制学和临床数据标准化专家。

请对中药名称“{raw_name}”进行标准化整理，用于生成中药属性映射表。

核心任务：
1. 识别原始名称对应的标准中药名。
2. 区分“炮制品”和“形态变化”。
3. 识别中药别名、地方名、简写名。
4. 输出适合临床数据建模使用的统一字段。

请严格返回 JSON，不要输出解释文字，不要使用 Markdown 代码块。

返回字段必须为：

{{
  "标准中药名": "",
  "是否炮制品": "",
  "炮制类型": "",
  "基础药材名": "",
  "性": "",
  "味": "",
  "归经": "",
  "功效类别": "",
  "治疗属性": ""
}}

====================
一、重要标准化原则
====================

1. “粉、末、细粉、超微粉、饮片、片、段、块、丝、粒、颗粒、粉剂”等通常属于药材形态、粉碎状态或剂型描述，
   不是炮制方法，不应因此判定为炮制品。

   示例：
   - 白及粉 -> 标准中药名：白及；是否炮制品：否；炮制类型：空；基础药材名：白及
   - 白及末 -> 标准中药名：白及；是否炮制品：否；炮制类型：空；基础药材名：白及
   - 三七粉 -> 标准中药名：三七；是否炮制品：否；炮制类型：空；基础药材名：三七
   - 川贝母粉 -> 标准中药名：川贝母；是否炮制品：否；炮制类型：空；基础药材名：川贝母

2. 只有明确存在炮制方式时，才判定为炮制品。

   常见炮制标志包括：
   炙、蜜炙、酒制、酒炙、醋制、醋炙、盐制、盐炙、姜制、姜炙、炒、麸炒、土炒、砂炒、煅、煨、蒸、燀、制、炒炭、焦、熟。

   示例：
   - 炙甘草 -> 是否炮制品：是；炮制类型：炙；基础药材名：甘草
   - 酒当归 -> 是否炮制品：是；炮制类型：酒制；基础药材名：当归
   - 炒白术 -> 是否炮制品：是；炮制类型：炒；基础药材名：白术
   - 麸炒枳壳 -> 是否炮制品：是；炮制类型：麸炒；基础药材名：枳壳
   - 熟地黄 -> 是否炮制品：是；炮制类型：蒸制或熟制；基础药材名：地黄

3. 如果名称同时含有炮制方式和形态描述，应保留炮制信息，删除形态描述。

   示例：
   - 炙甘草粉 -> 标准中药名：炙甘草；是否炮制品：是；炮制类型：炙；基础药材名：甘草
   - 炒白术片 -> 标准中药名：炒白术；是否炮制品：是；炮制类型：炒；基础药材名：白术
   - 酒当归饮片 -> 标准中药名：酒当归；是否炮制品：是；炮制类型：酒制；基础药材名：当归

4. 如果只是规格、产地、等级、形态、包装描述，应删除这些描述，统一到标准中药名。

   示例：
   - 白及粉 -> 白及
   - 白及末 -> 白及
   - 生黄芪片 -> 黄芪
   - 黄芪饮片 -> 黄芪
   - 广藿香饮片 -> 广藿香
   - 川贝母粉 -> 川贝母

====================
二、别名和简写识别规则
====================

请尽量识别常见中药别名、简称、地方名，并统一为标准中药名。

特别注意以下规则：

1. “花粉”在中药处方语境中通常指“天花粉”，不是普通花粉。
   - 花粉 -> 天花粉
   - 天花粉 -> 天花粉
   - 栝楼根 -> 天花粉
   - 瓜蒌根 -> 天花粉

2. “化橘红胎”“橘红胎”“化州橘红胎”在本数据建模中统一归为“化橘红”。
   - 化橘红胎 -> 化橘红
   - 橘红胎 -> 化橘红
   - 化州橘红胎 -> 化橘红
   - 化州橘红 -> 化橘红

3. 常见简称或别名要尽量统一到中药标准名。
   示例：
   - 北芪 -> 黄芪
   - 绵芪 -> 黄芪
   - 云苓 -> 茯苓
   - 白茯苓 -> 茯苓
   - 赤苓 -> 茯苓或赤茯苓，按最可能中药名判断
   - 田七 -> 三七
   - 川七 -> 三七
   - 双花 -> 金银花
   - 银花 -> 金银花
   - 元胡 -> 延胡索
   - 玄胡 -> 延胡索
   - 首乌 -> 何首乌
   - 制首乌 -> 制何首乌
   - 生地 -> 生地黄
   - 熟地 -> 熟地黄
   - 大芸 -> 肉苁蓉
   - 寸冬 -> 麦冬
   - 天冬门 -> 天冬
   - 橘络 -> 橘络
   - 橘红 -> 橘红或化橘红，需结合原始名称判断

4. 如果名称可能对应多个中药，应选择中药处方中最常见、最合理的标准名。
   如果仍不确定，相关字段可填写“不确定”，但不要编造不存在的药材。

====================
三、字段填写规则
====================

1. “标准中药名”
   - 填写最终统一后的标准中药名。
   - 形态词如“粉、末、饮片、片、段、块、颗粒”一般不进入标准中药名。
   - 炮制品应保留炮制后的标准名，例如：炙甘草、酒当归、炒白术、麸炒枳壳、制何首乌、熟地黄。

2. “是否炮制品”
   - 只能填写：是、否、不确定。
   - 不能因为名称中有“粉、末、片、饮片、颗粒”就判定为“是”。

3. “炮制类型”
   - 如果是炮制品，填写具体炮制方式，例如：炙、蜜炙、酒制、醋制、盐制、炒、麸炒、煅、蒸、制、熟制、炒炭。
   - 如果不是炮制品，必须留空。

4. “基础药材名”
   - 填写炮制前或形态变化前的基础药材名称。
   - 如果不是炮制品，基础药材名通常等于标准中药名。
   - 如果只是粉、末、饮片等形态变化，基础药材名等于标准中药名。

5. “性”
   - 填写四气属性，例如：寒、热、温、凉、平。
   - 多个用顿号“、”连接。
   - 对于粉、末、饮片等形态变化，应继承基础药材属性。

6. “味”
   - 填写五味属性，例如：酸、苦、甘、辛、咸、淡、涩。
   - 多个用顿号“、”连接。
   - 对于粉、末、饮片等形态变化，应继承基础药材属性。

7. “归经”
   - 填写归经，例如：肺、脾、肝、肾、心、胃、大肠、小肠、胆、膀胱、三焦。
   - 多个用顿号“、”连接。
   - 对于粉、末、饮片等形态变化，应继承基础药材属性。

8. “功效类别”
   - 填写主要功效类别或核心功效。
   - 尽量简洁，多个用顿号“、”连接。
   - 示例：益气、补血活血、利水渗湿、清热解毒、化痰止咳、活血化瘀、健脾祛湿、凉血止血、消肿生肌。

9. “治疗属性”
   - 用于临床数据分析的归类标签。
   - 格式统一为：“xxx类中药”。
   - 示例：益气类中药、活血补血类中药、祛湿健脾类中药、清热解毒类中药、化痰止咳类中药、止血消肿类中药。
   - 不要写成长句。

====================
四、输出示例
====================

输入：白及粉
输出：
{{
  "标准中药名": "白及",
  "是否炮制品": "否",
  "炮制类型": "",
  "基础药材名": "白及",
  "性": "微寒",
  "味": "苦、甘、涩",
  "归经": "肺、肝、胃",
  "功效类别": "收敛止血、消肿生肌",
  "治疗属性": "止血消肿类中药"
}}

输入：白及末
输出：
{{
  "标准中药名": "白及",
  "是否炮制品": "否",
  "炮制类型": "",
  "基础药材名": "白及",
  "性": "微寒",
  "味": "苦、甘、涩",
  "归经": "肺、肝、胃",
  "功效类别": "收敛止血、消肿生肌",
  "治疗属性": "止血消肿类中药"
}}

输入：花粉
输出：
{{
  "标准中药名": "天花粉",
  "是否炮制品": "否",
  "炮制类型": "",
  "基础药材名": "天花粉",
  "性": "微寒",
  "味": "甘、微苦",
  "归经": "肺、胃",
  "功效类别": "清热泻火、生津止渴、消肿排脓",
  "治疗属性": "清热生津类中药"
}}

输入：化橘红胎
输出：
{{
  "标准中药名": "化橘红",
  "是否炮制品": "否",
  "炮制类型": "",
  "基础药材名": "化橘红",
  "性": "温",
  "味": "辛、苦",
  "归经": "肺、脾",
  "功效类别": "理气宽中、燥湿化痰",
  "治疗属性": "理气化痰类中药"
}}

输入：炙甘草粉
输出：
{{
  "标准中药名": "炙甘草",
  "是否炮制品": "是",
  "炮制类型": "炙",
  "基础药材名": "甘草",
  "性": "平",
  "味": "甘",
  "归经": "心、肺、脾、胃",
  "功效类别": "补脾和胃、益气复脉、调和诸药",
  "治疗属性": "益气调和类中药"
}}

====================
五、最终要求
====================

- 必须返回合法 JSON。
- 不要返回 Markdown。
- 不要输出任何解释文字。
- 不要把“粉、末、片、饮片、颗粒”等形态词误判为炮制。
- 对形态变化类名称，标准中药名、基础药材名、性、味、归经、功效类别、治疗属性应继承原基础药材。
- 对别名、简写、地方名，应尽量统一到标准中药名。
"""
    last_error = None
    max_attempts = 3

    for attempt in range(1, max_attempts + 1):
        try:
            response = client.chat.completions.create(
                model= MODEL_NAME,
                messages=[
                    {
                        "role": "system",
                        "content": "你是一名严谨的中药学专家，擅长中药名称标准化、炮制品识别、性味归经和功效分类。"
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False,
                timeout=60.0,
            )

            content = response.choices[0].message.content
            data = extract_json(content)
            result = normalize_result(data, raw_name)

            # 只更新内存缓存，由主流程分批持久化，避免每条记录重写整个 JSON。
            if SAVE_CACHE:
                cache[cache_key] = result
                print(f"已加入待保存缓存：{raw_name} -> {result['标准中药名']}")

            return result, True, None

        except Exception as e:
            last_error = e
            if attempt < max_attempts:
                retry_wait = max(1.0, SLEEP_SECONDS) * (2 ** (attempt - 1))
                print(
                    f"处理“{raw_name}”第 {attempt}/{max_attempts} 次失败：{e}；"
                    f"{retry_wait:.1f} 秒后重试。"
                )
                time.sleep(retry_wait)

    error_message = f"处理“{raw_name}”连续 {max_attempts} 次失败：{last_error}"
    print(error_message)
    # 不再伪造兜底记录；由主流程汇总失败并阻止覆盖正式输出。
    return None, True, error_message


# =========================
# 9. 自动识别输入列
# =========================

def detect_columns(df):
    """
    自动识别原始中药名列和频数列
    """
    def choose_column(exact_names, contains_names, kind):
        candidates = []
        for position, col in enumerate(df.columns):
            text = str(col).strip()
            lower = text.lower()

            if text in exact_names:
                score = 100 - exact_names.index(text)
            else:
                matched = [term for term in contains_names if term in text or term in lower]
                if not matched:
                    continue
                score = 50 + max(len(term) for term in matched)

            candidates.append((score, -position, col))

        if not candidates:
            return None

        candidates.sort(reverse=True, key=lambda item: (item[0], item[1]))
        best_score = candidates[0][0]
        tied = [item[2] for item in candidates if item[0] == best_score]
        if len(tied) > 1:
            raise ValueError(f"检测到多个同优先级的{kind}列：{tied}，请明确列名")
        return candidates[0][2]

    name_col = choose_column(
        ["原始中药名", "中药名", "中药名称", "药材名", "药材名称", "名称", "name"],
        ["原始中药名", "中药名", "中药", "药材", "名称", "name"],
        "中药名称",
    )
    freq_col = choose_column(
        ["频数", "次数", "数量", "frequency", "freq", "count"],
        ["频数", "次数", "数量", "frequency", "freq", "count"],
        "频数",
    )

    if name_col is None:
        raise ValueError("未找到中药名称列，请确保列名包含：原始中药名、中药名、中药、药材、名称 或 name")

    return name_col, freq_col


# =========================
# 10. 收集唯一名称 & 生成队列输出
# =========================

def collect_unique_names(cohorts):
    """
    从所有队列的 CSV 中提取名称列，合并去重
    """
    all_names = []

    for cohort in cohorts:
        df = read_csv_safely(cohort["input"])
        name_col, _ = detect_columns(df)
        print(f"[{cohort['name']}] 读取到 {len(df)} 行，名称列：{name_col}")

        for val in df[name_col]:
            key = clean_cache_key(val)
            if key:
                all_names.append(key)

    unique_names = sorted(set(all_names))
    print(f"\n合并去重后共 {len(unique_names)} 个唯一中药名称。")
    return unique_names


def generate_cohort_output(cohort, standard_map, test_mode=False):
    """
    为单个队列生成输出文件。
    KEEP_FREQ=False 时仅输出10个标准化字段；True 时在末尾保留频数列。
    测试模式仅输出已测试名称，并使用独立的 _TEST 文件，绝不覆盖正式结果。
    """
    df = read_csv_safely(cohort["input"])
    name_col, freq_col = detect_columns(df)

    result_rows = []

    for _, row in df.iterrows():
        raw_name = row[name_col]
        cache_key = clean_cache_key(raw_name)

        # 从 standard_map 查找标准化结果
        if not cache_key:
            std_result = {col: "" for col in OUTPUT_COLUMNS}
        elif cache_key in standard_map:
            std_result = standard_map[cache_key].copy()
        else:
            if test_mode:
                continue
            raise KeyError(f"[{cohort['name']}] 中药“{raw_name}”缺少标准化结果")

        # 原始中药名保留输入文件中的显示形式，而缓存键仍使用清洗后的形式。
        std_result["原始中药名"] = "" if pd.isna(raw_name) else str(raw_name).strip()
        combined = {col: std_result.get(col, "") for col in OUTPUT_COLUMNS}

        if KEEP_FREQ:
            if freq_col is None:
                raise ValueError(f"[{cohort['name']}] KEEP_FREQ=True，但未检测到频数列")
            combined[freq_col] = row[freq_col]

        result_rows.append(combined)

    output_columns = OUTPUT_COLUMNS.copy()
    if KEEP_FREQ and freq_col is not None:
        output_columns.append(freq_col)
    result_df = pd.DataFrame(result_rows, columns=output_columns)

    output_path = Path(cohort["output"])
    if test_mode:
        output_path = output_path.with_name(f"{output_path.stem}_TEST{output_path.suffix}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")
    result_df.to_excel(temp_path, index=False)
    os.replace(temp_path, output_path)

    print(f"[{cohort['name']}] 输出完成，共 {len(result_df)} 行 → {output_path}")


def check_standard_consistency(standard_map):
    """
    自检：同一标准中药名下，基础药材名、性、味、归经、功效类别、治疗属性是否一致
    """
    print("\n" + "=" * 50)
    print("自检：标准中药名属性一致性检查")
    print("=" * 50)

    CHECK_FIELDS = ["基础药材名", "性", "味", "归经", "功效类别", "治疗属性"]

    def canonical_value(field, value):
        """统一空白、分隔符和多值顺序，减少仅由书写顺序造成的假冲突。"""
        text = re.sub(r"\s+", "", str(value).strip())
        if field in {"性", "味", "归经", "功效类别"}:
            parts = [part for part in re.split(r"[、,，;；/]+", text) if part]
            return "、".join(sorted(set(parts)))
        return text

    # 按标准中药名分组
    groups = {}
    for key, item in standard_map.items():
        std_name = str(item.get("标准中药名", "")).strip()
        if not std_name or std_name.lower() in ["nan", "none", "null", "不确定"]:
            continue
        groups.setdefault(std_name, []).append(item)

    conflicts = []

    for std_name, items in groups.items():
        if len(items) <= 1:
            continue

        field_conflicts = {}
        for field in CHECK_FIELDS:
            # 收集该字段的不同值及其来源（原始中药名）
            value_sources = {}
            for item in items:
                val = str(item.get(field, "")).strip()
                raw = str(item.get("原始中药名", "")).strip()
                canonical = canonical_value(field, val)
                value_sources.setdefault(canonical, {"values": set(), "sources": []})
                value_sources[canonical]["values"].add(val)
                value_sources[canonical]["sources"].append(raw)

            # 去掉空值后仍有多个不同值 → 冲突
            non_empty = {}
            for canonical, detail in value_sources.items():
                if not canonical or canonical.lower() in ["nan", "none", "null"]:
                    continue
                display_value = " / ".join(sorted(detail["values"]))
                non_empty[display_value] = detail["sources"]
            if len(non_empty) > 1:
                field_conflicts[field] = non_empty

        if field_conflicts:
            conflicts.append((std_name, field_conflicts))

    if not conflicts:
        print("[通过] 自检通过，所有标准中药名的属性一致。")
    else:
        print(f"[警告] 发现 {len(conflicts)} 个标准中药名存在属性冲突：\n")
        for std_name, field_conflicts in conflicts:
            print(f"  标准中药名: {std_name}")
            for field, value_sources in field_conflicts.items():
                print(f"    字段 [{field}]:")
                for val, sources in value_sources.items():
                    print(f"      \"{val}\" ← {sources}")
            print()

    return conflicts


def save_conflict_report(conflicts, report_path):
    """把一致性冲突保存为长表，便于人工核查后修正缓存。"""
    rows = []
    for std_name, field_conflicts in conflicts:
        for field, value_sources in field_conflicts.items():
            for value, sources in value_sources.items():
                rows.append({
                    "标准中药名": std_name,
                    "冲突字段": field,
                    "字段值": value,
                    "来源原始中药名": "、".join(sources),
                })

    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = report_path.with_suffix(report_path.suffix + ".tmp")
    pd.DataFrame(
        rows,
        columns=["标准中药名", "冲突字段", "字段值", "来源原始中药名"],
    ).to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, report_path)
    print(f"一致性冲突报告已保存 → {report_path}")


# 标准名级一致性收敛涉及的字段，与 check_standard_consistency 的 CHECK_FIELDS 保持一致。
STANDARD_CONSISTENCY_FIELDS = ["基础药材名", "性", "味", "归经", "功效类别", "治疗属性"]


def resolve_standard_name_consistency(standard_map):
    """
    标准名级一致性收敛：

    同一「标准中药名」下的别名、炮制品、形态变化等原始名，其
    基础药材名/性/味/归经/功效类别/治疗属性应以权威条目为准统一。

    权威条目 = 原始中药名与标准中药名完全一致的那条记录（如“三七”之于
    “三七片、三七粉、田七……”）。没有权威条目的标准名会被跳过，
    交给 check_standard_consistency 报告并交由人工处理。

    返回：(收敛后的 standard_map, 审计行列表, 已收敛标准名数,
           被覆盖字段数, 跳过的标准名列表)
    """
    groups = {}
    for key, item in standard_map.items():
        std_name = str(item.get("标准中药名", "")).strip()
        if not std_name or std_name.lower() in {"nan", "none", "null", "不确定"}:
            continue
        groups.setdefault(std_name, []).append(key)

    audit_rows = []
    skipped = []
    resolved_groups = 0
    changed_fields = 0

    for std_name, keys in groups.items():
        if len(keys) <= 1:
            continue

        canonical_key = None
        for key in keys:
            raw = clean_cache_key(standard_map[key].get("原始中药名", ""))
            if raw == clean_cache_key(std_name):
                canonical_key = key
                break

        if canonical_key is None:
            skipped.append(std_name)
            continue

        canonical_item = standard_map[canonical_key]
        for field in STANDARD_CONSISTENCY_FIELDS:
            authoritative = str(canonical_item.get(field, "")).strip()
            for key in keys:
                if key == canonical_key:
                    continue
                item = standard_map[key]
                current = str(item.get(field, "")).strip()
                if current != authoritative:
                    audit_rows.append({
                        "标准中药名": std_name,
                        "原始中药名": item.get("原始中药名", ""),
                        "收敛字段": field,
                        "原值": current,
                        "收敛后值": authoritative,
                        "权威来源": canonical_item.get("原始中药名", ""),
                    })
                    item[field] = authoritative
                    changed_fields += 1

        resolved_groups += 1

    return standard_map, audit_rows, resolved_groups, changed_fields, skipped


def save_convergence_audit(audit_rows, output_path):
    """保存标准名一致性收敛审计表，便于人工复核被覆盖的字段。"""
    columns = ["标准中药名", "原始中药名", "收敛字段", "原值", "收敛后值", "权威来源"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pd.DataFrame(audit_rows, columns=columns).to_csv(
        temp_path, index=False, encoding="utf-8-sig"
    )
    os.replace(temp_path, output_path)
    print(f"标准名一致性收敛审计表已保存 → {output_path}（{len(audit_rows)} 条变更）")


def save_standard_map(standard_map, output_path):
    """按固定字段顺序原子保存全局映射表。"""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    global_df = pd.DataFrame(list(standard_map.values()), columns=OUTPUT_COLUMNS)
    global_df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    os.replace(temp_path, output_path)
    print(f"全局映射表已保存 → {output_path}（{len(global_df)} 条）")


# =========================
# 11. 主处理函数
# =========================

def process_all_cohorts():
    # ===== 阶段1：收集所有队列的唯一中药名 =====
    print("=" * 50)
    print("阶段1：收集所有队列的唯一中药名")
    print("=" * 50)
    unique_names = collect_unique_names(COHORTS)

    if TEST_MODE:
        unique_names = unique_names[:TEST_N]
        print(f"测试模式，仅处理前 {TEST_N} 个唯一名称。")

    # ===== 阶段2：批量标准化唯一名称 =====
    print("\n" + "=" * 50)
    print("阶段2：批量标准化唯一名称")
    print("=" * 50)
    cache = load_cache(CACHE_FILE)
    manual_overrides, manual_override_path = load_manual_overrides()
    standard_map = {}
    failures = []
    api_calls = 0
    cache_hits = 0
    manual_hits = 0
    pending_cache_updates = 0
    label_audit_rows = []
    total = len(unique_names)

    for i, raw_name in enumerate(unique_names, start=1):
        print(f"\n标准化 {i}/{total}：{raw_name}")
        cache_key = clean_cache_key(raw_name)

        # 人工裁决优先于缓存和 API，且不会回写覆盖原缓存。
        if cache_key in manual_overrides:
            manual_item = manual_overrides[cache_key]
            result = manual_item["result"].copy()
            result["原始中药名"] = raw_name
            result, audit = standardize_label_fields(result)
            standard_map[cache_key] = result
            label_audit_rows.append(audit)
            manual_hits += 1
            print(
                f"使用人工裁决：{raw_name}；动作={manual_item['action']}；"
                f"审核状态={manual_item['review_status']}"
            )
            continue

        result, api_called, error_message = get_tcm_property_info(raw_name, cache)

        if api_called:
            api_calls += 1
        else:
            cache_hits += 1

        if error_message is not None:
            failures.append((raw_name, error_message))
        else:
            result, audit = standardize_label_fields(result)
            standard_map[cache_key] = result
            label_audit_rows.append(audit)

        if SAVE_CACHE and api_called and result is not None:
            pending_cache_updates += 1
            # 分批保存，兼顾断点恢复和磁盘写入效率。
            if pending_cache_updates >= 20:
                save_cache(cache, CACHE_FILE)
                pending_cache_updates = 0
                print("缓存已批量保存。")

        # 仅真实调用 API 后等待；缓存命中不再浪费时间。
        if api_called and SLEEP_SECONDS > 0:
            time.sleep(SLEEP_SECONDS)

    if SAVE_CACHE and pending_cache_updates > 0:
        save_cache(cache, CACHE_FILE)
        print("缓存已保存。")

    print(
        f"\n本轮统计：人工裁决 {manual_hits} 条，缓存命中 {cache_hits} 条，"
        f"调用 API {api_calls} 条，失败 {len(failures)} 条。"
    )

    if failures:
        print("\n以下名称处理失败：")
        for raw_name, error_message in failures:
            print(f"  - {raw_name}: {error_message}")
        raise RuntimeError(
            f"共有 {len(failures)} 个中药名称处理失败。"
            "为避免污染研究数据，本次未覆盖全局映射表和队列输出。"
        )

    if not standard_map:
        raise RuntimeError("未生成任何标准化结果，程序停止且不会覆盖既有输出。")

    # ===== 阶段3：标准名一致性收敛 → 一致性检查 → 保存全局映射表 =====
    print("\n" + "=" * 50)
    print("阶段3：标准名一致性收敛、检查与保存全局映射表")
    print("=" * 50)

    if TEST_MODE:
        global_output_path = GLOBAL_MAP_FILE.with_name(
            f"{GLOBAL_MAP_FILE.stem}_TEST{GLOBAL_MAP_FILE.suffix}"
        )
        conflict_report_path = GLOBAL_MAP_FILE.with_name("TCM_standard_conflicts_TEST.csv")
        label_audit_path = GLOBAL_MAP_FILE.with_name("TCM_label_standardization_audit_TEST.csv")
    else:
        global_output_path = GLOBAL_MAP_FILE
        conflict_report_path = GLOBAL_MAP_FILE.with_name("TCM_standard_conflicts.csv")
        label_audit_path = GLOBAL_MAP_FILE.with_name("TCM_label_standardization_audit.csv")

    convergence_audit_path = GLOBAL_MAP_FILE.with_name(
        f"TCM_standard_convergence_audit{'_TEST' if TEST_MODE else ''}.csv"
    )

    standard_map, convergence_rows, resolved_groups, changed_fields, skipped = \
        resolve_standard_name_consistency(standard_map)
    if convergence_rows:
        save_convergence_audit(convergence_rows, convergence_audit_path)
        print(
            f"标准名一致性收敛完成：{resolved_groups} 个标准名、"
            f"{changed_fields} 个字段已统一。"
        )
    else:
        print("标准名一致性收敛：未发生字段覆盖。")
    if skipped:
        print(
            f"[提示] 以下 {len(skipped)} 个标准名无权威条目，已跳过收敛；"
            f"如仍有冲突需人工处理：{'、'.join(skipped)}"
        )

    conflicts = check_standard_consistency(standard_map)

    if conflicts:
        save_conflict_report(conflicts, conflict_report_path)
        raise RuntimeError(
            f"发现 {len(conflicts)} 个标准中药名存在属性冲突。"
            f"请先核查 {conflict_report_path}；本次未覆盖正式映射表和队列输出。"
        )

    if conflict_report_path.exists():
        conflict_report_path.unlink()

    save_standard_map(standard_map, global_output_path)
    save_label_standardization_audit(label_audit_rows, label_audit_path)

    effect_values = {
        item.get("功效类别", "") for item in standard_map.values()
        if item.get("功效类别", "")
    }
    treatment_values = {
        item.get("治疗属性", "") for item in standard_map.values()
        if item.get("治疗属性", "")
    }
    effect_tags = {
        tag
        for value in effect_values
        for tag in str(value).split("、")
        if tag
    }
    unmatched_count = sum(
        bool(row.get("功效未识别短语", "")) for row in label_audit_rows
    )
    other_treatment_count = sum(
        row.get("治疗一级分类", "") == "其他类中药"
        for row in label_audit_rows
    )
    print(
        f"标签标准化统计：功效原子标签 {len(effect_tags)} 个，"
        f"功效组合 {len(effect_values)} 种，治疗一级分类 {len(treatment_values)} 个；"
        f"含未识别功效短语 {unmatched_count} 条，治疗归入其他 {other_treatment_count} 条。"
    )

    # ===== 阶段4：为每个队列生成独立输出 =====
    print("\n" + "=" * 50)
    print("阶段4：为每个队列生成独立输出")
    print("=" * 50)
    for cohort in COHORTS:
        generate_cohort_output(cohort, standard_map, test_mode=TEST_MODE)

    print("\n" + "=" * 50)
    if TEST_MODE:
        print("测试处理完成：所有结果均写入 _TEST 文件，正式文件未被覆盖。")
    else:
        print("全部处理完成！")
    print(f"缓存文件位置：{CACHE_FILE}")
    print(f"人工裁决表位置：{manual_override_path}")


# =========================
# 11. 运行入口
# =========================

if __name__ == "__main__":
    process_all_cohorts()
