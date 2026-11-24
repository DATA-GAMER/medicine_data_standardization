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
MODEL_NAME ="deepseek-v4-pro"
BASE_DIR = Path(r'D:\python\A硕士毕业_RA')

# 四个队列的输入文件（Drug_only.csv）和输出文件（Drug_map.xlsx）
QUEUE_CONFIG = {
    "CKD": {
        "input":  Path(r"D:\python\TCM-THGNet\CKD\Drug_only.csv"),
        "output":Path(r"D:\python\TCM-THGNet\01maping\map\CKD_Drug_map.xlsx"),
    },
    "RA": {
        "input":  Path(r"D:\python\TCM-THGNet\RA\Drug_only.csv"),
        "output":Path(r"D:\python\TCM-THGNet\01maping\map\RA_Drug_map.xlsx"),
    },
    "MN": {
        "input":  Path(r"D:\python\TCM-THGNet\MN\Drug_only.csv"),
        "output":Path(r"D:\python\TCM-THGNet\01maping\map\MN_Drug_map.xlsx"),
    },
    "LC": {
        "input": Path(r"D:\python\TCM-THGNet\LC\Drug_only.csv"),
        "output": Path(r"D:\python\TCM-THGNet\01maping\map\LC_Drug_map.xlsx"),
    },
    "BC": {
        "input": Path(r"D:\python\TCM-THGNet\BC\Drug_only.csv"),
        "output": Path(r"D:\python\TCM-THGNet\01maping\map\BC_Drug_map.xlsx"),
    },
}

# 全局标准化映射表输出路径
GLOBAL_MAP_FILE = Path(r"D:\python\TCM-THGNet\01maping\map\Drug_global_standard_map.csv")

# 一致性冲突报告输出路径
CONFLICTS_FILE = Path(r"D:\python\TCM-THGNet\01maping\map\Drug_consistency_conflicts.csv")

# 本地补充字典（API 无法识别的手工补全，Excel 可编辑）
FALLBACK_FILE = Path(r"D:\python\TCM-THGNet\01maping\map\Drug_fallback.xlsx")

# 缓存文件路径
CACHE_FILE = r"D:\python\A硕士毕业_RA\cache\Drug_standard_cache.json"

# =========================
# 1.5 术语归一化表（一致性检查前统一同义词）
# =========================

# 药物大类：将 API 返回的复合/变体词映射到规范词
CANONICAL_DRUG_CLASS = {
    # ===== 希腊字母统一 =====
    "alpha受体阻滞剂": "心血管系统用药",
    "alpha1受体阻滞剂": "心血管系统用药",
    "beta受体阻滞剂": "心血管系统用药",
    "beta2受体激动剂": "呼吸系统用药",
    "5alpha-还原酶抑制剂": "泌尿系统用药",
    "alpha-葡萄糖苷酶抑制剂": "内分泌及代谢用药",
    "beta-内酰胺类抗菌药": "抗感染用药",
    "beta-内酰胺类抗菌药酶抑制剂复方": "抗感染用药",
    "β-内酰胺类抗菌药/酶抑制剂复方": "抗感染用药",

    # ===== 麻醉统一 =====
    "静脉麻醉药": "麻醉药",
    "吸入麻醉药": "麻醉药",
    "全身麻醉药": "麻醉药",
    "局部麻醉药": "麻醉药",
    "局部麻醉药复方": "麻醉药",
    "局部麻醉药/抗心律失常药": "麻醉药",

    # ===== PPI — 统一为质子泵抑制剂 =====
    "胃肠用药/质子泵抑制剂": "消化系统用药",

    # ===== H2RA =====
    "胃肠用药/H2受体拮抗剂": "消化系统用药",

    # ===== 神经系统用药 → 具体类别 =====
    "神经系统用药/抗眩晕药": "神经系统用药",
    "抗抑郁药/SNRI": "精神药品",
    "抗抑郁药/5-羟色胺和去甲肾上腺素再摄取抑制剂": "精神药品",
    "抗抑郁药/SSRI": "精神药品",
    "抗焦虑药/5-HT1A受体部分激动剂": "精神药品",
    "抗帕金森病药/中枢性抗胆碱药": "神经系统用药",
    "抗癫痫药/神经病理性疼痛治疗药": "神经系统用药",

    # ===== 抗菌药 → 统一子类命名 =====
    "喹诺酮类抗菌药": "抗感染用药",
    "头孢菌素类抗菌药": "抗感染用药",
    "大环内酯类抗菌药": "抗感染用药",
    "大环内酯类抗生素": "抗感染用药",
    "青霉素类抗菌药": "抗感染用药",
    "青霉素类抗生素": "抗感染用药",
    "林可酰胺类抗菌药": "抗感染用药",
    "四环素类抗菌药": "抗感染用药",
    "四环素类抗生素": "抗感染用药",
    "硝基咪唑类抗菌药": "抗感染用药",
    "噁唑烷酮类抗菌药": "抗感染用药",
    "恶唑烷酮类抗菌药": "抗感染用药",
    "抗菌药复方": "抗感染用药",

    # ===== 呼吸系统 =====
    "吸入性糖皮质激素/长效β2受体激动剂复方": "呼吸系统用药",
    "祛痰药/黏液溶解剂": "呼吸系统用药",
    "化痰药": "呼吸系统用药",
    "平喘药": "呼吸系统用药",

    # ===== 消化系统 =====
    "胃肠用药/止泻药": "消化系统用药",
    # 胃肠动力/止泻/通便/止吐/黏膜保护/解痉 → 统一为胃肠道用药
    "胃肠动力药/止吐药": "消化系统用药",
    "胃肠促动力药/止吐药": "消化系统用药",
    "胃肠动力药": "消化系统用药",
    "胃肠促动力药": "消化系统用药",
    "胃肠解痉药": "消化系统用药",
    "胃黏膜保护剂": "消化系统用药",
    "胃肠用药复方": "消化系统用药",
    "抗酸药复方": "消化系统用药",
    "肝病辅助用药/磷脂类制剂": "消化系统用药",
    "肝病辅助用药/营养支持药": "消化系统用药",
    "肝病用药": "消化系统用药",

    # ===== 血液/肿瘤 =====
    "止血药/血管保护剂": "血液系统用药",
    "促凝血药/止血药": "血液系统用药",
    "促凝血药": "血液系统用药",
    "抗凝血药拮抗剂": "血液系统用药",
    "溶栓药/纤溶酶原激活剂": "血液系统用药",
    "造血生长因子/升白细胞药": "血液系统用药",
    "铁剂/抗贫血药": "血液系统用药",
    "抗贫血药复方": "血液系统用药",

    # ===== 免疫/生物制剂 =====
    "生物制剂/生长因子": "免疫调节/抗风湿药",
    "生物制剂/组织修复药": "免疫调节/抗风湿药",
    "肿瘤坏死因子抑制剂/生物制剂": "免疫调节/抗风湿药",
    "肿瘤坏死因子抑制剂/生物制剂DMARDs": "免疫调节/抗风湿药",
    "免疫抑制剂/烷化剂": "免疫调节/抗风湿药",
    "免疫调节剂": "免疫调节/抗风湿药",
    "免疫增强药": "免疫调节/抗风湿药",
    "免疫兴奋剂": "免疫调节/抗风湿药",

    # ===== 心血管 =====
    "血管扩张剂": "心血管系统用药",
    "血管保护剂/静脉活性药": "心血管系统用药",
    "静脉活性药/血管保护药": "心血管系统用药",

    # ===== 骨代谢 =====
    "骨关节炎治疗药/软骨保护剂": "骨骼肌肉系统用药",
    "骨吸收抑制剂": "内分泌及代谢用药",

    # ===== 维生素统一 → 全部归为"维生素" =====
    "维生素类药物": "营养支持药",
    "维生素类药": "营养支持药",
    "维生素类": "营养支持药",
    "维生素复方": "营养支持药",
    "维生素复方制剂": "营养支持药",
    "维生素矿物质复方": "营养支持药",
    "维生素/矿物质复方": "营养支持药",
    "维生素与矿物质复方": "营养支持药",
    "维生素K类": "营养支持药",
    "活性维生素D": "营养支持药",
    "维生素D类药物": "营养支持药",
    "维生素D类似物": "营养支持药",
    "钙剂/维生素D制剂复方": "营养支持药",
    "钙补充剂复方": "营养支持药",
    "神经营养药": "营养支持药",
    "维生素B12及其衍生物": "营养支持药",
    "维生素B12类药物": "营养支持药",
    "复方矿物质/维生素": "营养支持药",

    # ===== 营养支持 → 全部归为"营养支持药" =====
    "营养支持": "营养支持药",
    "营养支持药": "营养支持药",
    "营养支持复方": "营养支持药",
    "营养补充药复方": "营养支持药",
    "营养药复方": "营养支持药",
    "营养药": "营养支持药",
    "复方营养药": "营养支持药",
    "复方营养制剂": "营养支持药",
    "肠外营养药": "营养支持药",
    "心肌营养药": "营养支持药",

    # ===== 电解质/输液统一 =====
    "电解质补充剂/尿液碱化剂": "营养支持药",
    "电解质平衡药": "营养支持药",
    "电解质补充剂复方": "营养支持药",
    "复方电解质输液": "营养支持药",
    "复方电解质补充剂": "营养支持药",

    # ===== 血浆/血液制品 =====
    "血浆代用品/胶体溶液": "血液系统用药",
    "血容量扩充剂": "血液系统用药",

    # ===== 其他 =====
    "微生态制剂/益生菌": "消化系统用药",
    "微生态制剂/益生菌复方": "消化系统用药",
    "微生态制剂/益生菌制剂": "消化系统用药",
    "抗组胺药/抗过敏药": "抗组胺药",
    "抗过敏药": "抗组胺药",
    "抗胆碱能药/膀胱解痉药": "泌尿系统用药",
    "抗胆碱能药/膀胱过度活动症用药": "泌尿系统用药",
    "眼科用药/组织修复剂": "皮肤科用药",
    "脑代谢改善药": "神经系统用药",
    "脑功能改善药/营养支持药": "神经系统用药",
    "脑血管保护药": "神经系统用药",
    "脑血管病用药": "神经系统用药",
    "保肝药": "消化系统用药",
    "肝脏保护药": "消化系统用药",
    "抗炎保肝药": "消化系统用药",
    "肝病辅助用药/抗炎免疫调节剂": "消化系统用药",
    "复方保肝药": "消化系统用药",
    "利胆药": "消化系统用药",
    "胆汁酸类药物": "消化系统用药",
    "降尿酸药": "泌尿系统用药",
    "抗痛风药": "泌尿系统用药",
    "拟钙剂": "内分泌及代谢用药",
    "钙调节剂": "内分泌及代谢用药",
    "双膦酸盐类抗骨质疏松药": "内分泌及代谢用药",
    "双膦酸盐类骨代谢调节剂": "内分泌及代谢用药",
    "抗骨质疏松药": "内分泌及代谢用药",
    "解毒药": "解毒药",
    "消毒防腐药": "抗感染用药",
    "抗感染药": "抗感染用药",
    "硝基呋喃类抗菌药": "抗感染用药",
    "抗真菌药": "抗感染用药",
    "三唑类抗真菌药": "抗感染用药",
    "抗病毒药": "抗感染用药",
    "镇静催眠药": "精神药品",
    "苯二氮䓬类镇静催眠药": "精神药品",
    "苯二氮䓬类": "精神药品",
    "镇静药": "精神药品",
    "抗抑郁药/SNRI": "精神药品",
    "抗抑郁药复方": "精神药品",
    "三环类抗抑郁药": "精神药品",
    "强心苷类": "心血管系统用药",
    "窦房结If电流抑制剂": "心血管系统用药",
    "If通道抑制剂": "心血管系统用药",
    "硝酸酯类血管扩张剂": "心血管系统用药",
    "硝酸酯类": "心血管系统用药",
    "中枢性肌肉松弛剂": "神经系统用药",
    "神经肌肉阻滞剂": "麻醉药",
    "非去极化神经肌肉阻滞剂": "麻醉药",
    "血小板生成素受体激动剂": "血液系统用药",
    "抗血小板药": "血液系统用药",
    "抗凝药": "血液系统用药",
    "溶栓药": "血液系统用药",
    "抗利尿激素类": "内分泌及代谢用药",
    "抗利尿激素": "内分泌及代谢用药",
    "5-羟色胺3受体拮抗剂": "消化系统用药",
    "5-羟色胺3（5-HT3）受体拮抗剂": "消化系统用药",
    "白三烯受体拮抗剂": "呼吸系统用药",
    "H1受体拮抗剂": "抗组胺药",
    "第二代H1受体拮抗剂": "抗组胺药",
    "抗组胺药（H1受体拮抗剂）": "抗组胺药",
    "选择性5-羟色胺再摄取抑制剂": "精神药品",
    "选择性5-HT再摄取抑制剂": "精神药品",
    "5-羟色胺和去甲肾上腺素再摄取抑制剂": "精神药品",
    "多巴胺受体激动剂": "神经系统用药",
    "α2肾上腺素能受体激动剂": "心血管系统用药",
    "醛固酮受体拮抗剂": "内分泌及代谢用药",
    "醛糖还原酶抑制剂": "内分泌及代谢用药",
    "胆碱酯酶抑制剂": "神经系统用药",
    "抗胆碱酯酶药": "神经系统用药",
    "质子泵抑制剂": "消化系统用药",
    "H2受体拮抗剂": "消化系统用药",
    "血管紧张素受体脑啡肽酶抑制剂复方": "心血管系统用药",
    "ARB/脑啡肽酶抑制剂复方": "心血管系统用药",
    "复方降压药": "心血管系统用药",
    "复方降糖药": "内分泌及代谢用药",
    "降糖药复方": "内分泌及代谢用药",
    "感冒复方制剂": "呼吸系统用药",
    "消化酶复方": "消化系统用药",
    "支气管扩张剂复方": "呼吸系统用药",
    "微生态制剂复方": "消化系统用药",
    "微生态复方": "消化系统用药",
    "益生菌": "消化系统用药",
    "抗帕金森病药复方": "神经系统用药",
    "复方抗帕金森药": "神经系统用药",
    "复方氨基酸制剂": "营养支持药",
    "氨基酸/酮酸制剂": "营养支持药",
    "复方氨基酸/酮酸制剂": "营养支持药",
    "疫苗": "生物制品",
    "生物制剂/TNF抑制剂/免疫抑制剂": "免疫调节/抗风湿药",
    "生物制剂/TNF抑制剂/DMARDs": "免疫调节/抗风湿药",
    "止痛药": "神经系统用药",
    "镇痛药": "神经系统用药",
    "阿片类镇痛药": "神经系统用药",
    "非甾体抗炎药": "免疫调节/抗风湿药",
    "糖皮质激素": "免疫调节/抗风湿药",
    "免疫抑制剂/DMARDs": "免疫调节/抗风湿药",
    "免疫抑制剂": "免疫调节/抗风湿药",
    "ACEI/ARB": "心血管系统用药",
    "ARB": "心血管系统用药",
    "钙通道阻滞剂": "心血管系统用药",
    "利尿剂": "泌尿系统用药",
    "降糖药": "内分泌及代谢用药",
    "胰岛素类降糖药": "内分泌及代谢用药",
    "调脂药": "心血管系统用药",
    "抗心绞痛药": "心血管系统用药",
    "抗心律失常药": "心血管系统用药",
    "抗癫痫药": "神经系统用药",
    "抗帕金森病药": "神经系统用药",
    "抗痴呆药": "神经系统用药",
    "抗抑郁药": "精神药品",
    "抗焦虑药": "精神药品",
    "抗眩晕药": "神经系统用药",
    "抗肿瘤药": "抗肿瘤药",
    "抗贫血药": "血液系统用药",
    "祛痰药": "呼吸系统用药",
    "止血药": "血液系统用药",
    "抗纤溶止血药": "血液系统用药",
    "止泻药": "消化系统用药",
    "镇吐药": "消化系统用药",
    "止吐药": "消化系统用药",
    "通便药": "消化系统用药",
    "泻药": "消化系统用药",
    "电解质补充剂": "营养支持药",
    "β2受体激动剂": "呼吸系统用药",
    "支气管扩张剂": "呼吸系统用药",
    "调脂药(他汀)": "心血管系统用药",
    "调脂药(贝特)": "心血管系统用药",
    "调脂药(他汀类)": "心血管系统用药",
    "调脂药(贝特类)": "心血管系统用药",
    "调脂药(PCSK9抑制剂)": "心血管系统用药",
    "调脂药(他汀/贝特)": "心血管系统用药",
    "化疗抗肿瘤药": "抗肿瘤药",
    "靶向抗肿瘤药": "抗肿瘤药",
    "免疫抗肿瘤药": "抗肿瘤药",
    "抗肿瘤药(激素类)": "抗肿瘤药",
}

# 药物大类中 / 表示专业组合词、不应自动拆分的集合
CANONICAL_DRUG_CLASS_KEEP_SLASH = {
    "免疫抑制剂/DMARDs",
    "ACEI/ARB",
}

# 治疗属性：宽泛词 — 出现这些词时在比较中忽略（但保留有意义核心词）
TREATMENT_VAGUE_TERMS = {"对症支持", "不确定"}

CANONICAL_TREATMENT_ATTR = {
    "促红细胞生成": "抗贫血",
    "保肝降酶": "保肝",
    "健脑安神": "安神",
    "健脾利湿": "健脾",
    "健脾和胃": "健脾",
    "健脾开胃": "健脾",
    "健脾止带": "健脾",
    "健脾止泻": "健脾",
    "健脾益智": "健脾",
    "健脾益气": "健脾",
    "健脾益肺": "健脾",
    "健脾调肠": "健脾",
    "免疫抑制": "抗炎、免疫抑制",
    "养心安神": "安神",
    "养血安神": "安神",
    "开窍醒神": "安神",
    "通腑醒神": "安神",
    "滋阴安神": "安神",
    "养阴舒肝": "疏肝理气",
    "凉血止血": "止血",
    "利尿通淋": "利尿",
    "制酸止痛": "抗酸",
    "化痰止咳": "化痰平喘",
    "化瘀消斑": "活血化瘀",
    "化瘀降脂": "活血化瘀",
    "升白细胞": "升高白细胞",
    "升血小板": "促血小板生成",
    "和胃止痛": "行气止痛",
    "安神免疫": "安神",
    "安神定惊": "安神",
    "宣肺化痰": "化痰平喘",
    "宣肺止咳": "化痰平喘",
    "平喘": "化痰平喘",
    "强心": "抗心力衰竭",
    "强脑安神": "安神",
    "扩充血容量": "抗休克、升压",
    "扩容": "抗休克、升压",
    "扩张血管": "改善循环",
    "扩血管": "改善循环",
    "扶正抑瘤": "扶正固本",
    "抗休克": "抗休克、升压",
    "抗帕金森": "抗帕金森病",
    "抗炎消肿": "抗炎止痛",
    "抗酸药": "抗酸",
    "收敛止血": "止血",
    "改善微循环": "改善循环",
    "改善脑循环": "改善脑功能",
    "改善认知": "改善脑功能",
    "改善认知功能": "改善脑功能",
    "改善静脉循环": "改善循环",
    "散寒平喘": "化痰平喘",
    "明目退翳": "明目",
    "止咳": "化痰平喘",
    "止咳化痰": "化痰平喘",
    "止咳平喘": "化痰平喘",
    "止嗽祛痰": "化痰平喘",
    "泻下通便": "泻药",
    "活血利水": "活血化瘀",
    "活血散瘀": "活血化瘀",
    "活血止痛": "活血化瘀",
    "活血消斑": "活血化瘀",
    "活血通络": "活血化瘀",
    "活血通脉": "活血化瘀",
    "消炎镇痛": "抗炎止痛",
    "消积导滞": "助消化",
    "消肿定痛": "抗炎止痛",
    "消肿止痛": "抗炎止痛",
    "消食化滞": "助消化",
    "消食导滞": "助消化",
    "调节肠道菌群": "助消化",
    "润肠通便": "泻药",
    "清热利尿": "清热利湿",
    "清热化湿": "清热利湿",
    "清热排石": "清热利湿",
    "清热止痛": "抗炎止痛",
    "清热止血": "止血",
    "清热祛湿": "清热利湿",
    "清热通便": "泻药",
    "清热通淋": "清热利湿",
    "清热除湿": "清热利湿",
    "清肝明目": "明目",
    "清肺化痰": "化痰平喘",
    "清肺止咳": "化痰平喘",
    "温肺止咳": "化痰平喘",
    "温肾健脾": "补肾",
    "温肾助阳": "补肾",
    "温肾壮阳": "补肾",
    "温肾补脾": "补肾",
    "溶栓": "溶栓降纤",
    "滋养肝肾": "补肾",
    "滋肾养肝": "补肾",
    "滋肾养阴": "补肾",
    "滋补肝肾": "补肾",
    "理气化痰": "行气止痛",
    "理气化瘀": "行气止痛",
    "理气止痛": "行气止痛",
    "理气通便": "行气止痛",
    "通络止痛": "行气止痛",
    "疏肝利胆": "清肝利胆",
    "疏肝宁神": "疏肝理气",
    "疏肝安神": "疏肝理气",
    "疏肝散结": "疏肝理气",
    "疏肝活血": "疏肝理气",
    "疏肝清胃": "疏肝理气",
    "疏肝解郁": "疏肝理气",
    "疏肝降浊": "疏肝理气",
    "疏肝降脂": "疏肝理气",
    "益智安神": "安神",
    "益气安神": "安神",
    "益气扶正": "扶正固本",
    "益气明目": "明目",
    "益肾填精": "补肾",
    "益肾活血": "活血化瘀",
    "化瘀通络": "活血化瘀",
    "化瘀散结": "活血化瘀",
    "消癥化瘀": "活血化瘀",
    "破瘀散结": "活血化瘀",
    "破血消瘀": "活血化瘀",
    "祛痰止咳": "化痰平喘",
    "祛风散寒": "祛风除湿",
    "祛风止痛": "祛风除湿",
    "祛风通窍": "祛风除湿",
    "祛风通络": "祛风除湿",
    "肌松": "抗肌肉痉挛",
    "肝病辅助治疗": "保肝",
    "肠内营养支持": "营养支持",
    "肠外营养支持": "营养支持",
    "舒肝利胆": "清肝利胆",
    "舒肝理气": "疏肝理气",
    "蛇毒血凝酶": "止血",
    "行气温中": "行气止痛",
    "行气降逆": "行气止痛",
    "补气安神": "安神",
    "补益肝肾": "补肾",
    "补肝强腰": "强筋健骨",
    "补肝肾": "补肾",
    "补肾健脑": "补肾",
    "补肾健脾": "补肾",
    "补肾健骨": "补肾",
    "补肾固肾": "补肾",
    "补肾填精": "补肾",
    "补肾壮阳": "补肾",
    "补肾壮骨": "补肾",
    "补肾生精": "补肾",
    "补肾益气": "补肾",
    "解毒化瘀": "解毒",
    "通便": "泻药",
    "镇咳": "化痰平喘",
    "镇咳祛痰": "化痰平喘",
    "镇静安神": "安神",
    "降溶栓": "溶栓降纤",


    # ===== 第二轮：精简中成药治疗属性（244 -> <200）=====
    "升压": "抗休克、升压",
    # 解表 / 疏风（风热外感）
    "疏风散热": "疏风清热",
    "散风清热": "疏风清热",
    "疏散风热": "疏风清热",
    "解表清热": "疏风清热",
    "清热解表": "疏风清热",
    "清热祛风": "疏风清热",
    "疏风宣肺": "疏风清热",
    # 解表化湿（湿邪外感）
    "解表祛湿": "解表化湿",
    "解表通络": "解表化湿",
    "散寒化湿": "解表化湿",
    "祛风止咳": "化痰平喘",
    # 平肝
    "平肝息风": "平肝",
    "平肝潜阳": "平肝",
    "平肝止眩": "平肝",
    "清肝熄风": "平肝",
    "养血平肝": "平肝",
    "熄风通络": "平肝",
    # 养血

    # 补气
    "补益心气": "补气",
    "补肺益气": "补气",
    "益气固表": "补气",
    "益气固脱": "补气",
    "益气复脉": "补气",
    "益气泄浊": "补气",
    "补益心脑": "补气",
    "益气健脾": "健脾",
    # 补血
    "养血柔肝": "益气活血",
    "养血润燥": "益气活血",
    "养血止痒": "益气活血",
    "养血定悸": "益气活血",
    "补气益血": "益气活血",
    "益气补血": "益气活血",
    "益气养血": "益气活血",
    # 温阳散寒
    "益气温阳": "温阳散寒",
    "益气通阳": "温阳散寒",
    "温经散寒": "温阳散寒",
    "温中散寒": "温阳散寒",
    "回阳救逆": "温阳散寒",
    "宽胸通阳": "温阳散寒",
    "温中和胃": "温阳散寒",
    # 散结
    "消癥散结": "散结",
    "化痰散结": "散结",
    "软坚散结": "散结",
    # 祛湿 / 化浊
    "祛湿消滞": "祛湿",
    "降浊祛湿": "祛湿",
    "化浊排毒": "祛湿",
    "通腑降浊": "祛湿",
    "益肾化湿": "祛湿",
    "益肾利湿": "祛湿",
    "清热燥湿": "祛湿",
    # 清热解毒
    "清热消炎": "清热解毒",
    "清瘟解毒": "清热解毒",
    "消炎解毒": "清热解毒",
    "凉血解毒": "清热解毒",
    "清热化瘀": "清热解毒",
    # 清热（裸）
    "清热凉血": "清热",
    "清热理气": "清热",
    "清热止带": "清热",
    "清热固龈": "清热",
    "清肝泻火": "清热",
    # 化痰
    "清热化痰": "化痰平喘",
    # 安神
    "滋养心阴": "安神",
    # 消化
    "和胃降逆": "止吐",
    "养胃": "胃黏膜保护",
    # 抗炎止痛
    "消肿": "抗炎止痛",
    "消肿止痒": "抗炎止痛",
    "消炎止痒": "抗炎止痛",
    "活血消肿": "活血化瘀",
    # 调脂
    "降脂减肥": "调脂",
    # 抗过敏
    "抗瘙痒": "抗过敏",
    "祛风止痒":"抗过敏",
    "杀虫止痒":"抗过敏",

    # 活血通络
    "舒筋活络": "祛风除湿",
    # 辅助 / 支持 → 主分类
    "抗炎辅助": "抗炎",
    "脑卒中辅助治疗": "改善脑功能",
    "麻醉辅助": "麻醉",
    "利胆": "清肝利胆",
    "促进神经修复": "营养神经",
    "补充维生素K": "止血",
    "补充钙、锌": "补充钙",
    "养气养阴": "益气养阴",
    "抗血管生成": "靶向抗肿瘤",
    "预防破伤风": "被动免疫",
    "抗脑血管痉挛": "解痉",
    "促卵泡发育": "促排卵",
    "清肝降压": "降压",
}

def normalize_treatment_attr_value(val):
    """治疗属性同类不同名归一化：拆分并列 -> 别名映射 -> 去重 -> 用、重接。"""
    if not val:
        return val
    s = str(val).strip()
    parts = [p.strip() for p in s.replace("，", "、").replace(",", "、").split("、") if p.strip()]
    flat = []
    for p in parts:
        mapped = CANONICAL_TREATMENT_ATTR.get(p, p)
        for sub in mapped.split("、"):
            sub = sub.strip()
            # 靶向抗肿瘤(…) 及同义变体统一为“靶向抗肿瘤”
            sub = re.sub(r"^靶向抗肿瘤(\(.*\))?$", "靶向抗肿瘤", sub)
            if sub and sub not in flat:
                flat.append(sub)
    return "、".join(flat) if flat else s


# =========================
# 2. 测试开关
# =========================

TEST_MODE = False
# True：只测试前 TEST_N 行
# False：全量运行

TEST_N = 20

# =========================
# 3. 缓存开关
# =========================

USE_CACHE = True
# True：优先使用缓存，已经处理过的中药不重复调用API
# False：不读取缓存，每次都重新调用API

SAVE_CACHE = True
# True：本次运行结果写入缓存
# False：不保存缓存，适合临时测试

CLEAR_CACHE_BEFORE_RUN = False
# True：运行前清空旧缓存
# False：保留旧缓存

SLEEP_SECONDS = 0.5


# =========================
# 4. 文件与缓存函数
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

    raise ValueError(f"无法识别CSV文件编码：{file_path}")


def load_cache(cache_file):
    """
    读取缓存文件
    """
    cache_path = Path(cache_file)

    if CLEAR_CACHE_BEFORE_RUN:
        print("已开启 CLEAR_CACHE_BEFORE_RUN，本次运行将清空旧缓存。")
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
    """
    保存缓存文件
    """
    cache_path = Path(cache_file)
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def clean_cache_key(name):
    """
    缓存键清洗：避免空格、全角空格造成重复查询
    """
    if pd.isna(name):
        return ""

    name = str(name).strip()
    name = name.replace(" ", "")
    name = name.replace("　", "")

    return name


# =========================
# 5. JSON解析函数
# =========================

def extract_json(content):
    """
    兼容模型返回 ```json ... ``` 或纯 JSON 的情况
    """
    if not isinstance(content, str):
        raise ValueError("API返回内容不是字符串")

    content = content.strip()

    # 提取 Markdown 代码块中的 JSON
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", content, re.S)
    if match:
        content = match.group(1).strip()

    # 提取第一个 JSON 对象
    match = re.search(r"\{.*\}", content, re.S)
    if match:
        content = match.group(0).strip()

    return json.loads(content)


# =========================
# 6. API查询函数
# =========================

_fallback_cache = None

VAGUE_TREATMENT = {"对症支持", "对症治疗", "不确定"}


def _clean_treatment_attr(val):
    """去掉治疗属性中的宽泛词，全为宽泛词时返回空字符串。"""
    v = str(val).strip() if val else ""
    if not v:
        return v
    parts = [p.strip() for p in v.replace("、", ",").replace("，", ",").split(",") if p.strip()]
    core = [p for p in parts if p not in VAGUE_TREATMENT]
    return "、".join(core) if core else ""

def _get_fallback_dict():
    """加载本地补充字典（懒加载，只读一次）。"""
    global _fallback_cache
    if _fallback_cache is not None:
        return _fallback_cache

    fallback_path = Path(FALLBACK_FILE)
    if not fallback_path.exists():
        print(f"本地补充字典不存在：{fallback_path}，跳过。")
        _fallback_cache = {}
        return _fallback_cache

    try:
        df = read_csv_safely(fallback_path) if fallback_path.suffix == ".csv" else pd.read_excel(fallback_path)
    except Exception as e:
        print(f"读取本地补充字典失败：{e}，跳过。")
        _fallback_cache = {}
        return _fallback_cache

    if "名称" not in df.columns:
        print("本地补充字典缺少\"名称\"列，跳过。")
        _fallback_cache = {}
        return _fallback_cache

    _fallback_cache = {}
    for _, row in df.iterrows():
        key = clean_cache_key(str(row["名称"]))
        if not key:
            continue
        _fallback_cache[key] = {
            "标准药品名": str(row.get("标准药品名", "")).strip() if pd.notna(row.get("标准药品名")) else "",
            "药品类型": str(row.get("药品类型", "")).strip() if pd.notna(row.get("药品类型")) else "",
            "是否复方": str(row.get("是否复方", "")).strip() if pd.notna(row.get("是否复方")) else "",
            "有效成分": str(row.get("有效成分", "")).strip() if pd.notna(row.get("有效成分")) else "",
            "组成药味": str(row.get("组成药味", "")).strip() if pd.notna(row.get("组成药味")) else "",
            "药物大类": str(row.get("药物大类", "")).strip() if pd.notna(row.get("药物大类")) else "",
            "功效类别": str(row.get("功效类别", "")).strip() if pd.notna(row.get("功效类别")) else "",
            "治疗属性": str(row.get("治疗属性", "")).strip() if pd.notna(row.get("治疗属性")) else "",
        }

    print(f"已加载本地补充字典：{len(_fallback_cache)} 条")
    return _fallback_cache


def get_standard_info(name, cache):
    """
    获取药品标准化信息：
    1. 标准药品名
    2. 药品类型
    3. 是否复方
    4. 有效成分
    5. 组成药味
    6. 药物大类
    7. 功效类别
    8. 治疗属性
    """
    if pd.isna(name) or str(name).strip() == "":
        return "", "不确定", "不确定", "", "", "不确定", "", "不确定"

    raw_name = str(name).strip()
    cache_key = clean_cache_key(raw_name)

    # ---------- 0. 查本地补充字典 ----------
    fallback = _get_fallback_dict()
    if cache_key in fallback:
        item = fallback[cache_key]
        fb = lambda k: str(item.get(k, "")).strip()
        print(
            f"使用本地字典：{raw_name} -> "
            f"{fb('标准药品名')} | {fb('药品类型')} | {fb('是否复方')} | "
            f"{fb('有效成分')} | {fb('组成药味')} | {fb('药物大类')} | "
            f"{fb('功效类别')} | {fb('治疗属性')}"
        )
        ta = _clean_treatment_attr(fb("治疗属性"))
        return (
            fb("标准药品名"), fb("药品类型"), fb("是否复方"),
            fb("有效成分"), fb("组成药味"), fb("药物大类"),
            fb("功效类别"), ta,
        )

    # ---------- 1. 读取缓存 ----------
    if USE_CACHE and cache_key in cache:
        item = cache[cache_key]

        cached_standard_name = str(item.get("标准药品名", "")).strip()
        cached_drug_type = str(item.get("药品类型", "")).strip()
        cached_is_compound = str(item.get("是否复方", "")).strip()
        cached_ingredient = str(item.get("有效成分", "")).strip()
        cached_herb_composition = str(item.get("组成药味", "")).strip()
        cached_drug_class = str(item.get("药物大类", "")).strip()
        cached_effect_class = str(item.get("功效类别", "")).strip()
        cached_treatment_attr = str(item.get("治疗属性", "")).strip()

        # 核心字段完整且非"不确定"才使用缓存
        if (cached_standard_name
                and cached_drug_type and cached_drug_type != "不确定"
                and cached_is_compound
                and cached_drug_class and cached_drug_class != "不确定"
                and cached_treatment_attr and cached_treatment_attr != "不确定"):
            cached_ta = _clean_treatment_attr(cached_treatment_attr)
            print(
                f"使用缓存：{raw_name} -> "
                f"{cached_standard_name} | {cached_drug_type} | {cached_is_compound} | "
                f"{cached_ingredient} | {cached_herb_composition} | "
                f"{cached_drug_class} | {cached_effect_class} | {cached_ta}"
            )
            return (
                cached_standard_name,
                cached_drug_type,
                cached_is_compound,
                cached_ingredient,
                cached_herb_composition,
                cached_drug_class,
                cached_effect_class,
                cached_ta,
            )

        print(f"缓存中“{raw_name}”结果不完整，将重新调用API。")

    # ---------- 2. 调用 API ----------
    prompt = (
        f'药品标准化任务：将”{raw_name}”标准化为 JSON，八个字段如下：\n'
        f'\n'
        f'1.”标准药品名”\n'
        f'  去除剂型(片/胶囊/颗粒/注射液/注射用/吸入用/口服液/分散片/缓释片/肠溶片/滴丸/丸/散/膏/溶液/混悬液)及规格/厂家/包装/商品名。\n'
        f'  单方西药→核心药名(去盐/酯)  复方西药→保留”/”分隔各成分(如”头孢哌酮/他唑巴坦”)\n'
        f'  中成药→去剂型通用名(如”正清风痛宁缓释片”→”正清风痛宁”、”百令胶囊”→”百令”)\n'
        f'\n'
        f'2.”药品类型”  枚举：西药 | 中成药 | 生物制品 | 治疗项目 | 不确定\n'
        f'  化学药/抗菌药/激素/降压/降糖/调脂→西药  疫苗/单克隆抗体/融合蛋白/重组细胞因子与生长因子/肉毒毒素/血液制品等生物大分子→生物制品  放疗/透析/操作→治疗项目\n'
        f'\n'
        f'3.”是否复方”  枚举：是 | 否 | 不确定\n'
        f'  单方西药→否  复方西药→是  中成药→是  疫苗/单抗→否\n'
        f'\n'
        f'4.”有效成分”  仅西药、生物制品填写，中成药/治疗项目留空\n'
        f'  单方→活性主体  复方→全部主要成分(用”、”分隔)\n'
        f'  抗菌复方须列酶抑制剂  感冒复方须列解热镇痛/抗病毒/抗过敏等全部成分\n'
        f'\n'
        f'5.”组成药味”  仅中成药填写，西药/生物制品留空  无法判断→”不确定”\n'
        f'  中成药→主要药味或来源成分(用”、”分隔)，单来源也填(如百令胶囊→”发酵冬虫夏草菌粉”)\n'
        f'\n'
        f'6.”药物大类”  按药理机制/治疗领域划分，输出一级系统分类(粗分类，具体机制/子类放治疗属性字段)。\n'
        f'  一级系统分类：抗感染用药 | 心血管系统用药 | 呼吸系统用药 | 消化系统用药 | 泌尿系统用药 | 血液系统用药 | 内分泌及代谢用药 | 神经系统用药 | 抗肿瘤药 | 免疫调节/抗风湿药 | 营养支持药 | 眼科用药 | 耳鼻喉科用药 | 皮肤科用药 | 骨骼肌肉系统用药 | 麻醉药 | 精神药品 | 抗组胺药 | 解毒药 | 放射性诊断用药 | 生物制品 | 体外诊断/血液保存试剂\n'
        f'  抗肿瘤药→药物大类统一填”抗肿瘤药”，不再细分化疗/靶向/免疫/激素(该细分放治疗属性字段)\n'
        f'  血液系统用药主要调控血液本身的组成与功能（如凝血、造血），而心血管系统用药主要调控心脏和血管的动力与状态（如血压、心律）。\n'
        f'  复方西药→按核心作用成分归类，药物大类不标注复方(是否复方字段已单独记录)；中西药复方含明确化学药核心成分→按化学药归类\n'
        f'  生物制品按治疗领域归入对应一级分类(疫苗→生物制品，胰岛素→内分泌及代谢用药，促红细胞生成素→血液系统用药，单抗/融合蛋白→按适应症归相应系统)  治疗项目→”治疗项目”  无法判断→”不确定”\n'
        f'  中成药→按系统/科室分类(与西药一级系统接轨，按功能主治首位适应症归一类)：心血管系统用药 | 神经系统用药 | 呼吸系统用药 | 消化系统用药 | 泌尿系统用药 | 内分泌代谢系统用药 | 骨骼肌肉系统用药 | 妇产科用药 | 皮肤科用药 | 耳鼻喉科眼科用药 | 抗肿瘤用药 | 外科肛肠科用药 | 补益药 | 口腔科用药 | 儿科用药\n'
        f'  中成药系统归类参考(功效→科室)：活血化瘀/益气通脉→心血管系统用药；凉开/温开/镇惊安神→神经系统用药；清肺/止咳化痰/平喘→呼吸系统用药；疏肝/健脾/消食/通便/止泻/保肝→消化系统用药；利尿通淋/益肾/清热利湿→泌尿系统用药；益气养阴/化痰散结/降糖→内分泌代谢系统用药；祛风除湿/活血通络/强筋健骨→骨骼肌肉系统用药；调经/止带/产后→妇产科用药；祛风止痒/清热利湿/凉血解毒→皮肤科用药；清利咽喉/通鼻窍/清肝明目→耳鼻喉科眼科用药；消癥散结→抗肿瘤用药；解毒消痈/生肌收口→外科肛肠科用药；补气/养血/滋阴/温阳→补益药；清泻胃火/消肿止痛→口腔科用药；健脾和胃/止咳化痰/清热解毒→儿科用药\n'
        f'\n'
        f'7.”功效类别”  仅中成药填写，西药/生物制品留空  无法判断→”不确定”\n'
        f'  以四字格(可八字格组合)描述核心功效，按”核心功效在前、辅助功效在后”排列，多个功效用”、”分隔；功效必须与主治/适应症一致，不写无关功效。\n'
        f'  标准四字格按功效大类选用：清热类→清热解毒/清热燥湿/清热凉血/清热泻火/清退虚热；温里类→温中散寒/回阳救逆；解表类→发散风寒/疏散风热；补气类→益气健脾/补气固表/大补元气；补血类→养血补血/滋阴养血；补阳类→温肾助阳/补肾壮阳；补阴类→滋阴润肺/滋养肝肾；理气类→疏肝理气/行气止痛；活血化瘀类→活血化瘀/化瘀通络/活血通经；止血类→凉血止血/化瘀止血/收敛止血；收涩类→固精缩尿/涩肠止泻；化湿类→芳香化湿/苦温燥湿；利水渗湿类→利水消肿/利尿通淋/利湿退黄；祛风湿类→祛风除湿/通络止痛/强筋健骨；消食类→消食导滞/健脾消食；化痰止咳平喘类→燥湿化痰/润肺止咳/降气平喘；泻下类→润肠通便/峻下逐水；安神类→养心安神/镇惊安神；平肝息风类→平肝潜阳/息风定惊；开窍类→开窍醒神/豁痰开窍；驱虫类→杀虫消积\n'
        f'  炮制品按炮制归类：炒炭→止血类、蜜炙→润肺止咳类、酒制/醋制/盐制/姜制→类别不变\n'
        f'  示例：丹参→”活血化瘀、清热凉血”；三七→”止血、活血化瘀”；黄芪→”补气、利水消肿”；百令胶囊→”补肺肾、益精气”；脑心通胶囊→”益气活血、化瘀通络”；连花清瘟胶囊→”清瘟解毒、宣肺泄热”\n'
        f'\n'
        f'8.”治疗属性”  从药理机制/适应症提取核心作用短词(2-6字)，可用”、”并列(≤2个核心作用)。\n'
        f'  西药常见短词：抗感染 | 抗真菌 | 抗病毒 | 抗结核 | 降压 | 调脂 | 抗心绞痛 | 抗心力衰竭 | 抗心律失常 | 利尿 | 抗凝 | 抗血小板 | 溶栓/降纤 | 止血 | 平喘 | 祛痰 | 镇咳 | 抑酸药 | 胃黏膜保护 | 抗酸 | 促胃动力 | 解痉 | 止吐 | 止泻 | 泻药 | 调节肠道菌群 | 保肝解毒 | 助消化 | 降尿酸 | 碱化尿液 | 降钾 | 抗前列腺增生 | 抗贫血 | 升血小板 | 促血小板生成 | 升高白细胞 | 降血糖 | 抗骨质疏松 | 抗甲亢 | 抗精神病 | 抗抑郁 | 抗焦虑 | 镇静催眠 | 抗癫痫 | 抗帕金森 | 改善脑功能 | 改善认知 | 解热镇痛 | 抗炎止痛 | 镇痛 | 肌松 | 局部麻醉 | 抗过敏 | 免疫抑制 | 免疫调节 | 营养支持 | 补充维生素 | 补充电解质 | 促进组织修复 | 解毒 | 化疗保护 等\n'
        f'  电解质类药品治疗属性统一用”补充电解质”，不用”电解质补充”\n'
        f'  抗肿瘤必须细分：化疗抗肿瘤 / 靶向抗肿瘤 / 免疫抗肿瘤 / 激素抗肿瘤 / 抗肿瘤辅助(可加靶点，如”靶向抗肿瘤(EGFR-TKI)”)\n'
        f'  生物制品：预防接种(具体疾病) / 具体治疗用途(单抗/融合蛋白/细胞因子)\n'
        f'  中成药→从功效类别提取核心短词(2-4字，取首组四字格或最具代表性组合)：益气活血，化瘀通络→益气活血；清热利湿，化瘀散结→清热化瘀；清瘟解毒，宣肺泄热→清瘟解毒；清热解毒，镇惊安神→清热镇惊\n'
        f'  治疗属性统一用标准词(避免同义变体)：止咳化痰/宣肺止咳/镇咳祛痰→化痰平喘；补肾壮阳/滋肾养阴/温肾壮阳→补肾；健脾益气/健脾和胃/健脾利湿→健脾；疏肝解郁/舒肝理气/疏肝散结→疏肝理气；养心安神/镇静安神/益气安神→安神；益气扶正→扶正固本；升白细胞→升高白细胞；升血小板→促血小板生成；肠内营养支持/肠外营养支持→营养支持；肌松→抗肌肉痉挛；通便/润肠通便→泻药；抗帕金森→抗帕金森病；抗酸药→抗酸\n'
        f'  无法判断→”不确定”\n'
        f'\n'
        f'特殊规则：\n'
        f'· 复方制剂：是否复方=是，有效成分列全部活性成分\n'
        f'· 中西药复方：含明确化学药核心成分且为主要作用者→按化学药归类(如复方环磷酰胺片→抗肿瘤药)\n'
        f'· 医院制剂/院内中成药：按临床用途归入对应系统大类，功效参考说明书；信息不完整时标注”不确定”，不虚构成分\n'
        f'· 放射性药物：放射性诊断用药 / 放射性治疗用药\n'
        f'· 体外诊断/血液保存试剂：体外诊断试剂 / 血液保存剂，注明”不用于临床治疗”\n'
        f'\n'
        f'示例：\n'
        f'甲氨蝶呤片 -> {{”标准药品名”:”甲氨蝶呤”,”药品类型”:”西药”,”是否复方”:”否”,”有效成分”:”甲氨蝶呤”,”组成药味”:””,”药物大类”:”免疫抑制剂/DMARDs”,”功效类别”:””,”治疗属性”:”抗炎、免疫抑制”}}\n'
        f'苯磺酸氨氯地平片 -> {{”标准药品名”:”氨氯地平”,”药品类型”:”西药”,”是否复方”:”否”,”有效成分”:”氨氯地平”,”组成药味”:””,”药物大类”:”钙通道阻滞剂”,”功效类别”:””,”治疗属性”:”降压”}}\n'
        f'注射用头孢哌酮钠他唑巴坦钠 -> {{”标准药品名”:”头孢哌酮他唑巴坦”,”药品类型”:”西药”,”是否复方”:”是”,”有效成分”:”头孢哌酮、他唑巴坦”,”组成药味”:””,”药物大类”:”抗菌药”,”功效类别”:””,”治疗属性”:”抗感染”}}\n'
        f'头孢克肟胶囊 -> {{”标准药品名”:”头孢克肟”,”药品类型”:”西药”,”是否复方”:”否”,”有效成分”:”头孢克肟”,”组成药味”:””,“药物大类”:”头孢菌素类抗菌药”,”功效类别”:””,”治疗属性”:”抗感染”}}\n'
        f'达格列净片 -> {{”标准药品名”:”达格列净”,”药品类型”:”西药”,”是否复方”:”否”,”有效成分”:”达格列净”,”组成药味”:””,”药物大类”:”降糖药(SGLT-2抑制剂)”,”功效类别”:””,”治疗属性”:”降血糖”}}\n'
        f'曲妥珠单抗注射液 -> {{”标准药品名”:”曲妥珠单抗”,”药品类型”:”生物制品”,”是否复方”:”否”,”有效成分”:”曲妥珠单抗”,”组成药味”:””,”药物大类”:”抗肿瘤药”,”功效类别”:””,”治疗属性”:”靶向抗肿瘤”}}\n'
        f'重组乙型肝炎疫苗 -> {{”标准药品名”:”乙肝疫苗”,”药品类型”:”生物制品”,”是否复方”:”否”,”有效成分”:”重组乙型肝炎表面抗原”,”组成药味”:””,”药物大类”:”生物制品”,”功效类别”:””,”治疗属性”:”预防接种”}}\n'
        f'百令胶囊 -> {{”标准药品名”:”百令”,”药品类型”:”中成药”,”是否复方”:”是”,”有效成分”:””,”组成药味”:”发酵冬虫夏草菌粉”,”药物大类”:”泌尿系统用药”,”功效类别”:”补肺肾、益精气”,”治疗属性”:”补肺益肾”}}\n'
        f'桑菊清解汤 -> {{”标准药品名”:”桑菊清解”,”药品类型”:”中成药”,”是否复方”:”是”,”有效成分”:””,”组成药味”:”桑叶、菊花、杏仁、连翘、薄荷、苦桔梗、生甘草、芦根”,”药物大类”:”呼吸系统用药”,”功效类别”:”疏风清热、宣肺止咳”,”治疗属性”:”疏风清热”}}\n'
        f'\n'
        f'硬约束：只返回 JSON，不用 Markdown；不解释；不输出英文/拉丁名/厂家/规格/用法用量；不虚构；无法判断→标准名保留原名、其余填”不确定”或不适用留空。\n'
        f'\n'
        f'处理：{raw_name}'
    )

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {
                    "role": "system",  # 修复直引号
                    "content": "你是一位严谨的临床药学与中医药标准化专家，擅长西药、中成药、中药饮片的标准化、复方识别和治疗用途归类。"
                },
                {
                    "role": "user",  # 修复直引号
                    "content": prompt
                }
            ],
            stream=False,
            temperature=0.0,
            response_format={"type": "json_object"},  # 修复直引号
            max_tokens=4096,
        )

        content = response.choices[0].message.content

        if not content or not content.strip():
            raise ValueError("API 返回 content 为空")  # 修复直引号

        # 同时兼容纯 JSON 和 ```json ... ``` 形式
        data = extract_json(content)

        if not isinstance(data, dict):
            raise ValueError("API 返回结果不是 JSON 对象")

    except Exception as e:
        print(f"处理“{raw_name}”时 API 调用或解析失败：{e}")
        return (
            raw_name,
            "不确定",
            "不确定",
            "",
            "",
            "不确定",
            "",
            "不确定"
        )

    # 避免 JSON 中的 null 被转换成字符串 "None"
    def get_text(field, default=""):
        value = data.get(field, default)
        if value is None:
            return default
        return str(value).strip()

    standard_name = get_text("标准药品名", raw_name)
    drug_type = get_text("药品类型", "不确定")
    is_compound = get_text("是否复方", "不确定")
    ingredient = get_text("有效成分", "")
    herb_composition = get_text("组成药味", "")
    drug_class = get_text("药物大类", "不确定")
    effect_class = get_text("功效类别", "")
    treatment_attr = get_text("治疗属性", "不确定")

    valid_types = ["西药", "中成药", "生物制品", "治疗项目", "不确定"]
    if drug_type not in valid_types:
        drug_type = "不确定"

    valid_compound = ["是", "否", "不确定"]
    if is_compound not in valid_compound:
        is_compound = "不确定"

    if not standard_name:
        standard_name = raw_name

    if not drug_class:
        drug_class = "不确定"

    if not treatment_attr:
        treatment_attr = "不确定"

    # 清洗治疗属性
    treatment_attr = _clean_treatment_attr(treatment_attr)

    # 西药、生物制品不填写组成药味和功效类别
    if drug_type in ("西药", "生物制品"):
        herb_composition = ""
        effect_class = ""

    # 中成药不填写有效成分
    elif drug_type == "中成药":
        ingredient = ""

    # 治疗项目不填写药物成分信息
    elif drug_type == "治疗项目":
        ingredient = ""
        herb_composition = ""
        effect_class = ""

    if SAVE_CACHE:
        # 药品类型为"不确定"的不缓存（API 未能识别，下次重试）
        if drug_type != "不确定":
            cache[cache_key] = {
                "原始名": raw_name,
                "标准药品名": standard_name,
                "药品类型": drug_type,
                "是否复方": is_compound,
                "有效成分": ingredient,
                "组成药味": herb_composition,
                "药物大类": drug_class,
                "功效类别": effect_class,
                "治疗属性": treatment_attr
            }
            save_cache(cache, CACHE_FILE)
            print(
                f"已写入缓存：{raw_name} -> "
                f"{standard_name} | {drug_type} | {is_compound} | "
                f"{ingredient} | {herb_composition} | {drug_class} | "
                f"{effect_class} | {treatment_attr}"
            )
        else:
            print(f"跳过缓存：{raw_name} 药品类型\"不确定\"，API 未能识别")

    return (
        standard_name,
        drug_type,
        is_compound,
        ingredient,
        herb_composition,
        drug_class,
        effect_class,
        treatment_attr
    )

# =========================
# 7. 自动识别列名
# =========================

def detect_columns(df):
    """
    自动识别名称列和频数列
    """
    name_col = None
    freq_col = None

    for col in df.columns:
        col_str = str(col).strip()

        if name_col is None and (
            "名称" in col_str
            or "药名" in col_str
            or "name" in col_str.lower()
            or "中药" in col_str
            or "药材" in col_str
        ):
            name_col = col

        if freq_col is None and (
            "频数" in col_str
            or "次数" in col_str
            or "freq" in col_str.lower()
            or "frequency" in col_str.lower()
            or "count" in col_str.lower()
        ):
            freq_col = col

    if name_col is None:
        raise ValueError("未找到名称列，请确保列名包含：名称、name、中药 或 药材")

    if freq_col is None:
        raise ValueError("未找到频数列，请确保列名包含：频数、次数、freq、frequency 或 count")

    return name_col, freq_col


# =========================
# 8. 收集唯一药品名（阶段1）
# =========================

def collect_unique_names(queue_config):
    """
    阶段1：从所有队列的 Drug_only.csv 中收集唯一药品名
    返回 {清洗后药名: 原始药名} 映射
    """
    name_map = {}  # cleaned_key → raw_name

    for queue_name, paths in queue_config.items():
        input_file = paths["input"]

        if not input_file.exists():
            print(f"警告：{queue_name} 队列文件不存在：{input_file}，跳过。")
            continue

        df = read_csv_safely(input_file)
        name_col, _ = detect_columns(df)

        count = 0
        for raw_name in df[name_col].dropna():
            raw_name = str(raw_name).strip()
            if not raw_name:
                continue
            key = clean_cache_key(raw_name)
            if key and key not in name_map:
                name_map[key] = raw_name
            count += 1

        print(f"{queue_name}：读取 {count} 条记录，累计唯一药名 {len(name_map)} 个")

    print(f"\n总计唯一药品名：{len(name_map)} 个")
    return name_map


# =========================
# 9. 人工审核通过的全局强制规则
# =========================

FORCE_LOCKED_FIELDS = (
    "药品类型",
    "是否复方",
    "有效成分",
    "组成药味",
    "药物大类",
    "功效类别",
    "治疗属性",
)


def load_force_rules():
    """加载人工审核通过的正式规则；规则文件缺失时安全退化为空规则。"""
    try:
        from Drug_force_rules import (
            FORCE_BY_COMMON_NAME,
            FORCE_COMMON_NAME_UNIFY,
        )
    except ImportError as exc:
        print(f"未加载药品强制规则，将按空规则继续：{exc}")
        return {}, {}

    if not isinstance(FORCE_COMMON_NAME_UNIFY, dict):
        raise TypeError("FORCE_COMMON_NAME_UNIFY 必须是字典")
    if not isinstance(FORCE_BY_COMMON_NAME, dict):
        raise TypeError("FORCE_BY_COMMON_NAME 必须是字典")

    return FORCE_COMMON_NAME_UNIFY, FORCE_BY_COMMON_NAME


def resolve_forced_common_name(standard_name, name_rules):
    """解析名称统一规则链，并阻止循环规则进入生产结果。"""
    current = "" if standard_name is None else str(standard_name).strip()
    visited = []

    while current in name_rules:
        if current in visited:
            cycle = " -> ".join(visited + [current])
            raise ValueError(f"FORCE_COMMON_NAME_UNIFY 存在循环：{cycle}")

        visited.append(current)
        target = name_rules[current]
        target = "" if target is None else str(target).strip()

        if not target:
            raise ValueError(f"名称统一规则的目标名称为空：{current}")
        if target == current:
            break
        current = target

    return current


def apply_force_rules(standard_map):
    """
    按人工审核结果执行全局后处理：
    1. FORCE_COMMON_NAME_UNIFY：统一标准药品名；
    2. FORCE_BY_COMMON_NAME：按统一后的标准药品名锁定分类字段。

    未经审核的候选不会出现在正式规则文件中，因此不会被本函数应用。
    """
    name_rules, field_rules = load_force_rules()
    name_change_count = 0
    field_lock_count = 0

    # 预先校验全部名称规则，避免只在命中某条数据时才发现循环。
    for source_name in name_rules:
        resolve_forced_common_name(source_name, name_rules)

    for fields in standard_map.values():
        if not isinstance(fields, dict):
            continue

        original_name = str(fields.get("标准药品名", "")).strip()
        unified_name = resolve_forced_common_name(original_name, name_rules)
        applied_marks = []

        if unified_name and unified_name != original_name:
            fields["标准药品名"] = unified_name
            name_change_count += 1
            applied_marks.append(f"名称统一：{original_name}->{unified_name}")

        common_name = str(fields.get("标准药品名", "")).strip()
        locked_values = field_rules.get(common_name)
        if locked_values is not None:
            if not isinstance(locked_values, dict):
                raise TypeError(f"{common_name} 的 FORCE_BY_COMMON_NAME 规则必须是字典")

            for field in FORCE_LOCKED_FIELDS:
                if field in locked_values:
                    value = locked_values[field]
                    fields[field] = "" if value is None else str(value).strip()

            field_lock_count += 1
            applied_marks.append(f"分类锁定：{common_name}")

        if applied_marks:
            fields["强制规则标记"] = "；".join(applied_marks)

    # 治疗属性同类不同名统一（后处理归并）
    for _fields in standard_map.values():
        if isinstance(_fields, dict) and _fields.get("治疗属性"):
            _fields["治疗属性"] = normalize_treatment_attr_value(_fields["治疗属性"])

    print(
        f"强制规则后处理完成：名称规则 {len(name_rules)} 条，"
        f"分类规则 {len(field_rules)} 条；本次名称修正 {name_change_count} 条，"
        f"分类锁定 {field_lock_count} 条。"
    )
    return standard_map


# =========================
# 10. 术语归一化
# =========================

def normalize_field(field_name, value):
    """将字段值归一化，减少语义等价造成的假冲突。不影响 standard_map 原值。"""
    if not value or not value.strip():
        return value

    v = value.strip()

    # ===== 治疗属性归一化 =====
    if field_name == "治疗属性":
        parts = [p.strip() for p in v.replace("、", ",").replace("，", ",").split(",") if p.strip()]
        core = [p for p in parts if p not in TREATMENT_VAGUE_TERMS]
        if not core:
            return v
        return "、".join(sorted(core))

    # ===== 药物大类归一化 =====
    if field_name == "药物大类":
        # 1. 精确映射优先
        if v in CANONICAL_DRUG_CLASS:
            return CANONICAL_DRUG_CLASS[v]
        # 2. / 分隔的复合词：除保留名单外，取最具体（最长）部分
        if "/" in v and v not in CANONICAL_DRUG_CLASS_KEEP_SLASH:
            parts = [p.strip() for p in v.split("/")]
            return max(parts, key=len)
        return v

    # ===== 功效类别归一化 =====
    if field_name == "功效类别":
        if v == "不确定":
            return ""

    return v


# =========================
# 11. 一致性检查
# =========================

def validate_drug_consistency(standard_map):
    """
    检查 standard_map 中的一致性：
    西药 → 同有效成分是否标准药名/药物大类/治疗属性一致
         同标准药名是否有效成分/药物大类/治疗属性一致
    中成药/饮片 → 同组成药味是否标准药名/药物大类/功效类别/治疗属性一致
                 同标准药名是否组成药味/药物大类/功效类别/治疗属性一致
    返回 (standard_map_with_mark, conflicts_list)
    """
    conflicts = []

    # 只检查非空、非"不确定"的条目
    def is_valid(val):
        return val and val.strip() not in ("", "不确定")

    # ===== 西药 =====
    western_drugs = {
        k: v for k, v in standard_map.items()
        if v.get("药品类型") in ("西药", "生物制品")
    }

    def _norm(fields_dict, *field_names):
        """对指定字段执行归一化，返回元组用于比较。"""
        return tuple(normalize_field(n, fields_dict.get(n, "")) for n in field_names)

    # W1: 同有效成分 → 标准药品名/药物大类/治疗属性
    ingr_map = {}  # 有效成分 → [(cleaned_key, fields)]
    for key, fields in western_drugs.items():
        ingr = fields.get("有效成分", "")
        if ingr and ingr.strip():
            parts = tuple(sorted([p.strip() for p in ingr.replace("、", ",").split(",") if p.strip()]))
            if parts:
                ingr_map.setdefault(parts, []).append((key, fields))

    for parts, items in ingr_map.items():
        if len(items) <= 1:
            continue
        groups = {}
        for key, f in items:
            sig = _norm(f, "标准药品名", "药物大类", "治疗属性")
            groups.setdefault(sig, []).append(key)

        if len(groups) > 1:
            sig_list = list(groups.keys())
            conflicts.append({
                "冲突类型": "西药-同有效成分",
                "分组键": "有效成分",
                "分组值": "、".join(parts),
                "冲突字段": "标准药品名/药物大类/治疗属性",
                "冲突值": " | ".join([f"({s[0]},{s[1]},{s[2]})" for s in sig_list])
            })

    # W2: 同标准药名 → 有效成分/药物大类/治疗属性
    sn_map = {}
    for key, fields in western_drugs.items():
        sn = fields.get("标准药品名", "")
        if is_valid(sn):
            sn_map.setdefault(sn, []).append((key, fields))

    for sn, items in sn_map.items():
        if len(items) <= 1:
            continue
        groups = {}
        for key, f in items:
            sig = _norm(f, "有效成分", "药物大类", "治疗属性")
            groups.setdefault(sig, []).append(key)

        if len(groups) > 1:
            sig_list = list(groups.keys())
            conflicts.append({
                "冲突类型": "西药-同标准药名",
                "分组键": "标准药品名",
                "分组值": sn,
                "冲突字段": "有效成分/药物大类/治疗属性",
                "冲突值": " | ".join([f"({s[0]},{s[1]},{s[2]})" for s in sig_list])
            })

    # ===== 中成药/中药饮片 =====
    tcm_drugs = {
        k: v for k, v in standard_map.items()
        if v.get("药品类型") == "中成药"
    }

    # T1: 同组成药味 → 标准药品名/药物大类/功效类别/治疗属性
    herb_map = {}
    for key, fields in tcm_drugs.items():
        herb = fields.get("组成药味", "")
        if herb and herb.strip() and herb.strip() != "不确定":
            parts = tuple(sorted([p.strip() for p in herb.replace("、", ",").split(",") if p.strip()]))
            if parts:
                herb_map.setdefault(parts, []).append((key, fields))

    for parts, items in herb_map.items():
        if len(items) <= 1:
            continue
        groups = {}
        for key, f in items:
            sig = _norm(f, "标准药品名", "药物大类", "功效类别", "治疗属性")
            groups.setdefault(sig, []).append(key)

        if len(groups) > 1:
            sig_list = list(groups.keys())
            conflicts.append({
                "冲突类型": "中成药-同组成药味",
                "分组键": "组成药味",
                "分组值": "、".join(parts),
                "冲突字段": "标准药品名/药物大类/功效类别/治疗属性",
                "冲突值": " | ".join([f"({s[0]},{s[1]},{s[2]},{s[3]})" for s in sig_list])
            })

    # T2: 同标准药名 → 组成药味/药物大类/功效类别/治疗属性
    tcm_sn_map = {}
    for key, fields in tcm_drugs.items():
        sn = fields.get("标准药品名", "")
        if is_valid(sn):
            tcm_sn_map.setdefault(sn, []).append((key, fields))

    for sn, items in tcm_sn_map.items():
        if len(items) <= 1:
            continue
        groups = {}
        for key, f in items:
            sig = _norm(f, "组成药味", "药物大类", "功效类别", "治疗属性")
            groups.setdefault(sig, []).append(key)

        if len(groups) > 1:
            sig_list = list(groups.keys())
            conflicts.append({
                "冲突类型": "中成药-同标准药名",
                "分组键": "标准药品名",
                "分组值": sn,
                "冲突字段": "组成药味/药物大类/功效类别/治疗属性",
                "冲突值": " | ".join([f"({s[0]},{s[1]},{s[2]},{s[3]})" for s in sig_list])
            })

    # 构建冲突键集合，用于标记
    conflict_keys = set()

    # 重新扫描以标记冲突的键（归一化后比较）
    # W1
    for parts, items in ingr_map.items():
        if len(items) > 1:
            groups = {}
            for key, f in items:
                sig = _norm(f, "标准药品名", "药物大类", "治疗属性")
                groups.setdefault(sig, []).append(key)
            if len(groups) > 1:
                for keys in groups.values():
                    conflict_keys.update(keys)
    # W2
    for sn, items in sn_map.items():
        if len(items) > 1:
            groups = {}
            for key, f in items:
                sig = _norm(f, "有效成分", "药物大类", "治疗属性")
                groups.setdefault(sig, []).append(key)
            if len(groups) > 1:
                for keys in groups.values():
                    conflict_keys.update(keys)
    # T1
    for parts, items in herb_map.items():
        if len(items) > 1:
            groups = {}
            for key, f in items:
                sig = _norm(f, "标准药品名", "药物大类", "功效类别", "治疗属性")
                groups.setdefault(sig, []).append(key)
            if len(groups) > 1:
                for keys in groups.values():
                    conflict_keys.update(keys)
    # T2
    for sn, items in tcm_sn_map.items():
        if len(items) > 1:
            groups = {}
            for key, f in items:
                sig = _norm(f, "组成药味", "药物大类", "功效类别", "治疗属性")
                groups.setdefault(sig, []).append(key)
            if len(groups) > 1:
                for keys in groups.values():
                    conflict_keys.update(keys)

    # 给 standard_map 添加一致性标记
    for key in standard_map:
        standard_map[key]["一致性标记"] = "⚠待复核" if key in conflict_keys else "✓一致"

    return standard_map, conflicts


# =========================
# 12. 主处理函数（阶段1-4 + 一致性检查）
# =========================

def process_all_queues(queue_config, global_map_file):
    """
    阶段1：收集所有队列的唯一药品名
    阶段2：批量标准化唯一名称（API + 缓存）
    一致性检查：验证同有效成分/组成药味/标准药名的关键字段一致性
    阶段3：保存全局标准化映射表（含一致性标记） + 冲突报告
    阶段4：为每个队列生成独立 Drug_map.xlsx
    """
    cache = load_cache(CACHE_FILE)

    # ========== 阶段1：收集唯一药名 ==========
    print("\n" + "=" * 60)
    print("阶段1：收集所有队列的唯一药品名")
    print("=" * 60)

    name_map = collect_unique_names(queue_config)
    # name_map: {cleaned_key → raw_name}

    if TEST_MODE:
        test_keys = list(name_map.keys())[:TEST_N]
        name_map = {k: name_map[k] for k in test_keys}
        print(f"测试模式：仅处理前 {TEST_N} 个唯一药品名。")

    # ========== 阶段2：批量标准化唯一药名 ==========
    print("\n" + "=" * 60)
    print("阶段2：批量标准化唯一药品名")
    print("=" * 60)

    standard_map = {}  # cleaned_key → {原始名, 标准化字段, ...}

    total = len(name_map)
    for i, (clean_key, raw_name) in enumerate(name_map.items(), start=1):
        print(f"\n正在处理 {i}/{total}：{raw_name}")

        (
            standard_name,
            drug_type,
            is_compound,
            ingredient,
            herb_composition,
            drug_class,
            effect_class,
            treatment_attr
        ) = get_standard_info(raw_name, cache)

        standard_map[clean_key] = {
            "原始名": raw_name,
            "标准药品名": standard_name,
            "药品类型": drug_type,
            "是否复方": is_compound,
            "有效成分": ingredient,
            "组成药味": herb_composition,
            "药物大类": drug_class,
            "功效类别": effect_class,
            "治疗属性": treatment_attr
        }

        time.sleep(SLEEP_SECONDS)

    # ========== 全局强制规则后处理 ==========
    print("\n" + "=" * 60)
    print("全局后处理：应用人工审核通过的名称统一与分类锁定规则")
    print("=" * 60)

    standard_map = apply_force_rules(standard_map)

    # ========== 一致性检查 ==========
    print("\n" + "=" * 60)
    print("一致性检查：验证西药/中成药标准化字段一致性")
    print("=" * 60)

    standard_map, conflicts = validate_drug_consistency(standard_map)

    conflict_count = len(conflicts)
    conflict_keys_count = sum(
        1 for v in standard_map.values() if v.get("一致性标记") == "⚠待复核"
    )
    if conflict_count > 0:
        print(f"发现 {conflict_count} 组冲突，涉及 {conflict_keys_count} 条记录（标记为 ⚠待复核）")
    else:
        print("未发现冲突，全部一致 ✓")

    # ========== 阶段3：保存全局映射表 + 冲突报告 ==========
    print("\n" + "=" * 60)
    print("阶段3：保存全局标准化映射表")
    print("=" * 60)

    map_rows = []
    for clean_key, fields in standard_map.items():
        row = {"清洗后药名": clean_key}
        row.update(fields)
        map_rows.append(row)

    map_df = pd.DataFrame(map_rows)
    global_map_path = Path(global_map_file)
    global_map_path.parent.mkdir(parents=True, exist_ok=True)
    map_df.to_csv(global_map_path, index=False, encoding="utf-8-sig")
    print(f"全局映射表已保存至：{global_map_path}（共 {len(map_df)} 条）")

    # 导出"不确定"条目，方便补全到 Drug_fallback.xlsx
    is_tcm = map_df["药品类型"] == "中成药"
    uncertain_mask = (
        (map_df["药品类型"] == "不确定")
        | (map_df["药物大类"] == "不确定")
        | (map_df["是否复方"] == "不确定")
        | (map_df["治疗属性"].isin(["", "不确定"]))
        | (is_tcm & map_df["组成药味"].isin(["", "不确定"]))
        | (is_tcm & map_df["功效类别"].isin(["", "不确定"]))
    )
    uncertain_df = map_df[uncertain_mask].copy()
    if len(uncertain_df) > 0:
        uncertain_out = Path(FALLBACK_FILE).parent / "Drug_uncertain_to_fill.xlsx"
        # 只保留 Fallback 需要的列；fallback 按"名称"列匹配
        uncertain_df = uncertain_df.rename(columns={"清洗后药名": "名称"})
        fallback_cols = ["名称", "标准药品名", "药品类型", "是否复方",
                         "有效成分", "组成药味", "药物大类", "功效类别", "治疗属性"]
        available = [c for c in fallback_cols if c in uncertain_df.columns]
        uncertain_df[available].to_excel(uncertain_out, index=False)
        print(f"待补全条目：{len(uncertain_df)} 条 → {uncertain_out}")
        print("  请打开此文件，补全'不确定'字段后，将内容追加到 Drug_fallback.xlsx")

    # 保存冲突报告
    if conflicts:
        conflicts_df = pd.DataFrame(conflicts)
        conflicts_path = Path(CONFLICTS_FILE)
        conflicts_path.parent.mkdir(parents=True, exist_ok=True)
        conflicts_df.to_csv(conflicts_path, index=False, encoding="utf-8-sig")
        print(f"一致性冲突报告已保存至：{conflicts_path}（共 {len(conflicts_df)} 组冲突）")
    else:
        print("无冲突，跳过冲突报告。")

    # ========== 阶段4：为每个队列生成独立输出 ==========
    print("\n" + "=" * 60)
    print("阶段4：为每个队列生成独立 Drug_map.xlsx")
    print("=" * 60)

    for queue_name, paths in queue_config.items():
        input_file = paths["input"]
        output_file = paths["output"]

        if not input_file.exists():
            print(f"警告：{queue_name} 队列文件不存在：{input_file}，跳过。")
            continue

        print(f"\n--- {queue_name} 队列 ---")
        df = read_csv_safely(input_file)
        name_col, freq_col = detect_columns(df)
        print(f"  输入：{input_file}（{len(df)} 行）")

        # 用 standard_map 查找标准化结果
        standard_names = []
        drug_types = []
        is_compounds = []
        ingredients = []
        herb_compositions = []
        drug_classes = []
        effect_classes = []
        treatment_attrs = []

        for _, row in df.iterrows():
            raw_name = str(row[name_col]).strip()
            key = clean_cache_key(raw_name)

            if key in standard_map:
                fields = standard_map[key]
                standard_names.append(fields["标准药品名"])
                drug_types.append(fields["药品类型"])
                is_compounds.append(fields["是否复方"])
                ingredients.append(fields["有效成分"])
                herb_compositions.append(fields["组成药味"])
                drug_classes.append(fields["药物大类"])
                effect_classes.append(fields["功效类别"])
                treatment_attrs.append(fields["治疗属性"])
            else:
                # 兜底：理论上不应发生
                standard_names.append(raw_name)
                drug_types.append("不确定")
                is_compounds.append("不确定")
                ingredients.append("")
                herb_compositions.append("")
                drug_classes.append("不确定")
                effect_classes.append("")
                treatment_attrs.append("不确定")

        df["标准药品名"] = standard_names
        df["药品类型"] = drug_types
        df["是否复方"] = is_compounds
        df["有效成分"] = ingredients
        df["组成药味"] = herb_compositions
        df["药物大类"] = drug_classes
        df["功效类别"] = effect_classes
        df["治疗属性"] = treatment_attrs

        columns_order = [
            name_col,
            "标准药品名",
            "药品类型",
            "是否复方",
            "有效成分",
            "组成药味",
            "药物大类",
            "功效类别",
            "治疗属性",
            freq_col
        ]
        df = df[columns_order]

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_excel(output_path, index=False)
        print(f"  输出：{output_path}（{len(df)} 行）")

    print(f"\n" + "=" * 60)
    print("全流程完成！")
    print(f"全局映射表：{global_map_path}")
    print(f"缓存文件位置：{CACHE_FILE}")


# =========================
# 13. 运行入口
# =========================

if __name__ == "__main__":
    process_all_queues(QUEUE_CONFIG, GLOBAL_MAP_FILE)
