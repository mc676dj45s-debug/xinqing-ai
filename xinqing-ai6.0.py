# -*- coding: utf-8 -*-
"""
大学生心理健康预警系统
文件命名：xinqing-ai3.0.py
参考论文：《基于人工智能技术的大学生心理健康预警系统设计》黄丽芬

版本说明（v3.3）：
    - 学生端：填写 性别 / 年龄 / 年级 + PHQ-9 + GAD-7 + 心情自述
    - 学生端提交后：只显示"提交成功"，不显示任何得分与风险等级
    - 管理员端：需密码进入，可查看全部记录（含性别/年龄/年级）
    - 管理员端：可在后台修改密码（持久化到 admin_config.json）
"""

import os
import json
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib import font_manager

# =========================================================
# 全局配置
# =========================================================
DATA_FILE = "xinqing_data.csv"
CONFIG_FILE = "admin_config.json"
DEFAULT_PASSWORD = "admin123"   # 首次启动时的默认密码

CSV_COLUMNS = [
    "提交时间",
    "性别", "年龄", "年级",
    "PHQ9_1", "PHQ9_2", "PHQ9_3", "PHQ9_4", "PHQ9_5",
    "PHQ9_6", "PHQ9_7", "PHQ9_8", "PHQ9_9",
    "GAD7_1", "GAD7_2", "GAD7_3", "GAD7_4", "GAD7_5",
    "GAD7_6", "GAD7_7",
    "心情自述",
    "PHQ9总分", "GAD7总分",
    "关键词得分", "风险总分", "风险等级",
]


# =========================================================
# 中文字体
# =========================================================
def _setup_chinese_font():
    candidates = [
        "SimHei", "Microsoft YaHei", "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei", "Noto Sans CJK SC", "Source Han Sans CN",
        "PingFang SC", "Heiti SC", "Arial Unicode MS",
    ]
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


_setup_chinese_font()


# =========================================================
# 管理员密码管理
# =========================================================
def _initial_password() -> str:
    try:
        if hasattr(st, "secrets") and "ADMIN_PASSWORD" in st.secrets:
            return st.secrets["ADMIN_PASSWORD"]
    except Exception:
        pass
    return DEFAULT_PASSWORD


def save_admin_password(new_password: str) -> bool:
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump({"password": new_password}, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"保存密码失败：{e}")
        return False


def load_admin_password() -> str:
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                if data.get("password"):
                    return data["password"]
        except Exception:
            pass
    pwd = _initial_password()
    save_admin_password(pwd)
    return pwd


# =========================================================
# 量表配置
# =========================================================
PHQ9_QUESTIONS = [
    "1. 做事时提不起劲或没有兴趣",
    "2. 感到心情低落、沮丧或绝望",
    "3. 入睡困难、睡不安稳或睡眠过多",
    "4. 感觉疲倦或没有活力",
    "5. 食欲不振或吃太多",
    "6. 觉得自己很糟，或觉得自己很失败，或让自己/家人失望",
    "7. 对事物专注有困难，例如阅读报纸或看电视时",
    "8. 动作或说话速度缓慢到别人已经察觉？或正好相反——烦躁或坐立不安、动来动去的情况更胜于平时",
    "9. 有不如死掉或用某种方式伤害自己的念头",
]

GAD7_QUESTIONS = [
    "1. 感到紧张、焦虑或急切",
    "2. 不能够停止或控制担忧",
    "3. 对各种各样的事情担忧过多",
    "4. 很难放松下来",
    "5. 由于不安而无法静坐",
    "6. 变得容易烦恼或急躁",
    "7. 感到似乎将有可怕的事情发生而害怕",
]

SCALE_OPTIONS = {
    "完全不会（0分）": 0,
    "有几天（1分）": 1,
    "一半以上时间（2分）": 2,
    "几乎每天（3分）": 3,
}

GENDER_OPTIONS = ["男", "女", "其他 / 不愿透露"]
GRADE_OPTIONS = ["大一", "大二", "大三", "大四", "大五", "研究生及以上"]

NEGATIVE_KEYWORDS = [
    "难过", "悲伤", "抑郁", "绝望", "无助", "孤独", "孤单", "焦虑", "紧张",
    "害怕", "恐惧", "失眠", "睡不着", "疲惫", "累", "崩溃", "压力", "烦躁",
    "易怒", "生气", "自责", "内疚", "失败", "没用", "无价值", "厌学", "逃避",
    "不想活", "自杀", "轻生", "伤害自己", "自残", "痛苦", "空虚", "迷茫",
    "麻木", "冷漠", "心慌", "胸闷", "难受",
]


# =========================================================
# 数据层
# =========================================================
def init_data_file():
    if not os.path.exists(DATA_FILE):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(
            DATA_FILE, index=False, encoding="utf-8-sig"
        )
        return
    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
        missing = [c for c in CSV_COLUMNS if c not in df.columns]
        if missing:
            for c in missing:
                df[c] = ""
            df = df[CSV_COLUMNS]
            df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
    except Exception:
        pass


def load_data() -> pd.DataFrame:
    init_data_file()
    try:
        df = pd.read_csv(DATA_FILE, encoding="utf-8-sig")
    except Exception:
        df = pd.DataFrame(columns=CSV_COLUMNS)
    for col in CSV_COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[CSV_COLUMNS]


def save_record(record: dict):
    df = load_data()
    new_row = pd.DataFrame([{col: record.get(col, "") for col in CSV_COLUMNS}])
    df = pd.concat([df, new_row], ignore_index=True)
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")


# =========================================================
# 算法层
# =========================================================
def extract_keyword_score(text: str):
    if not isinstance(text, str) or not text.strip():
        return 0, []
    hits = [kw for kw in NEGATIVE_KEYWORDS if kw in text]
    return min(len(hits), 6), hits


def calc_risk_level(phq9_total: int, gad7_total: int, kw_score: int):
    total = int(phq9_total) + int(gad7_total) + int(kw_score)
    if total >= 20:
        level = "高风险"
    elif total >= 10:
        level = "中风险"
    else:
        level = "低风险"
    return total, level


def apply_phq9_q9_rule(level: str, phq9_q9_score: int):
    note = ""
    if phq9_q9_score >= 2:
        if level != "高风险":
            note = "⚠️ 检测到较强烈的自伤/自杀相关念头（PHQ-9 第9题≥2），已直接判定为高风险。"
        level = "高风险"
    elif phq9_q9_score == 1:
        if level == "低风险":
            level = "中风险"
            note = "⚠️ 检测到自伤/自杀相关念头（PHQ-9 第9题=1），风险等级已上调至中风险。"
    return level, note


LEVEL_COLOR = {"低风险": "#d4edda", "中风险": "#fff3cd", "高风险": "#f8d7da"}
LEVEL_TEXT_COLOR = {"低风险": "#155724", "中风险": "#856404", "高风险": "#721c24"}

INTERVENTION = {
    "低风险": [
        "保持规律作息，保证充足睡眠，坚持适度运动。",
        "继续保持与朋友、家人的良好沟通。",
        "遇到阶段性压力时，可通过呼吸放松、正念冥想等方式调节。",
        "建议每 2-4 周复测一次，持续关注自身情绪变化。",
    ],
    "中风险": [
        "建议主动与辅导员、班主任或信任的老师沟通近期状态。",
        "可预约学校心理健康教育中心进行一次面谈咨询。",
        "记录每日情绪日记，识别情绪波动的触发因素。",
        "减少熬夜与过量咖啡因摄入，保证每日 30 分钟户外活动。",
        "建议每 1-2 周复测一次，若持续加重请及时寻求专业帮助。",
    ],
    "高风险": [
        "强烈建议尽快联系学校心理健康教育中心或专业心理咨询师。",
        "如出现自伤、自杀念头，请立即联系辅导员、家人，或拨打心理援助热线。",
        "全国 24 小时心理援助热线：400-161-9995（希望24热线）。",
        "北京心理危机研究与干预中心热线：010-82951332。",
        "请勿独自承受，及时告知身边可信任的人陪伴你。",
    ],
}

ALERT_MSG = {
    "低风险": "当前心理状态总体良好，请继续保持积极的生活方式。",
    "中风险": "当前存在一定的心理压力，建议主动寻求支持与调节。",
    "高风险": "当前心理风险较高，请务必尽快寻求专业帮助！",
}


# =========================================================
# 页面：系统首页（仅管理员）
# =========================================================
def page_home():
    st.title("🎓 大学生心理健康预警系统（管理后台）")
    st.caption("基于人工智能技术的心理健康风险评估与干预原型系统")

    st.markdown("---")
    st.header("一、项目研究背景")
    st.markdown(
        """
        近年来，大学生心理健康问题日益突出。传统心理健康工作多依赖辅导员观察、
        定期普查等被动方式，存在**发现滞后、覆盖不足、干预不及时**等问题。

        参考论文《基于人工智能技术的大学生心理健康预警系统设计》（黄丽芬）的
        研究思路，本系统围绕 **数据采集与处理、特征提取与表示、预测模型构建、
        预警与提醒、干预与支持、监测与反馈** 六大业务模块构建闭环。
        """
    )

    st.header("二、系统功能模块（六大闭环）")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("**① 数据采集与处理**\n\n学生网页端填写基本信息与量表，数据持久化到本地 CSV。")
        st.markdown("**② 特征提取与表示**\n\nPHQ-9 / GAD-7 得分 + 文本关键词匹配。")
    with col2:
        st.markdown("**③ 预测模型构建**\n\n规则化加权计算风险总分，划分三级风险。")
        st.markdown("**④ 预警与提醒**\n\n彩色区块展示风险等级，给出对应提醒。")
    with col3:
        st.markdown("**⑤ 干预与支持**\n\n按风险等级输出差异化干预建议。")
        st.markdown("**⑥ 监测与反馈**\n\n历史真实记录折线图 + 明细表。")

    st.markdown("---")
    st.header("三、权限说明")
    st.info(
        "🔐 **学生端**：仅可填写并提交心理测评，**看不到**任何得分、风险等级和历史数据。\n\n"
        "🔐 **管理员端**：需密码登录，可查看全部预警结果、含性别/年龄/年级的完整记录与干预建议。"
    )

    st.header("四、免责声明")
    st.error(
        "⚠️ **本系统仅为筛查原型，不能替代专业心理诊断。**\n\n"
        "风险等级基于 PHQ-9、GAD-7 量表得分与关键词规则的简化计算，仅供初筛参考，"
        "不构成任何医学诊断或治疗建议。"
    )

    st.header("五、隐私保护说明")
    st.success(
        "🔒 **所有数据仅保存在本地电脑（或部署服务器本地）。**\n\n"
        "系统不联网、不上传、不采集第三方数据，测评内容以 CSV 文件形式保存在程序目录下。"
        "性别、年龄、年级等基本信息仅管理员可见，且**不采集**姓名、学号、联系方式等身份识别信息。"
    )


# =========================================================
# 页面：心理测评（学生端）
# =========================================================
def page_assess():
    st.title("📝 心理测评")
    st.caption("请根据您**最近两周**的真实感受作答，答案无对错之分。提交后将由老师统一查看。")

    with st.form("assess_form", clear_on_submit=False):
        st.subheader("〇、基本信息（匿名采集）")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            gender = st.selectbox("性别 *", options=["请选择"] + GENDER_OPTIONS)
        with col_b:
            age = st.number_input("年龄 *", min_value=15, max_value=60, value=18, step=1)
        with col_c:
            grade = st.selectbox("年级 *", options=["请选择"] + GRADE_OPTIONS)

        st.markdown("---")
        st.subheader("一、PHQ-9 抑郁量表")
        st.markdown("在过去两周里，您有多少时间受到以下问题困扰？")
        phq9_scores = []
        for i, q in enumerate(PHQ9_QUESTIONS, start=1):
            choice = st.radio(
                q, options=list(SCALE_OPTIONS.keys()),
                index=None, key=f"phq9_{i}", horizontal=True,
            )
            phq9_scores.append(choice)

        st.markdown("---")
        st.subheader("二、GAD-7 焦虑量表")
        st.markdown("在过去两周里，您有多少时间受到以下问题困扰？")
        gad7_scores = []
        for i, q in enumerate(GAD7_QUESTIONS, start=1):
            choice = st.radio(
                q, options=list(SCALE_OPTIONS.keys()),
                index=None, key=f"gad7_{i}", horizontal=True,
            )
            gad7_scores.append(choice)

        st.markdown("---")
        st.subheader("三、近期心理状态自述")
        mood_text = st.text_area(
            "请用一段文字描述您近期的心理状态、压力来源或情绪变化（可留空）：",
            height=140,
            placeholder="例如：最近因为考试压力很大，经常失眠，感到焦虑和疲惫……",
        )

        submitted = st.form_submit_button("✅ 提交测评", use_container_width=True)

    if not submitted:
        return

    # ---------- 校验 ----------
    if gender == "请选择":
        st.warning("⚠️ 请选择性别。")
        return
    if grade == "请选择":
        st.warning("⚠️ 请选择年级。")
        return
    if any(c is None for c in phq9_scores):
        st.warning("⚠️ 请完成 PHQ-9 量表的全部 9 道题目后再提交。")
        return
    if any(c is None for c in gad7_scores):
        st.warning("⚠️ 请完成 GAD-7 量表的全部 7 道题目后再提交。")
        return

    phq9_vals = [SCALE_OPTIONS[c] for c in phq9_scores]
    gad7_vals = [SCALE_OPTIONS[c] for c in gad7_scores]

    phq9_total = sum(phq9_vals)
    gad7_total = sum(gad7_vals)
    kw_score, _ = extract_keyword_score(mood_text)

    risk_total, level = calc_risk_level(phq9_total, gad7_total, kw_score)
    level, _ = apply_phq9_q9_rule(level, phq9_vals[8])

    record = {
        "提交时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "性别": gender,
        "年龄": int(age),
        "年级": grade,
        "PHQ9_1": phq9_vals[0], "PHQ9_2": phq9_vals[1], "PHQ9_3": phq9_vals[2],
        "PHQ9_4": phq9_vals[3], "PHQ9_5": phq9_vals[4], "PHQ9_6": phq9_vals[5],
        "PHQ9_7": phq9_vals[6], "PHQ9_8": phq9_vals[7], "PHQ9_9": phq9_vals[8],
        "GAD7_1": gad7_vals[0], "GAD7_2": gad7_vals[1], "GAD7_3": gad7_vals[2],
        "GAD7_4": gad7_vals[3], "GAD7_5": gad7_vals[4], "GAD7_6": gad7_vals[5],
        "GAD7_7": gad7_vals[6],
        "心情自述": mood_text.replace("\n", " ").strip(),
        "PHQ9总分": phq9_total,
        "GAD7总分": gad7_total,
        "关键词得分": kw_score,
        "风险总分": risk_total,
        "风险等级": level,
    }
    save_record(record)

    st.success("✅ 测评已提交成功，感谢您的配合。")
    st.info(
        "📌 您本次的测评结果将提交至心理健康中心，由专业老师统一查看与跟进。\n\n"
        "如需心理支持，可主动联系学校心理健康教育中心或辅导员。"
    )


# =========================================================
# 页面：风险预警与干预（仅管理员）
# =========================================================
def page_warning():
    st.title("🔔 风险预警与干预（管理后台）")
    st.caption("本页面展示**所有学生真实测评记录**的风险变化趋势、明细与干预建议。")

    df = load_data()

    if df.empty:
        st.info("暂无测评历史，请等待学生完成测评生成数据。")
        st.stop()

    df = df.copy()
    for col in ["PHQ9总分", "GAD7总分", "关键词得分", "风险总分"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    df["提交时间"] = pd.to_datetime(df["提交时间"], errors="coerce")
    df = df.dropna(subset=["提交时间"]).sort_values("提交时间").reset_index(drop=True)

    if df.empty:
        st.info("暂无测评历史，请等待学生完成测评生成数据。")
        st.stop()

    # ---------- 统计概览 ----------
    st.markdown("### 📊 风险等级分布概览")
    level_counts = df["风险等级"].value_counts().reindex(
        ["低风险", "中风险", "高风险"], fill_value=0
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("总测评份数", len(df))
    c2.metric("低风险", int(level_counts["低风险"]))
    c3.metric("中风险", int(level_counts["中风险"]))
    c4.metric("高风险", int(level_counts["高风险"]))

    # ---------- 最新一次预警 ----------
    latest = df.iloc[-1]
    level = str(latest["风险等级"])
    bg = LEVEL_COLOR.get(level, "#eeeeee")
    fg = LEVEL_TEXT_COLOR.get(level, "#333333")

    st.markdown("### 最近一次测评预警")
    st.markdown(
        f"""
        <div style="background-color:{bg}; color:{fg};
                    padding:18px; border-radius:12px;
                    border:2px solid {fg}; margin-bottom:12px;">
            <h3 style="color:{fg}; margin:0;">风险等级：{level}</h3>
            <p style="color:{fg}; margin-top:6px;">
                性别：{latest.get('性别','')}　|　年龄：{latest.get('年龄','')}　|　
                年级：{latest.get('年级','')}
            </p>
            <p style="color:{fg}; margin:0;">
                测评时间：{latest['提交时间'].strftime('%Y-%m-%d %H:%M:%S')}　|　
                风险总分：{int(latest['风险总分'])}
            </p>
            <p style="color:{fg}; margin-top:6px;">{ALERT_MSG.get(level, '')}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### 对应干预建议")
    for tip in INTERVENTION.get(level, []):
        st.markdown(f"- {tip}")

    st.markdown("---")

    # ---------- 折线图 ----------
    st.markdown("### 📈 心理风险分数变化折线图")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    x = df["提交时间"]
    ax.plot(x, df["风险总分"], marker="o", linewidth=2,
            color="#c0392b", label="风险总分")
    ax.plot(x, df["PHQ9总分"], marker="s", linewidth=1.5,
            linestyle="--", color="#4e79a7", label="PHQ-9 总分")
    ax.plot(x, df["GAD7总分"], marker="^", linewidth=1.5,
            linestyle="--", color="#f28e2b", label="GAD-7 总分")
    ax.axhline(10, color="#f1c40f", linestyle=":", linewidth=1.2, label="中风险阈值(10)")
    ax.axhline(20, color="#e74c3c", linestyle=":", linewidth=1.2, label="高风险阈值(20)")
    ax.set_title("心理风险分数变化趋势", fontsize=13)
    ax.set_xlabel("测评时间", fontsize=11)
    ax.set_ylabel("分数", fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=9)
    fig.autofmt_xdate(rotation=30)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    st.markdown("---")

    # ---------- 高风险名单 ----------
    high_df = df[df["风险等级"] == "高风险"]
    if not high_df.empty:
        st.markdown("### 🚨 高风险学生名单")
        high_show = high_df[[
            "提交时间", "性别", "年龄", "年级",
            "PHQ9总分", "GAD7总分", "风险总分", "心情自述"
        ]].copy()
        high_show["提交时间"] = high_show["提交时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
        st.dataframe(high_show, use_container_width=True, hide_index=True)
    else:
        st.success("✅ 当前没有高风险记录。")

    st.markdown("---")

    # ---------- 历史记录表 ----------
    st.markdown("### 📋 全部历史测评记录")
    show_cols = [
        "提交时间", "性别", "年龄", "年级",
        "PHQ9总分", "GAD7总分", "关键词得分", "风险总分", "风险等级"
    ]
    table = df[show_cols].copy()
    table["提交时间"] = table["提交时间"].dt.strftime("%Y-%m-%d %H:%M:%S")
    st.dataframe(table, use_container_width=True, hide_index=True)

    with st.expander("查看完整原始记录（含量表逐题与心情自述）"):
        st.dataframe(df, use_container_width=True, hide_index=True)

    # ---------- 导出 CSV ----------
    st.markdown("### ⬇️ 导出数据")
    csv_bytes = df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")
    st.download_button(
        label="📥 下载全部测评数据（CSV）",
        data=csv_bytes,
        file_name=f"xinqing_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.caption(f"当前共 {len(df)} 条真实测评记录，数据保存在本地文件：`{DATA_FILE}`")


# =========================================================
# 页面：修改管理员密码（仅管理员）
# =========================================================
def page_change_password():
    st.title("🔑 修改管理员密码")
    st.caption("修改后立即生效，请务必牢记新密码。")

    st.warning(
        "⚠️ 提示：密码保存在程序目录下的 `admin_config.json` 文件中。"
        "请妥善保管，并定期备份或修改。"
    )

    with st.form("change_pwd_form", clear_on_submit=True):
        old_pwd = st.text_input("当前密码 *", type="password")
        new_pwd = st.text_input("新密码 *（至少 6 位）", type="password")
        confirm_pwd = st.text_input("确认新密码 *", type="password")
        submitted = st.form_submit_button("✅ 确认修改", use_container_width=True)

    if not submitted:
        return

    current = load_admin_password()

    if not old_pwd:
        st.error("❌ 请输入当前密码。")
        return
    if old_pwd != current:
        st.error("❌ 当前密码不正确。")
        return
    if len(new_pwd) < 6:
        st.error("❌ 新密码长度至少 6 位。")
        return
    if new_pwd != confirm_pwd:
        st.error("❌ 两次输入的新密码不一致。")
        return
    if new_pwd == current:
        st.warning("⚠️ 新密码与当前密码相同，未做修改。")
        return

    if save_admin_password(new_pwd):
        st.success("✅ 密码修改成功！下次登录请使用新密码。")
        st.info(
            "📌 如部署在 Streamlit Cloud，重启应用后密码仍会保留（除非手动删除 "
            "`admin_config.json`）。"
        )


# =========================================================
# 主入口
# =========================================================
def main():
    st.set_page_config(
        page_title="大学生心理健康预警系统",
        page_icon="🧠",
        layout="wide",
    )
    init_data_file()
    load_admin_password()  # 确保首次运行生成 admin_config.json

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    with st.sidebar:
        st.header("👤 身份选择")
        role = st.radio(
            "请选择您的身份：",
            ["🎒 学生（填写测评）", "🔐 管理员（查看后台）"],
            index=0,
        )

        st.markdown("---")

        if role == "🔐 管理员（查看后台）":
            if not st.session_state.is_admin:
                pwd = st.text_input("请输入管理员密码：", type="password")
                if st.button("登录", use_container_width=True):
                    if pwd == load_admin_password():
                        st.session_state.is_admin = True
                        st.success("✅ 登录成功")
                        st.rerun()
                    else:
                        st.error("❌ 密码错误")
            else:
                st.success("✅ 已登录为管理员")
                if st.button("退出登录", use_container_width=True):
                    st.session_state.is_admin = False
                    st.rerun()

        st.markdown("---")
        st.caption("🔒 性别 / 年龄 / 年级仅管理员可见")
        st.caption("📁 数据保存在本地 CSV")

    if role == "🎒 学生（填写测评）":
        page_assess()
        return

    if not st.session_state.is_admin:
        st.title("🔐 管理员登录")
        st.warning("请先在左侧输入管理员密码后访问后台。")
        st.info(
            "说明：学生端仅提供心理测评填写，看不到任何结果与历史数据；"
            "所有预警、干预、图表与明细仅管理员可见。"
        )
        return

    with st.sidebar:
        st.header("🧭 后台导航")
        page = st.radio(
            "请选择页面：",
            ["🏠 系统首页", "🔔 风险预警与干预", "🔑 修改密码"],
            index=0,
        )

    if page == "🏠 系统首页":
        page_home()
    elif page == "🔔 风险预警与干预":
        page_warning()
    else:
        page_change_password()


if __name__ == "__main__":
    main()