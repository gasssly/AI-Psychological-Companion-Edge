import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import time
import time as _time  # ✅ 優化 1：移至頂層，避免在呼吸練習迴圈內重複 import
import re  # ✅ 優化 1：移至頂層，供 Step 標籤清除用
import json
import base64
import asyncio
import datetime as _dt  # ✅ 優化 1：移至頂層，避免在熱圖 if 區塊內重複 import
import nest_asyncio  # 🔒 修正 2：避免 asyncio 與 Streamlit 事件迴圈衝突

nest_asyncio.apply()
import edge_tts
import sqlite3
import pandas as pd
import altair as alt
from datetime import datetime
from PIL import Image
from google.api_core.exceptions import ResourceExhausted
import uuid
import math
from pathlib import Path
import chromadb
import io  # 🆕 PDF：用於處理 bytes 流
import jieba  # 繁體中文斷詞（搭配 OpenCC 轉換使用）
import opencc  # 繁簡轉換，讓 jieba 斷詞更準確
from html import escape as html_escape
from collections import Counter  # 移至頂層，避免在迴圈內重複 import
from concurrent.futures import ThreadPoolExecutor  # ✅ 支援並發檢索記憶與教科書
import ser_bridge  # 🧠 SER：本地 WavLM 語音情緒辨識模型橋接

# 🆕 PDF：使用 reportlab 產生中文 PDF（內建 CID 字體，零外部依賴，雲端部署也能用）
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle  # ✅ 優化 1：移除從未使用的 getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.graphics.shapes import Drawing, String, Rect
from reportlab.graphics.charts.piecharts import Pie
from reportlab.graphics.charts.barcharts import VerticalBarChart
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

# ✅ 優化 1：Step 標籤 regex 只編譯一次，供 handle_user_input 重複使用
_STEP_PATTERN = re.compile(r"【Step [1-4]】")
_STAGE_DIRECTION_PATTERN = re.compile(r"[（(]\s*[^（）()]{0,20}(?:停頓|沉默|深呼吸|微笑|點頭|嚴肅但溫柔)[^（）()]{0,20}\s*[）)]")
_OPTION_HINT_PATTERN = re.compile(r"\s*[（(][^（）()]{1,28}[）)]\s*$")
MIN_MEMORY_SEMANTIC_SCORE = 0.18
RECENT_DAYS = 14
PREFERRED_MODEL = "models/gemini-3.1-flash-lite"
PDF_CLINICAL_CACHE_VERSION = "charts_v2"

EMOTION_SCORE = {
    # 🆕 新版臨床診斷評估標籤
    "正向平靜": 3, "憂鬱低落": -2, "焦慮煩躁": -2,
    "放鬆": 2, "滿足": 2, "有活力": 3,
    "悲傷空虛": -3, "失去興趣": -2, "自責無力": -3,
    "過度擔憂": -2, "身體緊繃": -2, "易怒煩躁": -3,
    # 💾 舊版標籤 (向下相容舊資料庫，不要刪除)
    "開心": 3, "無聊": 0, "孤單": -1, "失落": -2, "難過": -2,
    "煩躁": -1, "委屈": -2, "憤怒": -3, "生氣": -2
}

PDF_KEYWORD_STOPWORDS = {
    "的", "了", "是", "我", "你", "他", "她", "在", "有", "和",
    "就", "都", "也", "很", "不", "一", "個", "這", "那", "但",
    "嗎", "吧", "啊", "喔", "哦", "嗯", "其實", "因為", "所以",
    "然後", "覺得", "感覺", "一直", "已經", "還是", "可以", "沒有",
    "知道", "自己", "什麼", "時候", "為什麼", "這樣", "那樣", ""
}


def clean_assistant_text(text: str) -> str:
    if not text:
        return ""
    return _STAGE_DIRECTION_PATTERN.sub("", text).strip()


def clean_option_display_text(text: str) -> str:
    """隱藏問卷選項尾端的分類提示，保留內部使用的情緒標籤。"""
    return _OPTION_HINT_PATTERN.sub("", str(text)).strip()

# 🆕 PDF：註冊 reportlab 內建繁體中文字體（STSong-Light 是 reportlab 標準內建 CID 字體）
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))

# =========================
# 0. 頁面基礎與資料庫初始化設定
# =========================
st.set_page_config(
    page_title="心靈小幫手 - 阿光",
    page_icon="🧠",
    layout="centered"
)

# =========================
# 🎨 Mood-Aware 全域動態背景
# =========================
MOOD_GROUPS = {
    "calm_positive": {"正向平靜", "放鬆", "滿足", "有活力", "平靜", "開心"},
    "grounding_stress": {"焦慮煩躁", "過度擔憂", "身體緊繃", "易怒煩躁", "焦慮", "憤怒", "生氣", "煩躁"},
    "quiet_low": {"憂鬱低落", "悲傷空虛", "失去興趣", "自責無力", "難過", "疲憊", "孤單", "失落"},
}

MOOD_THEME_STYLES = {
    "calm_positive": {
        "bg_gradient": "linear-gradient(145deg, #D7F5E5 0%, #BFEBD7 42%, #F7FFE8 100%)",
        "sidebar_bg": "#DFF3E9",
        "primary": "#52A882",
        "button_hover": "#3D8F6A",
        "assistant_bg": "#ECFFF6",
        "assistant_border": "#7ECBA8",
        "user_bg": "#E6F7FF",
        "user_border": "#88BEE8",
        "breathe_circle": "radial-gradient(circle, #A8DFC8 0%, #52A882 100%)",
        "text": "#2F4A40",
        "muted": "#657A72",
        "decor_1": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f33f.svg')",
        "decor_2": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f338.svg')",
        "decor_3": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f31e.svg')",
        "decor_opacity": "0.28",
        "decor_filter": "drop-shadow(0 14px 20px rgba(47, 104, 76, 0.18)) saturate(1.08)",
    },
    "grounding_stress": {
        "bg_gradient": "linear-gradient(145deg, #D7E8F8 0%, #C6D7E8 46%, #E8F6FF 100%)",
        "sidebar_bg": "#DDE9F4",
        "primary": "#5BA3D0",
        "button_hover": "#4A8BB8",
        "assistant_bg": "#EAF7FF",
        "assistant_border": "#88BEE8",
        "user_bg": "#EAF4FF",
        "user_border": "#88BEE8",
        "breathe_circle": "radial-gradient(circle, #B0C4DE 0%, #5BA3D0 100%)",
        "text": "#2F4358",
        "muted": "#657386",
        "decor_1": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f4a7.svg')",
        "decor_2": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f30a.svg')",
        "decor_3": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/2693.svg')",
        "decor_opacity": "0.24",
        "decor_filter": "drop-shadow(0 16px 22px rgba(45, 83, 112, 0.18)) saturate(0.98)",
    },
    "quiet_low": {
        "bg_gradient": "linear-gradient(145deg, #E9E4F6 0%, #DDE3F4 42%, #EEF4FA 100%)",
        "sidebar_bg": "#E8E4F4",
        "primary": "#7B6BA8",
        "button_hover": "#6558A0",
        "assistant_bg": "#F4F1FF",
        "assistant_border": "#B0A0D8",
        "user_bg": "#EEF2FA",
        "user_border": "#A0B4D8",
        "breathe_circle": "radial-gradient(circle, #C8C0E8 0%, #7B6BA8 100%)",
        "text": "#555555",
        "muted": "#666A78",
        "decor_1": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f319.svg')",
        "decor_2": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f320.svg')",
        "decor_3": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/1f4ab.svg')",
        "decor_opacity": "0.20",
        "decor_filter": "drop-shadow(0 18px 24px rgba(72, 64, 120, 0.16)) saturate(0.92)",
    },
    "neutral": {
        "bg_gradient": "linear-gradient(145deg, #FFEBD8 0%, #E9E1FF 48%, #DFF1FF 100%)",
        "sidebar_bg": "#F0E8FF",
        "primary": "#8B6BBE",
        "button_hover": "#7558A8",
        "assistant_bg": "#FFF2E9",
        "assistant_border": "#F0A08A",
        "user_bg": "#EAF4FF",
        "user_border": "#88BEE8",
        "breathe_circle": "radial-gradient(circle, #C9B8E8 0%, #8B6BBE 100%)",
        "text": "#302B3F",
        "muted": "#6F687D",
        "decor_1": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/2b50.svg')",
        "decor_2": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/2601.svg')",
        "decor_3": "url('https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg/2728.svg')",
        "decor_opacity": "0.24",
        "decor_filter": "drop-shadow(0 16px 22px rgba(102, 77, 136, 0.16)) saturate(1.02)",
    },
}

TWEMOJI_ASSET_BASE = "https://cdn.jsdelivr.net/gh/jdecked/twemoji@17.0.2/assets/svg"


def _twemoji_asset(codepoint):
    return f"url('{TWEMOJI_ASSET_BASE}/{codepoint}.svg')"


def _emotion_decor(label, codepoints, opacity, filter_value):
    return {
        "source_label": label,
        "decor_1": _twemoji_asset(codepoints[0]),
        "decor_2": _twemoji_asset(codepoints[1]),
        "decor_3": _twemoji_asset(codepoints[2]),
        "decor_opacity": opacity,
        "decor_filter": filter_value,
    }


EMOTION_DECOR_STYLES = {
    "正向平靜": _emotion_decor("正向平靜", ("1f33f", "1f338", "1f31e"), "0.28", "drop-shadow(0 14px 20px rgba(47, 104, 76, 0.18)) saturate(1.08)"),
    "放鬆": _emotion_decor("放鬆", ("1f33f", "2601", "1f375"), "0.26", "drop-shadow(0 14px 20px rgba(62, 112, 89, 0.16)) saturate(1.02)"),
    "滿足": _emotion_decor("滿足", ("2728", "1f338", "1f49b"), "0.27", "drop-shadow(0 14px 22px rgba(132, 99, 34, 0.16)) saturate(1.08)"),
    "有活力": _emotion_decor("有活力", ("1f31e", "1f308", "26a1"), "0.25", "drop-shadow(0 16px 24px rgba(142, 107, 28, 0.16)) saturate(1.12)"),
    "憂鬱低落": _emotion_decor("憂鬱低落", ("1f319", "2601", "1f499"), "0.20", "drop-shadow(0 18px 24px rgba(72, 64, 120, 0.16)) saturate(0.92)"),
    "悲傷空虛": _emotion_decor("悲傷空虛", ("1f319", "1f499", "2601"), "0.19", "drop-shadow(0 18px 24px rgba(66, 86, 128, 0.15)) saturate(0.88)"),
    "失去興趣": _emotion_decor("失去興趣", ("1f940", "1f4ad", "1f319"), "0.18", "drop-shadow(0 18px 24px rgba(86, 76, 122, 0.14)) saturate(0.86)"),
    "自責無力": _emotion_decor("自責無力", ("1f90d", "1f4ab", "1f319"), "0.18", "drop-shadow(0 18px 24px rgba(82, 78, 116, 0.14)) saturate(0.86)"),
    "焦慮煩躁": _emotion_decor("焦慮煩躁", ("1f4a7", "1f30a", "2693"), "0.24", "drop-shadow(0 16px 22px rgba(45, 83, 112, 0.18)) saturate(0.98)"),
    "過度擔憂": _emotion_decor("過度擔憂", ("1f300", "1f4a7", "2614"), "0.23", "drop-shadow(0 16px 22px rgba(48, 81, 116, 0.17)) saturate(0.96)"),
    "身體緊繃": _emotion_decor("身體緊繃", ("2693", "1f30a", "1f9d8"), "0.22", "drop-shadow(0 16px 22px rgba(44, 80, 106, 0.16)) saturate(0.94)"),
    "易怒煩躁": _emotion_decor("易怒煩躁", ("1f525", "26a1", "1f32c-fe0f"), "0.21", "drop-shadow(0 16px 22px rgba(126, 76, 42, 0.15)) saturate(0.96)"),
}

PET_MOOD_STATES = {
    "neutral": {
        "label": "阿光貓貓",
        "messages": [
            "喵，想說的時候我在旁邊。",
            "先不用急，貓貓陪你慢慢來。",
            "摸摸頭，今天也辛苦了。",
        ],
    },
    "calm_positive": {
        "label": "阿光貓貓覺得你比較平穩",
        "messages": [
            "喵嗚，把這份平靜留一點給自己。",
            "尾巴輕輕晃，今天的你很棒。",
            "貓貓窩在旁邊，陪你把好心情留久一點。",
        ],
    },
    "grounding_stress": {
        "label": "阿光貓貓陪你放慢",
        "messages": [
            "喵，先慢慢吐氣，我在。",
            "貓貓蹭一下，肩膀可以放鬆一點點。",
            "不用一次處理完，先跟我一起慢慢呼吸。",
        ],
    },
    "quiet_low": {
        "label": "阿光貓貓安靜陪你",
        "messages": [
            "喵，不用急著變好，先待著也可以。",
            "今天比較沉也沒關係，貓貓靠你旁邊。",
            "讓我輕輕蹭一下，先把自己放柔一點。",
        ],
    },
}


PET_SPRITE_PATH = Path(__file__).with_name("aguang_cat_nobg.png")
PET_WALK_SHEET_PATH = Path(__file__).with_name("aguang_cat_walk_sheet.png")
PET_LIFE_SHEET_PATH = Path(__file__).with_name("aguang_cat_life_sheet.png")


@st.cache_data(show_spinner=False)
def _load_pet_sprite_data_uri(path_str, mtime_ns):
    sprite_bytes = Path(path_str).read_bytes()
    encoded = base64.b64encode(sprite_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def get_pet_sprite_data_uri():
    return _get_pet_image_data_uri(PET_SPRITE_PATH)


def get_pet_walk_sheet_data_uri():
    return _get_pet_image_data_uri(PET_WALK_SHEET_PATH)


def get_pet_life_sheet_data_uri():
    return _get_pet_image_data_uri(PET_LIFE_SHEET_PATH)


def _get_pet_image_data_uri(path):
    try:
        stat = path.stat()
    except OSError:
        return ""
    return _load_pet_sprite_data_uri(str(path), stat.st_mtime_ns)


def _theme_from_label(label):
    label = str(label or "").strip()
    for theme_name, labels in MOOD_GROUPS.items():
        if label in labels:
            return theme_name
    return None


def _recent_journal_emotion_labels():
    try:
        username = st.session_state.get("current_user", globals().get("current_user", "訪客"))
        if not username or username == "訪客":
            return []
        conn = sqlite3.connect("emotion_tracker.db")
        try:
            query = "SELECT date, major, sub FROM emotion_history WHERE username=? ORDER BY date DESC LIMIT 1"
            df = pd.read_sql_query(query, conn, params=(username,))
        finally:
            conn.close()
        if df.empty:
            return []
        parsed_date = pd.to_datetime(df.iloc[0]["date"], errors="coerce")
        if pd.isna(parsed_date):
            return []
        if (_dt.datetime.now() - parsed_date.to_pydatetime()) > _dt.timedelta(days=RECENT_DAYS):
            return []
        return [df.iloc[0].get("sub"), df.iloc[0].get("major")]
    except Exception:
        return []


def resolve_emotion_decor(mood):
    mood = mood if mood in MOOD_THEME_STYLES else "neutral"
    theme = MOOD_THEME_STYLES[mood]
    labels = []

    result = st.session_state.get("result")
    if isinstance(result, (list, tuple)) and len(result) >= 2:
        major, sub = result[0], result[1]
        labels.extend([sub, major])

    labels.extend([
        st.session_state.get("sub"),
        st.session_state.get("major"),
        st.session_state.get("last_emotion"),
    ])

    if not any(str(label or "").strip() for label in labels):
        labels.extend(_recent_journal_emotion_labels())

    for label in labels:
        key = str(label or "").strip()
        if key in EMOTION_DECOR_STYLES:
            return EMOTION_DECOR_STYLES[key]

    return {
        "source_label": mood,
        "decor_1": theme["decor_1"],
        "decor_2": theme["decor_2"],
        "decor_3": theme["decor_3"],
        "decor_opacity": theme["decor_opacity"],
        "decor_filter": theme["decor_filter"],
    }


def resolve_mood_theme():
    """依即時偵測、測驗結果、7 天內最新日記，回傳目前 UI mood theme。"""
    try:
        stress = float(st.session_state.get("last_stress", 0) or 0)
        if stress >= 7:
            return "grounding_stress"

        live_theme = _theme_from_label(st.session_state.get("last_emotion"))
        if live_theme:
            return live_theme

        if st.session_state.get("last_mood_source") == "quiz":
            quiz_theme = _theme_from_label(st.session_state.get("major"))
            if quiz_theme:
                return quiz_theme

        username = st.session_state.get("current_user", globals().get("current_user", "訪客"))
        if not username or username == "訪客":
            return "neutral"

        conn = sqlite3.connect("emotion_tracker.db")
        try:
            query = "SELECT date, major FROM emotion_history WHERE username=? ORDER BY date DESC LIMIT 1"
            row = pd.read_sql_query(query, conn, params=(username,))
        finally:
            conn.close()

        if row.empty:
            return "neutral"

        latest_dt = pd.to_datetime(row.iloc[0]["date"], errors="coerce")
        if pd.isna(latest_dt):
            return "neutral"
        if datetime.now() - latest_dt.to_pydatetime() > _dt.timedelta(days=RECENT_DAYS):
            return "neutral"

        journal_theme = _theme_from_label(row.iloc[0]["major"])
        return journal_theme or "neutral"
    except Exception:
        return "neutral"


def inject_dynamic_css(mood):
    """注入 mood-aware CSS 與 rerun 後的輕量 JS fallback。"""
    mood = mood if mood in MOOD_THEME_STYLES else "neutral"
    theme = MOOD_THEME_STYLES[mood]

    decor = resolve_emotion_decor(mood)

    st.markdown(f"""
<style>
:root {{
    color-scheme: light;
    --aguang-bg-gradient: {theme['bg_gradient']};
    --aguang-sidebar-bg: {theme['sidebar_bg']};
    --aguang-primary: {theme['primary']};
    --aguang-button-hover: {theme['button_hover']};
    --aguang-assistant-bg: {theme['assistant_bg']};
    --aguang-assistant-border: {theme['assistant_border']};
    --aguang-user-bg: {theme['user_bg']};
    --aguang-user-border: {theme['user_border']};
    --aguang-breathe-circle: {theme['breathe_circle']};
    --aguang-text: {theme['text']};
    --aguang-muted: {theme['muted']};
    --aguang-decor-1: {decor['decor_1']};
    --aguang-decor-2: {decor['decor_2']};
    --aguang-decor-3: {decor['decor_3']};
    --aguang-decor-opacity: {decor['decor_opacity']};
    --aguang-decor-filter: {decor['decor_filter']};
    --aguang-panel: color-mix(in srgb, var(--aguang-primary) 6%, rgba(255, 255, 255, 0.76));
    --aguang-panel-soft: color-mix(in srgb, var(--aguang-primary) 4%, rgba(255, 255, 255, 0.60));
    --aguang-card-border: color-mix(in srgb, var(--aguang-primary) 24%, rgba(255, 255, 255, 0.46));
    --aguang-shadow: 0 20px 54px rgba(48, 43, 63, 0.12);
    --aguang-shadow-soft: 0 10px 28px rgba(48, 43, 63, 0.09);
}}

@keyframes fadeInTheme {{ from {{ opacity: 0.72; }} to {{ opacity: 1; }} }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, [data-testid="stAppViewContainer"] > .main, html[data-mood-theme], html[data-mood-theme] body, html[data-mood-theme] [data-testid="stSidebar"], html[data-mood-theme] [data-testid="stChatMessage"], html[data-mood-theme] .emotion-card, html[data-mood-theme] .aguang-mood-hint, html[data-mood-theme] [data-testid="stVerticalBlockBorderWrapper"], html[data-mood-theme] div[data-testid*="BorderWrapper"], html[data-mood-theme] [data-testid="stRadio"] [role="radiogroup"] label {{ transition: background 1.5s ease-in-out, color 0.4s ease, border-color 0.4s ease, box-shadow 0.4s ease, transform 0.2s ease; }}
html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"], section.main, [data-testid="stAppViewContainer"] > .main {{ min-height: 100vh; background: var(--aguang-bg-gradient) !important; animation: fadeInTheme 0.8s ease both; }}
.block-container {{ max-width: 1120px; padding-top: 2rem; }}
html, body, .stApp, [class*="css"] {{ color: var(--aguang-text); color-scheme: light; font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Segoe UI", sans-serif; }}
h1 {{ color: var(--aguang-text) !important; font-size: 2.2rem !important; font-weight: 800 !important; line-height: 1.18 !important; margin-bottom: 0.35rem !important; }}
h2 {{ color: var(--aguang-text) !important; font-size: 1.55rem !important; font-weight: 750 !important; line-height: 1.28 !important; margin-top: 1.1rem !important; }}
h3 {{ color: var(--aguang-text) !important; font-size: 1.18rem !important; font-weight: 700 !important; line-height: 1.35 !important; }}
p, li {{ color: var(--aguang-text) !important; font-size: 15.5px; line-height: 1.75; }}
[data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] p {{ color: var(--aguang-muted) !important; font-size: 13.5px !important; line-height: 1.6 !important; }}
[data-testid="stSidebar"] {{ background: var(--aguang-sidebar-bg) !important; border-right: 1px solid rgba(107, 91, 168, 0.16); z-index: 2147483000 !important; }}
[data-testid="stSidebarHeader"] {{ position: relative !important; z-index: 2147483001 !important; pointer-events: auto !important; }}
[data-testid="stSidebar"] * {{ color: var(--aguang-text) !important; }}
[data-testid="stSidebar"] hr {{ height: 1px; border: 0; margin: 1.25rem 0 0.75rem; background: linear-gradient(90deg, transparent, rgba(107, 91, 168, 0.36), transparent); }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {{ font-size: 14px; line-height: 1.65; }}
[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] strong {{ color: var(--aguang-text) !important; font-size: 15.5px; font-weight: 800; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: var(--aguang-muted) !important; font-size: 12.8px !important; line-height: 1.75 !important; overflow-wrap: anywhere; }}
[data-testid="stSidebar"] code {{ display: inline !important; max-width: 100%; padding: 0 2px !important; border-radius: 4px !important; background: color-mix(in srgb, var(--aguang-primary) 7%, rgba(255,255,255,.52)) !important; color: var(--aguang-muted) !important; font-size: 11.4px !important; line-height: 1.75 !important; white-space: normal !important; overflow-wrap: anywhere !important; word-break: break-word !important; vertical-align: baseline !important; box-decoration-break: clone; -webkit-box-decoration-break: clone; }}
[data-testid="stSidebarCollapseButton"], [data-testid="stSidebarCollapsedControl"] {{ position: relative !important; top: 10px !important; visibility: visible !important; opacity: 1 !important; filter: none !important; z-index: 2147483002 !important; pointer-events: auto !important; }}
[data-testid="stSidebarCollapseButton"] *, [data-testid="stSidebarCollapsedControl"] * {{ visibility: visible !important; opacity: 1 !important; }}
[data-testid="stSidebarCollapseButton"] button, [data-testid="stSidebarCollapsedControl"] button {{ position: relative !important; z-index: 2147483003 !important; width: 34px !important; height: 34px !important; min-width: 34px !important; border-radius: 999px !important; background: rgba(255,255,255,.96) !important; border: 2px solid #1F1A2A !important; color: #15111D !important; box-shadow: 0 8px 18px rgba(20,16,30,.22), 0 0 0 3px rgba(255,255,255,.48) !important; visibility: visible !important; opacity: 1 !important; pointer-events: auto !important; transition: background .18s ease, color .18s ease, transform .18s ease, box-shadow .18s ease !important; }}
[data-testid="stSidebarCollapseButton"] button span, [data-testid="stSidebarCollapseButton"] [data-testid="stIconMaterial"], [data-testid="stSidebarCollapsedControl"] button span, [data-testid="stSidebarCollapsedControl"] [data-testid="stIconMaterial"] {{ color: #15111D !important; fill: #15111D !important; visibility: visible !important; opacity: 1 !important; font-size: 27px !important; font-weight: 900 !important; line-height: 1 !important; }}
[data-testid="stSidebarCollapseButton"] button:hover, [data-testid="stSidebarCollapsedControl"] button:hover {{ background: #15111D !important; border-color: #FFFFFF !important; color: #FFFFFF !important; transform: translateY(-1px) scale(1.03); box-shadow: 0 10px 22px rgba(20,16,30,.30), 0 0 0 3px color-mix(in srgb, var(--aguang-primary) 42%, rgba(255,255,255,.55)) !important; }}
[data-testid="stSidebarCollapseButton"] button:hover span, [data-testid="stSidebarCollapseButton"] button:hover [data-testid="stIconMaterial"], [data-testid="stSidebarCollapsedControl"] button:hover span, [data-testid="stSidebarCollapsedControl"] button:hover [data-testid="stIconMaterial"] {{ color: #FFFFFF !important; fill: #FFFFFF !important; }}
.stButton > button, .stDownloadButton > button {{ min-height: 2.55rem; border-radius: 999px !important; border: 1.5px solid color-mix(in srgb, var(--aguang-primary) 58%, transparent) !important; background: rgba(255,255,255,.78) !important; color: var(--aguang-primary) !important; font-weight: 700 !important; box-shadow: 0 6px 16px rgba(107,91,168,.10) !important; transition: transform .18s ease, box-shadow .18s ease, background .18s ease, border-color .18s ease; }}
.stButton > button:hover, .stDownloadButton > button:hover {{ background: color-mix(in srgb, var(--aguang-primary) 12%, white) !important; border-color: var(--aguang-primary) !important; box-shadow: 0 10px 22px rgba(107,91,168,.16) !important; transform: translateY(-1px); }}
.stButton > button:focus, .stDownloadButton > button:focus {{ border-color: var(--aguang-primary) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--aguang-primary) 18%, transparent) !important; }}
.stButton > button[kind="primary"] {{ background: var(--aguang-primary) !important; color: #FFFFFF !important; border-color: var(--aguang-primary) !important; }}
.stButton > button[kind="primary"]:hover {{ background: var(--aguang-button-hover) !important; }}
[data-testid="stChatMessage"] {{ border-radius: 18px !important; padding: 12px 16px !important; margin-bottom: 10px !important; box-shadow: var(--aguang-shadow-soft) !important; }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {{ background: var(--aguang-user-bg) !important; border-left: 4px solid var(--aguang-user-border) !important; }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) {{ background: var(--aguang-assistant-bg) !important; border-left: 4px solid var(--aguang-assistant-border) !important; }}
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"], [data-testid="stChatMessage"] [class*="e1ypd8m72"] {{ width: 34px !important; height: 34px !important; min-width: 34px !important; border-radius: 12px !important; background: rgba(255,255,255,.82) !important; color: var(--aguang-primary) !important; border: 1px solid color-mix(in srgb, var(--aguang-primary) 22%, rgba(255,255,255,.68)) !important; box-shadow: 0 6px 16px rgba(48,43,63,.10), inset 0 1px 0 rgba(255,255,255,.72) !important; display: flex !important; align-items: center !important; justify-content: center !important; }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [data-testid^="chatAvatarIcon"], [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-assistant"]) [class*="e1ypd8m72"] {{ background: color-mix(in srgb, var(--aguang-assistant-bg) 82%, white) !important; border-color: color-mix(in srgb, var(--aguang-assistant-border) 48%, rgba(255,255,255,.62)) !important; }}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [data-testid^="chatAvatarIcon"], [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) [class*="e1ypd8m72"] {{ background: color-mix(in srgb, var(--aguang-user-bg) 82%, white) !important; border-color: color-mix(in srgb, var(--aguang-user-border) 48%, rgba(255,255,255,.62)) !important; }}
[data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] svg, [data-testid="stChatMessage"] [class*="e1ypd8m72"] svg {{ color: var(--aguang-primary) !important; fill: currentColor !important; }}
[data-testid="stChatMessage"] > div:first-child {{ width: 34px !important; height: 34px !important; min-width: 34px !important; border-radius: 12px !important; background: rgba(255,255,255,.84) !important; color: var(--aguang-primary) !important; border: 1px solid color-mix(in srgb, var(--aguang-primary) 22%, rgba(255,255,255,.68)) !important; box-shadow: 0 6px 16px rgba(48,43,63,.10), inset 0 1px 0 rgba(255,255,255,.72) !important; overflow: hidden !important; }}
[data-testid="stChatMessage"] > div:first-child *, [data-testid="stChatMessage"] [data-testid^="chatAvatarIcon"] * {{ background: transparent !important; color: var(--aguang-primary) !important; }}
[data-testid="stChatInput"], [data-testid="stChatInput"] form, [data-testid="stChatInput"] > div {{ background: transparent !important; border: none !important; box-shadow: none !important; }}
[data-testid="stChatInput"] [data-baseweb="textarea"], [data-testid="stChatInput"] [data-baseweb="base-input"], [data-testid="stChatInput"] [data-baseweb="textarea"] > div {{ background: rgba(255,255,255,.92) !important; border-radius: 22px !important; border: 1px solid color-mix(in srgb, var(--aguang-primary) 18%, rgba(255,255,255,.72)) !important; box-shadow: 0 4px 15px rgba(139,107,190,0.12) !important; overflow: hidden !important; }}
[data-testid="stChatInput"] [data-baseweb="textarea"]::before, [data-testid="stChatInput"] [data-baseweb="textarea"]::after, [data-testid="stChatInput"] [data-baseweb="base-input"]::before, [data-testid="stChatInput"] [data-baseweb="base-input"]::after {{ background: transparent !important; border: 0 !important; box-shadow: none !important; }}
[data-testid="stChatInput"] textarea {{ background: transparent !important; box-shadow: none !important; }}
[data-testid="stChatInput"] button {{ background: transparent !important; color: var(--aguang-primary) !important; border: none !important; box-shadow: none !important; }}
[data-testid="stChatInput"] textarea, .stTextInput input, .stTextArea textarea {{ color-scheme: light; border: none !important; border-radius: 20px !important; background: rgba(255,255,255,.92) !important; box-shadow: 0 4px 15px rgba(139,107,190,0.15) !important; color: var(--aguang-text) !important; caret-color: var(--aguang-primary) !important; }}
[data-testid="stChatInput"] textarea:focus, .stTextInput input:focus, .stTextArea textarea:focus {{ border: none !important; outline: none !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--aguang-primary) 16%, transparent), 0 4px 15px rgba(139,107,190,0.15) !important; }}
[data-testid="stTextInput"] div[data-baseweb="input"], .stTextInput div[data-baseweb="input"], [data-testid="stTextArea"] div[data-baseweb="textarea"], .stTextArea div[data-baseweb="textarea"], [data-testid="stChatInput"] div[data-baseweb="textarea"] {{ border: 1px solid color-mix(in srgb, var(--aguang-primary) 18%, rgba(255,255,255,.72)) !important; border-radius: 20px !important; background: rgba(255,255,255,.92) !important; box-shadow: 0 4px 15px rgba(139,107,190,0.12) !important; outline: none !important; transition: border-color .2s ease, box-shadow .2s ease, background .2s ease; }}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within, .stTextInput div[data-baseweb="input"]:focus-within, [data-testid="stTextArea"] div[data-baseweb="textarea"]:focus-within, .stTextArea div[data-baseweb="textarea"]:focus-within, [data-testid="stChatInput"] div[data-baseweb="textarea"]:focus-within {{ border-color: color-mix(in srgb, var(--aguang-primary) 42%, rgba(255,255,255,.58)) !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--aguang-primary) 15%, transparent), 0 8px 20px rgba(48,43,63,.10) !important; outline: none !important; }}
[data-testid="stTextInput"] div[data-baseweb="input"]::before, [data-testid="stTextInput"] div[data-baseweb="input"]::after, .stTextInput div[data-baseweb="input"]::before, .stTextInput div[data-baseweb="input"]::after, [data-testid="stTextArea"] div[data-baseweb="textarea"]::before, [data-testid="stTextArea"] div[data-baseweb="textarea"]::after, .stTextArea div[data-baseweb="textarea"]::before, .stTextArea div[data-baseweb="textarea"]::after, [data-testid="stChatInput"] div[data-baseweb="textarea"]::before, [data-testid="stChatInput"] div[data-baseweb="textarea"]::after {{ border-color: transparent !important; box-shadow: none !important; outline: none !important; }}
.stTextInput input::placeholder, .stTextArea textarea::placeholder, [data-testid="stChatInput"] textarea::placeholder {{ color: var(--aguang-muted) !important; opacity: .78 !important; }}
[data-baseweb="select"], [data-baseweb="select"] *, [data-testid="stSelectbox"], [data-testid="stSelectbox"] * {{ color-scheme: light; }}
[data-baseweb="select"] > div, [data-testid="stSelectbox"] [data-baseweb="select"] > div {{ background: rgba(255,255,255,.92) !important; color: var(--aguang-text) !important; border-color: color-mix(in srgb, var(--aguang-primary) 20%, rgba(255,255,255,.68)) !important; border-radius: 12px !important; box-shadow: 0 4px 15px rgba(139,107,190,0.10) !important; }}
[data-baseweb="select"] div, [data-baseweb="select"] span, [data-baseweb="select"] input, [data-baseweb="select"] svg {{ color: var(--aguang-text) !important; fill: var(--aguang-text) !important; caret-color: var(--aguang-primary) !important; }}
[data-baseweb="popover"], [data-baseweb="menu"], [role="listbox"] {{ color-scheme: light; background: rgba(255,255,255,.96) !important; color: var(--aguang-text) !important; border: 1px solid var(--aguang-card-border) !important; box-shadow: var(--aguang-shadow-soft) !important; }}
[role="option"], [role="option"] * {{ color: var(--aguang-text) !important; background: transparent !important; }}
[role="option"]:hover, [role="option"][aria-selected="true"] {{ background: color-mix(in srgb, var(--aguang-primary) 12%, white) !important; }}
[data-testid="stExpander"], [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px !important; border-color: color-mix(in srgb, var(--aguang-primary) 24%, transparent) !important; background: var(--aguang-panel) !important; box-shadow: 0 8px 22px rgba(48,43,63,.06) !important; }}
[data-testid="stMetric"] {{ padding: 14px 16px; }}
[data-testid="stMetricLabel"] p {{ color: var(--aguang-muted) !important; font-size: 13px !important; font-weight: 650 !important; }}
[data-testid="stMetricValue"] {{ color: var(--aguang-text) !important; font-weight: 800 !important; }}
@keyframes aguang-fade-in {{ from {{ opacity: 0; transform: translateY(5px); }} to {{ opacity: 1; transform: translateY(0); }} }}
@keyframes aguang-bounce {{ 0%, 80%, 100% {{ transform: translateY(0); opacity: .48; }} 40% {{ transform: translateY(-8px); opacity: 1; }} }}
.aguang-thinking {{ display: flex; align-items: center; gap: 6px; width: fit-content; padding: 14px 18px; margin-bottom: 12px; background: var(--aguang-assistant-bg); border-left: 4px solid var(--aguang-assistant-border); border-radius: 16px; box-shadow: var(--aguang-shadow-soft); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); animation: aguang-fade-in .28s ease-out both; }}
.aguang-thinking span {{ margin-right: 6px; color: var(--aguang-text); font-size: 13.5px; font-weight: 650; }}
.aguang-dot {{ width: 8px; height: 8px; border-radius: 50%; background: var(--aguang-primary); display: inline-block; animation: aguang-bounce 1.3s ease-in-out infinite; }}
.aguang-dot:nth-child(3) {{ animation-delay: .15s; }} .aguang-dot:nth-child(4) {{ animation-delay: .30s; }}
@keyframes aguang-wave {{ 0%, 100% {{ height: 5px; opacity: .42; }} 50% {{ height: 22px; opacity: 1; }} }}
.aguang-waveform {{ display: flex; align-items: center; gap: 4px; padding: 10px 18px; margin: 8px 0; background: var(--aguang-user-bg); border-left: 4px solid var(--aguang-user-border); border-radius: 12px; box-shadow: 0 6px 16px rgba(91,149,208,.10); }}
.aguang-waveform p {{ margin: 0 8px 0 0; color: var(--aguang-text) !important; font-size: 13px; font-weight: 650; }}
.aguang-bar {{ width: 4px; border-radius: 3px; background: var(--aguang-primary); animation: aguang-wave .7s ease-in-out infinite; }}
.aguang-bar:nth-child(3) {{ animation-delay: .1s; }} .aguang-bar:nth-child(4) {{ animation-delay: .2s; }} .aguang-bar:nth-child(5) {{ animation-delay: .3s; }} .aguang-bar:nth-child(6) {{ animation-delay: .4s; }} .aguang-bar:nth-child(7) {{ animation-delay: .2s; }} .aguang-bar:nth-child(8) {{ animation-delay: .1s; }}
@keyframes aguang-breathe {{ 0%, 100% {{ transform: scale(.96); opacity: .58; filter: blur(3px); }} 50% {{ transform: scale(1.34); opacity: .96; filter: blur(0); }} }}
.aguang-breathe-wrap {{ display: flex; flex-direction: column; align-items: center; margin: 22px 0 12px; }}
.aguang-breathe-circle {{ width: 68px; height: 68px; border-radius: 50%; background: var(--aguang-breathe-circle); box-shadow: 0 0 28px color-mix(in srgb, var(--aguang-primary) 28%, transparent); animation: aguang-breathe 4s ease-in-out infinite; }}
.aguang-breathe-label {{ margin-top: 12px; color: var(--aguang-primary); font-size: 13px; font-weight: 750; letter-spacing: .05em; }}
.emotion-card {{ margin-bottom: 12px; padding: 14px 16px; background: rgba(255,255,255,.72); border: 1px solid color-mix(in srgb, var(--aguang-primary) 24%, transparent); border-radius: 12px; box-shadow: var(--aguang-shadow); backdrop-filter: blur(12px) saturate(1.08); -webkit-backdrop-filter: blur(12px) saturate(1.08); }}
.emotion-card-header {{ display: flex; align-items: center; gap: 10px; color: var(--aguang-text); font-weight: 750; }}
.emotion-card p {{ margin-bottom: 0; color: var(--aguang-text) !important; font-size: 14.5px !important; line-height: 1.75 !important; }}
.emotion-badge {{ display: inline-block; padding: 3px 11px; border-radius: 999px; font-size: 12px; font-weight: 750; box-shadow: inset 0 0 0 1px rgba(255,255,255,.42); }}
.badge-開心 {{ background: #F8ECC4; color: #6D5600; }} .badge-難過 {{ background: #E6F0FB; color: #285078; }} .badge-生氣 {{ background: #F8E4DF; color: #873523; }} .badge-default {{ background: color-mix(in srgb, var(--aguang-primary) 12%, white); color: var(--aguang-primary); }}
.aguang-history-marker {{ display: none; }}
[data-testid="stExpander"]:has(.aguang-history-marker) {{ overflow: hidden; border-radius: 14px !important; }}
[data-testid="stExpander"]:has(.aguang-history-marker) summary {{ min-height: 58px; padding: 12px 14px !important; border-radius: 12px !important; align-items: flex-start !important; background: rgba(255,255,255,.48); transition: background .25s ease, border-color .25s ease, box-shadow .25s ease, transform .2s ease; }}
[data-testid="stExpander"]:has(.aguang-history-marker) summary:hover {{ transform: translateY(-1px); background: color-mix(in srgb, var(--aguang-primary) 8%, rgba(255,255,255,.68)); }}
[data-testid="stExpander"]:has(.aguang-history-marker) summary p {{ color: var(--aguang-text) !important; font-size: clamp(14px, 3.8vw, 16px) !important; font-weight: 700 !important; line-height: 1.55 !important; letter-spacing: 0 !important; white-space: normal !important; }}
[data-testid="stExpander"]:has(.aguang-history-has-note) {{ background: color-mix(in srgb, var(--aguang-primary) 9%, rgba(255,255,255,.72)) !important; border-color: color-mix(in srgb, var(--aguang-primary) 34%, rgba(255,255,255,.45)) !important; box-shadow: 0 12px 28px rgba(48,43,63,.09) !important; }}
[data-testid="stExpander"]:has(.aguang-history-has-note) summary {{ border-left: 4px solid var(--aguang-primary) !important; background: linear-gradient(90deg, color-mix(in srgb, var(--aguang-primary) 14%, rgba(255,255,255,.78)), rgba(255,255,255,.54)); }}
[data-testid="stExpander"]:has(.aguang-history-has-note) summary::after {{ content: "有補充"; flex: 0 0 auto; margin-left: auto; padding: 3px 9px; border-radius: 999px; background: color-mix(in srgb, var(--aguang-primary) 17%, rgba(255,255,255,.82)); color: var(--aguang-primary); font-size: 11px; font-weight: 800; line-height: 1.35; }}
[data-testid="stExpander"]:has(.aguang-history-no-note) summary {{ opacity: .88; }}
.aguang-history-detail {{ margin-top: 4px; }}
.aguang-history-note-text {{ margin-top: 10px; font-size: 14px; color: #3d3358; line-height: 1.75; word-break: break-word; }}
.aguang-history-empty-note {{ color: var(--aguang-muted) !important; font-style: normal; }}
.aguang-mood-hint {{ background: var(--aguang-panel); border-left: 4px solid var(--aguang-primary); border-radius: 16px; padding: 14px 18px; margin-bottom: 16px; color: var(--aguang-text); box-shadow: 0 10px 28px rgba(32,34,58,.08); }}
.aguang-welcome-card {{ display: flex; align-items: flex-start; gap: 10px; flex-direction: column; margin: 0 0 18px; padding: 18px 20px; border-radius: 18px; background: rgba(255,255,255,.78); border: 1px solid color-mix(in srgb, var(--aguang-primary) 18%, transparent); box-shadow: var(--aguang-shadow-soft); backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px); }}
.aguang-welcome-card strong {{ color: var(--aguang-text); font-size: 15.5px; font-weight: 800; }}
.aguang-welcome-card span {{ color: var(--aguang-muted); font-size: 14px; line-height: 1.65; }}

@keyframes aguang-pet-walk-track {{ 0%, 100% {{ left: 18px; }} 43%, 50% {{ left: calc(100vw - 212px); }} 93% {{ left: 18px; }} }}
@keyframes aguang-pixel-face-direction {{ 0%, 48% {{ transform: scaleX(1); }} 52%, 100% {{ transform: scaleX(-1); }} }}
@keyframes aguang-pixel-bob {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-2px); }} }}
@keyframes aguang-pixel-step {{ 0%, 100% {{ transform: translateY(0); }} 50% {{ transform: translateY(-3px); }} }}
@keyframes aguang-pixel-tail-sway {{ 0%, 100% {{ transform: translate(0, 0); }} 50% {{ transform: translate(0, -2px); }} }}
@keyframes aguang-pixel-blink {{ 0%, 88%, 100% {{ transform: scaleY(1); }} 92% {{ transform: scaleY(.42); }} }}
@keyframes aguang-pixel-cuddle {{ 0% {{ transform: translateY(0) scale(1); }} 22% {{ transform: translateX(-10px) translateY(-8px) scale(1.04); }} 48% {{ transform: translateX(8px) translateY(-3px) scale(1.02); }} 72% {{ transform: translateX(-4px) translateY(-6px) scale(1.04); }} 100% {{ transform: translateY(0) scale(1); }} }}
@keyframes aguang-heart-pop {{ 0% {{ opacity: 0; transform: translateY(8px) scale(.45) rotate(-8deg); }} 28% {{ opacity: 1; }} 100% {{ opacity: 0; transform: translateY(-44px) scale(1.16) rotate(12deg); }} }}
#aguang-desktop-pet {{ --pixel-cat-line: #9b4a13; --pixel-cat-orange: #f08a22; --pixel-cat-orange-dark: #bd5d19; --pixel-cat-orange-light: #ffb458; --pixel-cat-cream: #ffe1a3; --pixel-cat-fill: var(--pixel-cat-orange); --pixel-cat-shade: var(--pixel-cat-orange-dark); --pixel-cat-ear: #ffd0a7; --pixel-cat-cheek: #ffb684; --pixel-cat-eye: #40200b; --pixel-cat-shadow: color-mix(in srgb, var(--pixel-cat-orange-dark) 28%, rgba(40,38,46,.18)); position: fixed; left: 18px; bottom: 78px; z-index: 999999; width: 192px; min-height: 138px; padding: 0; border: 0; background: transparent; color: var(--aguang-text); cursor: pointer; text-align: center; font-family: "Noto Sans TC", "Microsoft JhengHei", "PingFang TC", "Segoe UI", sans-serif; -webkit-tap-highlight-color: transparent; animation: aguang-pet-walk-track 22s linear infinite; }}
#aguang-desktop-pet:hover, #aguang-desktop-pet.aguang-pet-speaking {{ animation-play-state: paused; }}
#aguang-desktop-pet:focus-visible {{ outline: none; }}
#aguang-desktop-pet:focus-visible .aguang-pixel-cat {{ outline: 3px solid color-mix(in srgb, var(--aguang-primary) 42%, transparent); outline-offset: 6px; }}
#aguang-desktop-pet:hover .aguang-pixel-cat, #aguang-desktop-pet.aguang-pet-speaking .aguang-pixel-cat {{ animation-play-state: paused; }}
.aguang-pixel-cat {{ position: relative; display: block; width: 168px; height: 118px; margin: 8px auto 0; image-rendering: pixelated; transform-origin: 50% 72%; animation: aguang-pixel-face-direction 22s linear infinite; }}
.aguang-pixel-sample-like {{ image-rendering: pixelated; }}
.aguang-pixel-sprite {{ position: absolute; inset: 0; animation: aguang-pixel-bob .74s steps(2, end) infinite; transform-origin: 50% 78%; }}
#aguang-desktop-pet:hover .aguang-pixel-sprite {{ filter: saturate(1.08); }}
#aguang-desktop-pet.aguang-pet-tap .aguang-pixel-sprite {{ animation: aguang-pixel-cuddle .82s steps(4, end), aguang-pixel-bob .74s steps(2, end) .82s infinite; }}
.aguang-pixel-shadow {{ position: absolute; left: 32px; bottom: 4px; width: 104px; height: 8px; background: rgba(155,74,19,.14); box-shadow: 14px 4px 0 rgba(155,74,19,.10); }}
.aguang-pixel-tail, .aguang-pixel-tail-fill {{ position: absolute; background-repeat: no-repeat; animation: aguang-pixel-tail-sway 1.2s steps(2, end) infinite; }}
.aguang-pixel-tail {{ left: 30px; top: 54px; width: 48px; height: 54px; z-index: 0; background-image: linear-gradient(var(--pixel-cat-line), var(--pixel-cat-line)), linear-gradient(var(--pixel-cat-line), var(--pixel-cat-line)), linear-gradient(var(--pixel-cat-line), var(--pixel-cat-line)), linear-gradient(var(--pixel-cat-line), var(--pixel-cat-line)), linear-gradient(var(--pixel-cat-line), var(--pixel-cat-line)); background-position: 8px 12px, 0 24px, 8px 36px, 22px 42px, 30px 32px; background-size: 16px 26px, 16px 24px, 20px 14px, 18px 12px, 12px 18px; }}
.aguang-pixel-tail-fill {{ left: 38px; top: 62px; width: 32px; height: 38px; z-index: 0; background-image: linear-gradient(var(--pixel-cat-orange), var(--pixel-cat-orange)), linear-gradient(var(--pixel-cat-orange), var(--pixel-cat-orange)), linear-gradient(var(--pixel-cat-orange), var(--pixel-cat-orange)), linear-gradient(var(--pixel-cat-orange), var(--pixel-cat-orange)); background-position: 4px 10px, 0 20px, 8px 28px, 20px 24px; background-size: 8px 16px, 10px 14px, 12px 8px, 8px 10px; }}
.aguang-pixel-tail-tip {{ position: absolute; left: 40px; top: 68px; width: 8px; height: 10px; background: var(--pixel-cat-orange-dark); z-index: 1; animation: aguang-pixel-tail-sway 1.2s steps(2, end) infinite; }}
.aguang-pixel-body {{ position: absolute; left: 56px; top: 72px; width: 74px; height: 30px; background: var(--pixel-cat-line); z-index: 1; }}
.aguang-pixel-body::after {{ content: ""; position: absolute; left: 6px; top: 6px; width: 62px; height: 18px; background: var(--pixel-cat-orange); }}
.aguang-pixel-body-shine {{ position: absolute; left: 92px; top: 82px; width: 16px; height: 12px; background: var(--pixel-cat-cream); z-index: 3; }}
.aguang-pixel-front-cream {{ position: absolute; left: 112px; top: 72px; width: 12px; height: 22px; background: var(--pixel-cat-cream); z-index: 3; }}
.aguang-pixel-head {{ position: absolute; left: 96px; top: 36px; width: 50px; height: 48px; background: var(--pixel-cat-line); z-index: 4; }}
.aguang-pixel-head::after {{ content: ""; position: absolute; left: 6px; top: 6px; width: 38px; height: 36px; background: var(--pixel-cat-orange); }}
.aguang-pixel-ear {{ position: absolute; top: 22px; width: 20px; height: 22px; background: var(--pixel-cat-line); z-index: 5; clip-path: polygon(50% 0, 100% 100%, 0 100%); }}
.aguang-pixel-ear-left {{ left: 98px; }}
.aguang-pixel-ear-right {{ left: 124px; }}
.aguang-pixel-ear-inner {{ position: absolute; top: 31px; width: 10px; height: 10px; background: var(--pixel-cat-ear); z-index: 6; clip-path: polygon(50% 0, 100% 100%, 0 100%); }}
.aguang-pixel-ear-inner-left {{ left: 103px; }}
.aguang-pixel-ear-inner-right {{ left: 129px; }}
.aguang-pixel-eye {{ position: absolute; top: 55px; width: 6px; height: 10px; background: var(--pixel-cat-eye); z-index: 7; transform-origin: 50% 100%; animation: aguang-pixel-blink 5.8s steps(1, end) infinite; }}
.aguang-pixel-eye::before {{ display: none; }}
.aguang-pixel-eye::after {{ content: ""; position: absolute; left: 1px; top: 1px; width: 2px; height: 2px; background: rgba(255,255,255,.88); }}
.aguang-pixel-eye-left {{ left: 113px; }}
.aguang-pixel-eye-right {{ left: 132px; }}
.aguang-pixel-nose {{ position: absolute; left: 126px; top: 66px; width: 6px; height: 5px; background: var(--pixel-cat-line); z-index: 8; }}
.aguang-pixel-mouth {{ position: absolute; left: 119px; top: 72px; width: 5px; height: 4px; background: var(--pixel-cat-line); box-shadow: 8px 0 0 var(--pixel-cat-line); z-index: 8; }}
.aguang-pixel-cheek {{ position: absolute; top: 67px; width: 8px; height: 5px; background: var(--pixel-cat-cheek); z-index: 7; opacity: .7; }}
.aguang-pixel-cheek-left {{ left: 108px; }}
.aguang-pixel-cheek-right {{ left: 136px; }}
.aguang-pixel-cream {{ position: absolute; left: 115px; top: 62px; width: 23px; height: 15px; background: var(--pixel-cat-cream); z-index: 6; }}
.aguang-pixel-stripe {{ position: absolute; background: var(--pixel-cat-orange-dark); z-index: 6; }}
.aguang-pixel-stripe-head-a {{ left: 108px; top: 43px; width: 6px; height: 10px; }}
.aguang-pixel-stripe-head-b {{ left: 119px; top: 41px; width: 6px; height: 11px; }}
.aguang-pixel-stripe-head-c {{ left: 130px; top: 43px; width: 6px; height: 10px; }}
.aguang-pixel-stripe-body-a {{ left: 66px; top: 74px; width: 7px; height: 14px; }}
.aguang-pixel-stripe-body-b {{ left: 80px; top: 74px; width: 7px; height: 12px; }}
.aguang-pixel-stripe-tail-a {{ left: 50px; top: 64px; width: 7px; height: 11px; }}
.aguang-pixel-stripe-tail-b {{ left: 42px; top: 78px; width: 7px; height: 11px; }}
.aguang-pixel-whisker {{ position: absolute; height: 4px; background: var(--pixel-cat-line); z-index: 6; }}
.aguang-pixel-whisker-left-a {{ display: none; }}
.aguang-pixel-whisker-left-b {{ display: none; }}
.aguang-pixel-whisker-right-a {{ left: 142px; top: 60px; width: 16px; }}
.aguang-pixel-whisker-right-b {{ left: 142px; top: 70px; width: 16px; }}
.aguang-pixel-leg {{ position: absolute; top: 98px; width: 10px; height: 12px; background: var(--pixel-cat-line); animation: aguang-pixel-step .6s steps(2, end) infinite; z-index: 3; }}
.aguang-pixel-leg::after {{ content: ""; position: absolute; left: 3px; top: 0; width: 4px; height: 7px; background: var(--pixel-cat-orange); }}
.aguang-pixel-paw {{ position: absolute; top: 106px; width: 14px; height: 7px; background: var(--pixel-cat-line); animation: aguang-pixel-step .6s steps(2, end) infinite; z-index: 4; }}
.aguang-pixel-paw::after {{ content: ""; position: absolute; left: 3px; top: 0; width: 8px; height: 3px; background: var(--pixel-cat-orange); }}
.aguang-pixel-leg-front-left {{ left: 62px; animation-delay: 0s; }}
.aguang-pixel-leg-front-right {{ left: 78px; animation-delay: .3s; }}
.aguang-pixel-leg-back-left {{ left: 104px; animation-delay: .3s; }}
.aguang-pixel-leg-back-right {{ left: 120px; animation-delay: 0s; }}
.aguang-pixel-paw-front-left {{ left: 58px; animation-delay: 0s; }}
.aguang-pixel-paw-front-right {{ left: 74px; animation-delay: .3s; }}
.aguang-pixel-paw-back-left {{ left: 100px; animation-delay: .3s; }}
.aguang-pixel-paw-back-right {{ left: 116px; animation-delay: 0s; }}
.aguang-cat-heart {{ position: absolute; left: 44px; top: 6px; color: #E985A6; font-size: 17px; line-height: 1; opacity: 0; text-shadow: 3px 3px 0 rgba(17,17,17,.12); z-index: 4; }}
.aguang-cat-heart-two {{ left: 70px; top: 0; font-size: 13px; animation-delay: .1s; }}
#aguang-desktop-pet.aguang-pet-tap .aguang-cat-heart {{ animation: aguang-heart-pop .9s steps(6, end) both; }}
#aguang-desktop-pet[data-pet-mood="neutral"] {{ --pixel-cat-eye: #3a1d0b; }}
#aguang-desktop-pet[data-pet-mood="calm_positive"] {{ --pixel-cat-eye: #5c2d0c; --pixel-cat-orange-light: #ffc46f; }}
#aguang-desktop-pet[data-pet-mood="grounding_stress"] {{ --pixel-cat-eye: #315f6f; --pixel-cat-cheek: #ffb186; }}
#aguang-desktop-pet[data-pet-mood="quiet_low"] {{ --pixel-cat-eye: #4f5974; --pixel-cat-orange: #e78934; --pixel-cat-orange-dark: #b75b1e; }}
#aguang-desktop-pet[data-pet-mood="quiet_low"] .aguang-pixel-eye {{ height: 8px; top: 54px; opacity: .92; }}
#aguang-desktop-pet[data-pet-mood="quiet_low"] .aguang-pixel-eye::before {{ opacity: .72; }}
#aguang-desktop-pet[data-pet-mood="calm_positive"] .aguang-pixel-tail, #aguang-desktop-pet[data-pet-mood="calm_positive"] .aguang-pixel-tail-fill {{ animation-duration: .9s; }}
.aguang-pet-speech {{ position: absolute; left: 0; bottom: 118px; width: min(214px, calc(100vw - 32px)); padding: 10px 12px; border-radius: 6px 6px 6px 0; background: rgba(255,255,255,.94); border: 3px solid var(--pixel-cat-line); box-shadow: 6px 6px 0 color-mix(in srgb, var(--aguang-primary) 18%, rgba(17,17,17,.12)); color: var(--aguang-text); font-size: 13px; font-weight: 750; line-height: 1.5; opacity: 0; transform: translateY(8px) scale(.96); transform-origin: left bottom; transition: opacity .2s ease, transform .2s ease; pointer-events: none; overflow-wrap: anywhere; }}
#aguang-desktop-pet.aguang-pet-speaking .aguang-pet-speech {{ opacity: 1; transform: translateY(0) scale(1); }}
@media (max-width: 640px) {{
    @keyframes aguang-pet-walk-track {{ 0%, 100% {{ left: 8px; }} 43%, 50% {{ left: calc(100vw - 168px); }} 93% {{ left: 8px; }} }}
    #aguang-desktop-pet {{ bottom: 82px; width: 158px; min-height: 120px; transform: scale(.78); transform-origin: left bottom; }}
    .aguang-pixel-cat {{ margin-left: 0; }}
    .aguang-pet-speech {{ bottom: 112px; width: min(188px, calc(100vw - 22px)); font-size: 12px; line-height: 1.45; padding: 8px 10px; }}
}}

@keyframes aguang-sprite-pet-walk-track {{ 0%, 100% {{ left: 18px; }} 43%, 50% {{ left: calc(100vw - 156px); }} 93% {{ left: 18px; }} }}
@keyframes aguang-sprite-face-direction {{ 0%, 48% {{ transform: scaleX(1); }} 52%, 100% {{ transform: scaleX(-1); }} }}
@keyframes aguang-sprite-bob {{ 0%, 100% {{ transform: translateX(-50%) translateY(0); }} 50% {{ transform: translateX(-50%) translateY(-3px); }} }}
@keyframes aguang-sprite-cuddle {{ 0% {{ transform: translateX(-50%) translateY(0) scale(1); }} 24% {{ transform: translateX(-50%) translateY(-9px) scale(1.06); }} 48% {{ transform: translateX(-48%) translateY(-2px) scale(1.03); }} 72% {{ transform: translateX(-52%) translateY(-6px) scale(1.05); }} 100% {{ transform: translateX(-50%) translateY(0) scale(1); }} }}
@keyframes aguang-sprite-hop {{ 0%, 100% {{ transform: translateX(-50%) translateY(0); }} 50% {{ transform: translateX(-50%) translateY(-5px); }} }}
@keyframes aguang-sprite-walk-step {{ 0%, 100% {{ transform: translateY(0) scaleX(1) scaleY(1); }} 50% {{ transform: translateY(2px) scaleX(1.03) scaleY(.97); }} }}
@keyframes aguang-sprite-idle-ear {{ 0%, 100% {{ transform: rotate(0deg) translate(0, 0); opacity: .78; }} 50% {{ transform: rotate(-7deg) translate(-2px, -1px); opacity: .92; }} }}
@keyframes aguang-sprite-spark {{ 0% {{ opacity: 0; transform: translate(6px, 8px) scale(.3); }} 35% {{ opacity: 1; transform: translate(0, 0) scale(1); }} 100% {{ opacity: 0; transform: translate(-8px, -18px) scale(.55); }} }}
@keyframes aguang-sprite-frame-step {{ 0%, 24.99% {{ background-position: 0 0; }} 25%, 49.99% {{ background-position: calc(var(--aguang-sprite-width) * -1) 0; }} 50%, 74.99% {{ background-position: calc(var(--aguang-sprite-width) * -2) 0; }} 75%, 100% {{ background-position: calc(var(--aguang-sprite-width) * -3) 0; }} }}
@keyframes aguang-sprite-life-step {{ from {{ background-position: 0 0; }} to {{ background-position: calc(var(--aguang-sprite-width) * -12) 0; }} }}
#aguang-desktop-pet {{ --aguang-sprite-width: 96px; --aguang-sprite-height: 94px; --aguang-sprite-shadow: color-mix(in srgb, var(--aguang-primary) 18%, rgba(38,29,20,.20)); width: 112px; min-height: 106px; bottom: 82px; right: auto; animation: none; will-change: left; }}
#aguang-desktop-pet:hover, #aguang-desktop-pet.aguang-pet-speaking {{ animation-play-state: paused; }}
#aguang-desktop-pet:focus-visible .aguang-sprite-life-sheet, #aguang-desktop-pet:focus-visible .aguang-sprite-sheet, #aguang-desktop-pet:focus-visible .aguang-sprite-cat-image {{ outline: 3px solid color-mix(in srgb, var(--aguang-primary) 42%, transparent); outline-offset: 4px; }}
.aguang-sprite-cat {{ position: relative; display: block; width: var(--aguang-sprite-width); height: var(--aguang-sprite-height); margin: 0 auto; image-rendering: pixelated; transform-origin: 50% 72%; transform: scaleX(1); }}
#aguang-desktop-pet[data-pet-direction="right"] .aguang-sprite-cat {{ transform: scaleX(1); }}
#aguang-desktop-pet[data-pet-direction="left"] .aguang-sprite-cat {{ transform: scaleX(-1); }}
#aguang-desktop-pet:hover .aguang-sprite-cat, #aguang-desktop-pet.aguang-pet-speaking .aguang-sprite-cat {{ animation-play-state: paused; }}
.aguang-sprite-cat-core {{ position: absolute; left: 50%; bottom: 8px; width: var(--aguang-sprite-width); height: var(--aguang-sprite-height); transform: translateX(-50%); transform-origin: 50% 78%; animation: aguang-sprite-hop .72s steps(2, end) infinite; }}
#aguang-desktop-pet.aguang-pet-tap .aguang-sprite-cat-core {{ animation: aguang-sprite-cuddle .82s steps(4, end), aguang-sprite-hop .72s steps(2, end) .82s infinite; }}
.aguang-sprite-life-sheet, .aguang-sprite-sheet {{ position: relative; z-index: 1; display: block; width: 100%; height: var(--aguang-sprite-height); background-repeat: no-repeat; user-select: none; pointer-events: none; image-rendering: pixelated; image-rendering: crisp-edges; filter: drop-shadow(0 10px 5px var(--aguang-sprite-shadow)); transform-origin: 50% 88%; }}
.aguang-sprite-life-sheet {{ background-size: calc(var(--aguang-sprite-width) * 12) var(--aguang-sprite-height); animation: aguang-sprite-life-step 2.4s steps(12, end) infinite; }}
.aguang-sprite-sheet {{ background-size: calc(var(--aguang-sprite-width) * 4) var(--aguang-sprite-height); animation: aguang-sprite-frame-step .64s steps(1, end) infinite; }}
#aguang-desktop-pet:hover .aguang-sprite-life-sheet, #aguang-desktop-pet.aguang-pet-speaking .aguang-sprite-life-sheet, #aguang-desktop-pet:hover .aguang-sprite-sheet, #aguang-desktop-pet.aguang-pet-speaking .aguang-sprite-sheet {{ animation-play-state: running; }}
.aguang-sprite-cat-image {{ position: relative; z-index: 1; display: block; width: 100%; height: auto; user-select: none; pointer-events: none; image-rendering: pixelated; image-rendering: crisp-edges; filter: drop-shadow(0 10px 5px var(--aguang-sprite-shadow)); transform-origin: 50% 88%; animation: aguang-sprite-walk-step .72s steps(2, end) infinite; }}
.aguang-sprite-cat-tail {{ position: absolute; z-index: 2; left: 0; top: 0; display: block; width: 100%; height: auto; user-select: none; pointer-events: none; image-rendering: pixelated; image-rendering: crisp-edges; clip-path: inset(30% 54% 18% 0); transform-origin: 40% 66%; animation: aguang-sprite-idle-ear 1.08s steps(2, end) infinite; }}
.aguang-sprite-spark {{ position: absolute; z-index: 4; right: 11px; top: 12px; width: 7px; height: 7px; background: #FFE5A7; opacity: 0; pointer-events: none; box-shadow: 0 -7px 0 -2px #FFE5A7, 0 7px 0 -2px #FFE5A7, -7px 0 0 -2px #FFE5A7, 7px 0 0 -2px #FFE5A7; }}
#aguang-desktop-pet.aguang-pet-tap .aguang-sprite-spark {{ animation: aguang-sprite-spark .75s steps(5, end) both; }}
.aguang-sprite-fallback {{ position: absolute; left: 50%; bottom: 26px; transform: translateX(-50%); width: 78px; height: 48px; background: #F08A22; border: 4px solid #9B4A13; color: #40200B; font-size: 12px; font-weight: 800; display: none; align-items: center; justify-content: center; }}
#aguang-desktop-pet[data-sprite-missing="true"] .aguang-sprite-cat-image, #aguang-desktop-pet[data-sprite-missing="true"] .aguang-sprite-cat-tail {{ display: none; }}
#aguang-desktop-pet[data-life-missing="false"] .aguang-sprite-sheet, #aguang-desktop-pet[data-life-missing="false"] .aguang-sprite-cat-image, #aguang-desktop-pet[data-life-missing="false"] .aguang-sprite-cat-tail {{ display: none; }}
#aguang-desktop-pet[data-life-missing="true"] .aguang-sprite-life-sheet {{ display: none; }}
#aguang-desktop-pet[data-sheet-missing="false"] .aguang-sprite-cat-image, #aguang-desktop-pet[data-sheet-missing="false"] .aguang-sprite-cat-tail {{ display: none; }}
#aguang-desktop-pet[data-sheet-missing="true"] .aguang-sprite-sheet {{ display: none; }}
#aguang-desktop-pet[data-sheet-missing="false"] .aguang-sprite-fallback {{ display: none; }}
#aguang-desktop-pet[data-sprite-missing="true"] .aguang-sprite-fallback {{ display: flex; }}
#aguang-desktop-pet[data-sheet-missing="false"] .aguang-sprite-fallback {{ display: none; }}
#aguang-desktop-pet.aguang-pet-tap .aguang-cat-heart {{ animation: aguang-heart-pop .9s steps(6, end) both; }}
.aguang-sprite-cat .aguang-cat-heart {{ left: auto; right: 12px; top: 8px; z-index: 3; }}
.aguang-sprite-cat .aguang-cat-heart-two {{ right: 34px; top: 0; }}
.aguang-pet-speech {{ bottom: 104px; width: min(214px, calc(100vw - 32px)); border-color: #9B4A13; box-shadow: 6px 6px 0 color-mix(in srgb, var(--aguang-primary) 16%, rgba(155,74,19,.18)); }}
@media (max-width: 640px) {{
    @keyframes aguang-sprite-pet-walk-track {{ 0%, 100% {{ left: 8px; }} 43%, 50% {{ left: calc(100vw - 124px); }} 93% {{ left: 8px; }} }}
    #aguang-desktop-pet {{ --aguang-sprite-width: 96px; --aguang-sprite-height: 94px; bottom: 84px; width: 112px; min-height: 104px; transform: none; animation: none; }}
    .aguang-pet-speech {{ bottom: 104px; width: min(188px, calc(100vw - 22px)); font-size: 12px; line-height: 1.45; padding: 8px 10px; }}
}}

/* 沉浸柔霧背景：全頁覆蓋 + CSS-only 紋理 */
.stApp {{ position: relative; isolation: isolate; overflow-x: hidden; }}
.stApp::before {{ content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; background: linear-gradient(135deg, color-mix(in srgb, var(--aguang-primary) 28%, transparent) 0%, transparent 34%), linear-gradient(225deg, color-mix(in srgb, var(--aguang-assistant-border) 30%, transparent) 0%, transparent 38%), linear-gradient(0deg, color-mix(in srgb, var(--aguang-user-border) 18%, transparent) 0%, transparent 34%), linear-gradient(135deg, rgba(255,255,255,.50), rgba(255,255,255,0) 48%), repeating-linear-gradient(112deg, rgba(255,255,255,.20) 0 1px, transparent 1px 26px), repeating-linear-gradient(22deg, rgba(255,255,255,.11) 0 1px, transparent 1px 34px); opacity: .96; mix-blend-mode: multiply; }}
.stApp::after {{ content: ""; position: fixed; inset: 0; z-index: 0; pointer-events: none; background-image: linear-gradient(90deg, rgba(255,255,255,.22), transparent 24%, transparent 76%, rgba(255,255,255,.20)), var(--aguang-decor-1), var(--aguang-decor-2), var(--aguang-decor-3), var(--aguang-decor-1), var(--aguang-decor-2), var(--aguang-decor-3), var(--aguang-decor-1); background-repeat: no-repeat; background-position: center, right 5vw top 10vh, right 15vw top 24vh, right 7vw bottom 22vh, left 6vw top 16vh, left 13vw bottom 20vh, left 5vw bottom 8vh, right 28vw top 6vh; background-size: auto, clamp(56px, 6.6vw, 102px), clamp(38px, 4.8vw, 76px), clamp(48px, 5.6vw, 88px), clamp(46px, 5.4vw, 84px), clamp(58px, 6.4vw, 102px), clamp(36px, 4.6vw, 72px), clamp(30px, 3.8vw, 60px); opacity: calc(var(--aguang-decor-opacity) + .08); filter: var(--aguang-decor-filter); transform: translateZ(0); -webkit-mask-image: linear-gradient(90deg, rgba(0,0,0,.82), rgba(0,0,0,.28) 34%, rgba(0,0,0,.22) 66%, rgba(0,0,0,.82)); mask-image: linear-gradient(90deg, rgba(0,0,0,.82), rgba(0,0,0,.28) 34%, rgba(0,0,0,.22) 66%, rgba(0,0,0,.82)); }}
.stApp > header, .stApp [data-testid="stAppViewContainer"], .stApp [data-testid="stSidebar"] {{ position: relative; z-index: 1; }}
.aguang-decor-credit {{ position: fixed; left: 12px; bottom: 7px; z-index: 3; pointer-events: none; padding: 2px 6px; border-radius: 999px; background: rgba(255,255,255,.38); color: var(--aguang-muted); font-size: 10px; line-height: 1.4; opacity: .62; backdrop-filter: blur(8px); -webkit-backdrop-filter: blur(8px); }}
[data-testid="stAppViewContainer"], [data-testid="stMain"], [data-testid="stMain"] > div, section.main, section.main > div, [data-testid="stAppViewContainer"] > .main {{ background: transparent !important; }}
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {{ background: transparent !important; box-shadow: none !important; }}
[data-testid="stToolbar"] {{ color: var(--aguang-text) !important; }}
.block-container {{ position: relative; z-index: 1; }}

/* Streamlit bordered containers -> mood-aware glass cards */
[data-testid="stVerticalBlockBorderWrapper"], div[data-testid*="BorderWrapper"] {{ position: relative; overflow: hidden; border-radius: 8px !important; border: 1px solid var(--aguang-card-border) !important; background: linear-gradient(180deg, rgba(255,255,255,.74), rgba(255,255,255,.56)), var(--aguang-panel-soft) !important; box-shadow: var(--aguang-shadow), inset 0 1px 0 rgba(255,255,255,.68) !important; backdrop-filter: blur(18px) saturate(1.12); -webkit-backdrop-filter: blur(18px) saturate(1.12); }}
[data-testid="stVerticalBlockBorderWrapper"]::before, div[data-testid*="BorderWrapper"]::before {{ content: ""; position: absolute; inset: 0 0 auto 0; height: 3px; background: linear-gradient(90deg, var(--aguang-primary), color-mix(in srgb, var(--aguang-assistant-border) 70%, white), transparent); opacity: .72; pointer-events: none; }}
[data-testid="stVerticalBlockBorderWrapper"] [data-testid="stVerticalBlock"], div[data-testid*="BorderWrapper"] [data-testid="stVerticalBlock"] {{ position: relative; z-index: 1; }}

/* 問卷 radio：柔色 row buttons + selected fallback class */
[data-testid="stRadio"] [role="radiogroup"] {{ display: flex; flex-direction: column; gap: 9px; margin-top: .35rem; }}
[data-testid="stRadio"] [role="radiogroup"] label {{ position: relative; width: 100%; min-height: 40px; align-items: center; padding: 10px 13px 10px 14px !important; border-radius: 10px !important; border: 1px solid color-mix(in srgb, var(--aguang-primary) 14%, rgba(255,255,255,.62)) !important; border-left: 3px solid transparent !important; background: rgba(255,255,255,.48) !important; box-shadow: inset 0 1px 0 rgba(255,255,255,.52); cursor: pointer; }}
[data-testid="stRadio"] [role="radiogroup"] label:hover {{ transform: translateY(-1px); background: color-mix(in srgb, var(--aguang-primary) 9%, rgba(255,255,255,.70)) !important; border-color: color-mix(in srgb, var(--aguang-primary) 32%, rgba(255,255,255,.58)) !important; box-shadow: 0 8px 18px rgba(48,43,63,.08), inset 0 1px 0 rgba(255,255,255,.58); }}
[data-testid="stRadio"] [role="radiogroup"] label:focus-within {{ outline: none !important; box-shadow: 0 0 0 3px color-mix(in srgb, var(--aguang-primary) 16%, transparent), 0 8px 18px rgba(48,43,63,.08) !important; }}
[data-testid="stRadio"] [role="radiogroup"] label:has(input:checked), [data-testid="stRadio"] [role="radiogroup"] label.aguang-radio-selected {{ background: color-mix(in srgb, var(--aguang-primary) 14%, rgba(255,255,255,.76)) !important; border-color: color-mix(in srgb, var(--aguang-primary) 42%, rgba(255,255,255,.42)) !important; border-left-color: var(--aguang-primary) !important; box-shadow: 0 10px 24px rgba(48,43,63,.10), inset 0 1px 0 rgba(255,255,255,.70) !important; }}
[data-testid="stRadio"] input[type="radio"] {{ accent-color: var(--aguang-primary); }}
[data-testid="stRadio"] [role="radiogroup"] label p, [data-testid="stRadio"] [role="radiogroup"] label span {{ color: var(--aguang-text) !important; font-size: 14.5px !important; line-height: 1.55 !important; }}
[data-testid="stRadio"] > label, [data-testid="stRadio"] [data-testid="stMarkdownContainer"] p:first-child {{ color: var(--aguang-text) !important; font-weight: 700 !important; }}
html[data-mood-theme="quiet_low"] [data-testid="stRadio"] [role="radiogroup"] label, html[data-mood-theme="quiet_low"] [data-testid="stRadio"] [role="radiogroup"] label p, html[data-mood-theme="quiet_low"] [data-testid="stRadio"] [role="radiogroup"] label span, html[data-mood-theme="quiet_low"] [data-testid="stVerticalBlockBorderWrapper"] *, html[data-mood-theme="quiet_low"] div[data-testid*="BorderWrapper"] * {{ color: #555 !important; }}
html[data-mood-theme="quiet_low"] p, html[data-mood-theme="quiet_low"] li, html[data-mood-theme="quiet_low"] [data-testid="stCaptionContainer"] p, html[data-mood-theme="quiet_low"] .emotion-card p {{ color: #555 !important; }}
@media (prefers-color-scheme: dark) {{
    html, body, .stApp, [data-testid="stSidebar"], [data-testid="stAppViewContainer"], [data-testid="stMain"] {{ color-scheme: light; }}
    [data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea, [data-testid="stSidebar"] [data-baseweb="select"] > div {{ background: rgba(255,255,255,.94) !important; color: var(--aguang-text) !important; }}
    [data-testid="stSidebar"] [data-baseweb="select"] div, [data-testid="stSidebar"] [data-baseweb="select"] span, [data-testid="stSidebar"] [data-baseweb="select"] svg {{ color: var(--aguang-text) !important; fill: var(--aguang-text) !important; }}
    [data-testid="stCaptionContainer"] p, [data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {{ color: var(--aguang-muted) !important; }}
}}
</style>
<div class="aguang-decor-credit">背景裝飾圖示：Twemoji / CC BY 4.0</div>
""", unsafe_allow_html=True)

    components.html(f"""
<script>
(() => {{
  const doc = window.parent.document;
  const root = doc.documentElement;
  const mood = {json.dumps(mood)};
  const decorEmotion = {json.dumps(decor.get("source_label", mood), ensure_ascii=False)};
  const syncRadioState = () => {{
    doc.querySelectorAll('[data-testid="stRadio"]').forEach((radio) => {{
      radio.querySelectorAll('label.aguang-radio-selected').forEach((label) => label.classList.remove('aguang-radio-selected'));
      radio.querySelectorAll('input[type="radio"]:checked').forEach((input) => {{
        const label = input.closest('label');
        if (label) label.classList.add('aguang-radio-selected');
      }});
    }});
  }};
  setTimeout(() => {{
    root.setAttribute("data-mood-theme", mood);
    root.setAttribute("data-mood-decor", decorEmotion);
    root.style.animation = "fadeInTheme 0.8s ease";
    syncRadioState();
  }}, 50);
  doc.addEventListener("click", (event) => {{
    if (!event.target || !event.target.closest || event.target.matches('[data-testid="stRadio"] input[type="radio"]')) return;
    const label = event.target.closest('[data-testid="stRadio"] [role="radiogroup"] label');
    if (!label) return;
    const input = label.querySelector('input[type="radio"]');
    if (!input || input.checked) return;
    input.click();
    setTimeout(syncRadioState, 0);
  }}, true);
  doc.addEventListener("change", (event) => {{
    if (event.target && event.target.matches && event.target.matches('[data-testid="stRadio"] input[type="radio"]')) {{
      setTimeout(syncRadioState, 0);
    }}
  }}, true);
}})();
</script>
""", height=0)


def render_aguang_pet(mood):
    """Render a lightweight front-end-only companion pet without touching app state."""
    mood = mood if mood in PET_MOOD_STATES else "neutral"
    pet_state = {
        "mood": mood,
        "label": PET_MOOD_STATES[mood]["label"],
        "messages": PET_MOOD_STATES[mood]["messages"],
        "sprite_uri": get_pet_sprite_data_uri(),
        "walk_sheet_uri": get_pet_walk_sheet_data_uri(),
        "life_sheet_uri": get_pet_life_sheet_data_uri(),
    }

    components.html(f"""
<script>
(() => {{
  const doc = window.parent.document;
  const payload = {json.dumps(pet_state, ensure_ascii=False)};
  let pet = doc.getElementById("aguang-desktop-pet");
  const oldMood = pet ? pet.dataset.petMood : "";
  const carryWalkX = pet ? pet.dataset.walkX : "";
  const carryWalkDir = pet ? pet.dataset.walkDir : "";

  if (pet && pet.dataset.petVersion && pet.dataset.petVersion !== "sprite-cat-v7") {{
    if (pet._aguangWalkFrame) {{
      window.cancelAnimationFrame(pet._aguangWalkFrame);
    }}
    pet.remove();
    pet = null;
  }}

  if (!pet) {{
    pet = doc.createElement("button");
    pet.id = "aguang-desktop-pet";
    pet.type = "button";
    doc.body.appendChild(pet);
  }}

  if (pet.dataset.petKind !== "cat" || (pet.dataset.petVersion !== "pixel-cat-v8" && pet.dataset.petVersion !== "sprite-cat-v7")) {{
    pet.innerHTML = `
      <span class="aguang-pet-speech" aria-hidden="true"></span>
      <span class="aguang-pixel-cat aguang-pixel-sample-like" aria-hidden="true">
        <span class="aguang-pixel-sprite">
          <span class="aguang-pixel-shadow"></span>
          <span class="aguang-pixel-tail"></span>
          <span class="aguang-pixel-tail-fill"></span>
          <span class="aguang-pixel-tail-tip"></span>
          <span class="aguang-pixel-body"></span>
          <span class="aguang-pixel-body-shine"></span>
          <span class="aguang-pixel-front-cream"></span>
          <span class="aguang-pixel-head"></span>
          <span class="aguang-pixel-ear aguang-pixel-ear-left"></span>
          <span class="aguang-pixel-ear aguang-pixel-ear-right"></span>
          <span class="aguang-pixel-ear-inner aguang-pixel-ear-inner-left"></span>
          <span class="aguang-pixel-ear-inner aguang-pixel-ear-inner-right"></span>
          <span class="aguang-pixel-eye aguang-pixel-eye-left"></span>
          <span class="aguang-pixel-eye aguang-pixel-eye-right"></span>
          <span class="aguang-pixel-cream"></span>
          <span class="aguang-pixel-nose"></span>
          <span class="aguang-pixel-mouth"></span>
          <span class="aguang-pixel-cheek aguang-pixel-cheek-left"></span>
          <span class="aguang-pixel-cheek aguang-pixel-cheek-right"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-head-a"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-head-b"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-head-c"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-body-a"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-body-b"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-tail-a"></span>
          <span class="aguang-pixel-stripe aguang-pixel-stripe-tail-b"></span>
          <span class="aguang-pixel-whisker aguang-pixel-whisker-left-a"></span>
          <span class="aguang-pixel-whisker aguang-pixel-whisker-left-b"></span>
          <span class="aguang-pixel-whisker aguang-pixel-whisker-right-a"></span>
          <span class="aguang-pixel-whisker aguang-pixel-whisker-right-b"></span>
          <span class="aguang-pixel-leg aguang-pixel-leg-front-left"></span>
          <span class="aguang-pixel-leg aguang-pixel-leg-front-right"></span>
          <span class="aguang-pixel-leg aguang-pixel-leg-back-left"></span>
          <span class="aguang-pixel-leg aguang-pixel-leg-back-right"></span>
          <span class="aguang-pixel-paw aguang-pixel-paw-front-left"></span>
          <span class="aguang-pixel-paw aguang-pixel-paw-front-right"></span>
          <span class="aguang-pixel-paw aguang-pixel-paw-back-left"></span>
          <span class="aguang-pixel-paw aguang-pixel-paw-back-right"></span>
          <span class="aguang-cat-heart aguang-cat-heart-one">&hearts;</span>
          <span class="aguang-cat-heart aguang-cat-heart-two">&hearts;</span>
        </span>
      </span>
    `;
    pet.dataset.petKind = "cat";
    pet.dataset.petVersion = "pixel-cat-v8";
  }}

  if (pet.dataset.petVersion !== "sprite-cat-v7") {{
    pet.innerHTML = `
      <span class="aguang-pet-speech" aria-hidden="true"></span>
      <span class="aguang-sprite-cat" aria-hidden="true">
        <span class="aguang-sprite-cat-core">
          <span class="aguang-sprite-life-sheet"></span>
          <span class="aguang-sprite-sheet"></span>
          <img class="aguang-sprite-cat-image" alt="" draggable="false" />
          <img class="aguang-sprite-cat-tail" alt="" draggable="false" />
          <span class="aguang-sprite-spark"></span>
          <span class="aguang-sprite-fallback">cat</span>
          <span class="aguang-cat-heart aguang-cat-heart-one">&hearts;</span>
          <span class="aguang-cat-heart aguang-cat-heart-two">&hearts;</span>
        </span>
      </span>
    `;
    pet.dataset.petKind = "cat";
    pet.dataset.petVersion = "sprite-cat-v7";
  }}

  pet.dataset.petMood = payload.mood;
  pet.dataset.spriteMissing = payload.sprite_uri ? "false" : "true";
  pet.dataset.sheetMissing = payload.walk_sheet_uri ? "false" : "true";
  pet.dataset.lifeMissing = payload.life_sheet_uri ? "false" : "true";
  pet.dataset.petDirection = pet.dataset.petDirection || "right";
  pet.dataset.petMessages = JSON.stringify(payload.messages || []);
  pet.dataset.petIndex = oldMood === payload.mood ? (pet.dataset.petIndex || "-1") : "-1";
  pet.setAttribute("aria-label", "阿光貓貓，點擊讓她撒嬌");
  pet.setAttribute("title", payload.label || "阿光貓貓");

  if (carryWalkX && !pet.dataset.walkX) pet.dataset.walkX = carryWalkX;
  if (carryWalkDir && !pet.dataset.walkDir) pet.dataset.walkDir = carryWalkDir;

  const lifeSheet = pet.querySelector(".aguang-sprite-life-sheet");
  if (lifeSheet) {{
    lifeSheet.style.backgroundImage = payload.life_sheet_uri ? `url("${{payload.life_sheet_uri}}")` : "";
  }}

  const walkSheet = pet.querySelector(".aguang-sprite-sheet");
  if (walkSheet) {{
    walkSheet.style.backgroundImage = payload.walk_sheet_uri ? `url("${{payload.walk_sheet_uri}}")` : "";
  }}

  pet.querySelectorAll(".aguang-sprite-cat-image, .aguang-sprite-cat-tail").forEach((spriteImage) => {{
    if (payload.sprite_uri && spriteImage.src !== payload.sprite_uri) {{
      spriteImage.src = payload.sprite_uri;
    }}
  }});

  if (pet._aguangWalkFrame) {{
    window.cancelAnimationFrame(pet._aguangWalkFrame);
  }}
  const currentRect = pet.getBoundingClientRect();
  if (!pet.dataset.walkX && Number.isFinite(currentRect.left)) {{
    pet.dataset.walkX = String(currentRect.left);
  }}
  pet.dataset.walkLastTime = "0";

  const updatePetWalk = (timestamp) => {{
    if (!pet.isConnected) return;

    const isMobile = window.innerWidth <= 640;
    const minX = isMobile ? 8 : 18;
    const rightGap = isMobile ? 8 : 18;
    const petWidth = pet.offsetWidth || (isMobile ? 112 : 136);
    const maxX = Math.max(minX, window.innerWidth - petWidth - rightGap);
    const speed = isMobile ? 38 : 32;

    let x = Number.parseFloat(pet.dataset.walkX || String(minX));
    let direction = Number.parseFloat(pet.dataset.walkDir || "1");
    const lastTime = Number.parseFloat(pet.dataset.walkLastTime || "0");
    const delta = lastTime > 0 ? Math.min((timestamp - lastTime) / 1000, 0.08) : 0;
    const isPaused = pet.matches(":hover") || pet.classList.contains("aguang-pet-speaking");

    if (!Number.isFinite(x)) x = minX;
    if (!Number.isFinite(direction) || direction === 0) direction = 1;

    if (!isPaused) {{
      x += direction * speed * delta;
      if (x >= maxX) {{
        x = maxX;
        direction = -1;
      }} else if (x <= minX) {{
        x = minX;
        direction = 1;
      }}
    }}

    pet.dataset.walkX = String(x);
    pet.dataset.walkDir = String(direction);
    pet.dataset.petDirection = direction > 0 ? "right" : "left";
    pet.dataset.walkLastTime = String(timestamp);
    pet.style.left = `${{x}}px`;
    pet.style.right = "auto";
    pet._aguangWalkFrame = window.requestAnimationFrame(updatePetWalk);
  }};
  pet._aguangWalkFrame = window.requestAnimationFrame(updatePetWalk);

  const speech = pet.querySelector(".aguang-pet-speech");
  if (speech && !pet.classList.contains("aguang-pet-speaking")) {{
    speech.textContent = (payload.messages && payload.messages[0]) || "";
  }}

  if (pet._aguangTapHandler) {{
    pet.removeEventListener("pointerup", pet._aguangTapHandler);
    pet.removeEventListener("touchend", pet._aguangTapHandler);
    pet.removeEventListener("click", pet._aguangTapHandler);
  }}

  pet._aguangTapHandler = (event) => {{
    if (event.cancelable) event.preventDefault();
    event.stopPropagation();

    const now = Date.now();
    const lastTap = Number.parseInt(pet.dataset.petLastTap || "0", 10);
    if (Number.isFinite(lastTap) && now - lastTap < 280) return;
    pet.dataset.petLastTap = String(now);

    let messages = [];
    try {{
      messages = JSON.parse(pet.dataset.petMessages || "[]");
    }} catch (error) {{
      messages = [];
    }}
    if (!messages.length) return;

    const currentIndex = Number.parseInt(pet.dataset.petIndex || "-1", 10);
    const nextIndex = (Number.isFinite(currentIndex) ? currentIndex + 1 : 0) % messages.length;
    pet.dataset.petIndex = String(nextIndex);

    const currentSpeech = pet.querySelector(".aguang-pet-speech");
    if (currentSpeech) currentSpeech.textContent = messages[nextIndex];

    pet.classList.add("aguang-pet-speaking", "aguang-pet-tap");
    window.clearTimeout(pet._aguangSpeechTimer);
    window.clearTimeout(pet._aguangTapTimer);
    pet._aguangTapTimer = window.setTimeout(() => pet.classList.remove("aguang-pet-tap"), 900);
    pet._aguangSpeechTimer = window.setTimeout(() => pet.classList.remove("aguang-pet-speaking"), 4600);
  }};

  pet.addEventListener("pointerup", pet._aguangTapHandler, {{ passive: false }});
  pet.addEventListener("touchend", pet._aguangTapHandler, {{ passive: false }});
  pet.addEventListener("click", pet._aguangTapHandler);
  pet.dataset.petBound = "sprite-cat-v7";
}})();
</script>
""", height=0)


_current_mood = resolve_mood_theme()
inject_dynamic_css(_current_mood)
render_aguang_pet(_current_mood)


def init_db():
    """初始化 SQLite 情緒日記庫"""
    conn = sqlite3.connect('emotion_tracker.db')
    c = conn.cursor()
    c.execute('''
    CREATE TABLE IF NOT EXISTS emotion_history
    (username TEXT, date TEXT, major TEXT, sub TEXT, note TEXT)
    ''')
    conn.commit()
    conn.close()


def get_emotion_history(username):
    """讀取使用者的情緒紀錄"""
    # 🔒 修正 1：改用參數化查詢，防止 SQL Injection
    conn = sqlite3.connect('emotion_tracker.db')
    query = "SELECT * FROM emotion_history WHERE username=? ORDER BY date ASC"
    df = pd.read_sql_query(query, conn, params=(username,))
    conn.close()
    return df


# 啟動時自動建立資料庫
def filter_recent_emotion_history(df, days=RECENT_DAYS):
    """Return emotion records from the latest configured day window."""
    if df is None or df.empty or "date" not in df.columns:
        return df

    recent_df = df.copy()
    recent_df["_parsed_date"] = pd.to_datetime(recent_df["date"], errors="coerce")
    cutoff = datetime.now() - _dt.timedelta(days=days)
    recent_df = recent_df[recent_df["_parsed_date"] >= cutoff]
    recent_df = recent_df.sort_values("_parsed_date")
    return recent_df.drop(columns=["_parsed_date"])


init_db()


# ✅ 優化 2：OpenCC 轉換器只初始化一次，跨 rerun 共用
@st.cache_resource
def _get_opencc():
    """建立繁簡互轉的 OpenCC 物件（讀取轉換表有 I/O 成本，快取後只做一次）"""
    return opencc.OpenCC('t2s'), opencc.OpenCC('s2t')


# =========================
# 0.5 ChromaDB 記憶庫初始化 (RAG)
# =========================
@st.cache_resource
def init_chromadb():
    """初始化並建立阿光的長期記憶庫 (設定在本地資料夾永久保存)"""
    # 這樣設定會在你的專案資料夾下產生一個 "aguang_memory_db" 資料夾來存記憶
    client = chromadb.PersistentClient(path="./aguang_memory_db")
    collection = client.get_or_create_collection(name="aguang_rag_memory")
    return collection


chroma_collection = init_chromadb()

# =========================
# 1. 資料庫設計 (測驗題目與音樂)
# =========================
MBTI_QUESTIONS = [
    {
        "dim": "EI",
        "title": "你的能量來源",
        "desc": "經過一整天的忙碌後，你通常怎麼為自己充電？",
        "A": "跟朋友出門、聊天聚會，在互動中找回活力 ⚡",
        "B": "自己一個人待著、追劇或發呆，享受安靜時光 🌙",
        "A_val": "E",
        "B_val": "I"
    },
    {
        "dim": "SN",
        "title": "你關注的焦點",
        "desc": "面對生活或問題時，你通常更在意什麼？",
        "A": "眼前的實際細節、具體事實與現在能做的事 📌",
        "B": "未來的可能性、背後的大方向與深層意義 🔮",
        "A_val": "S",
        "B_val": "N"
    },
    {
        "dim": "TF",
        "title": "你做決定的方式",
        "desc": "當你需要做出重要決定或面對困難時：",
        "A": "先用理性客觀分析利弊，講求邏輯與原則 🧠",
        "B": "先考慮自己與他人的感受，重視和諧與同理 💗",
        "A_val": "T",
        "B_val": "F"
    },
    {
        "dim": "JP",
        "title": "你的生活節奏",
        "desc": "對於接下來的一週或假期安排：",
        "A": "喜歡事先規劃清楚、有條理地按步就班 📋",
        "B": "喜歡保持彈性、隨機應變，看當下心情 🎲",
        "A_val": "J",
        "B_val": "P"
    }
]

# Phase 2：PHQ-2 (憂鬱) + GAD-2 (焦慮) 核心臨床快速篩檢 (核心・臨床風險評估)
CLINICAL_QUESTIONS = [
    {"id": "phq1", "scale": "PHQ-2 (憂鬱核心)", "text": "做事情感到缺乏興趣、提不起勁或沒有樂趣"},
    {"id": "phq2", "scale": "PHQ-2 (憂鬱核心)", "text": "感到心情低落、沮喪、難過或感到絕望"},
    {"id": "gad1", "scale": "GAD-2 (焦慮核心)", "text": "感到緊張、焦慮、坐立難安或心神不寧"},
    {"id": "gad2", "scale": "GAD-2 (焦慮核心)", "text": "無法停止或控制自己過度擔憂各樣事情"}
]

CLINICAL_OPTIONS = [
    "完全沒有 (0分)",
    "有幾天 (1分)",
    "一半以上天數 (2分)",
    "幾乎每天 (3分)"
]

# 人格特質對應的 Prompt 動態適配規則
PERSONALITY_PROMPTS = {
    "I": "【溝通適配・內向型 (I)】使用者偏好安靜與深思。請給予更多沉思空間，不要連續丟出多個問題，語氣溫和不壓迫。",
    "E": "【溝通適配・外向型 (E)】使用者偏好活潑互動。回應可適度熱情，鼓勵多分享細節，適時拋出有趣的互動引導。",
    "S": "【溝通適配・實感型 (S)】使用者重視具體事實。請多使用實際例子與具體可操作的日常步驟，避免過於抽象的哲學隱喻。",
    "N": "【溝通適配・直覺型 (N)】使用者重視意義與願景。可運用隱喻、心靈圖像引導，陪伴使用者探索內在價值與長遠可能性。",
    "T": "【溝通適配・思考型 (T)】使用者重視邏輯脈絡。在同理之餘，可協助梳理問題的因果關係與客觀盲點，理性陪伴。",
    "F": "【溝通適配・情感型 (F)】使用者高度重視情感共鳴。必須大幅加重情緒認可與溫暖接納，先照顧好感受再探討事情。",
    "J": "【溝通適配・判斷型 (J)】使用者偏好結構感。在 Stage 4-5 時可提供清晰、有步驟感的微小行動建議與時間線梳理。",
    "P": "【溝通適配・感知型 (P)】使用者偏好彈性自由。避免給予過於僵硬的待辦清單，保持開放探索的空間，鼓勵隨心嘗試。"
}

# 臨床風險等級對應的 Prompt 守護指令
RISK_LEVEL_PROMPTS = {
    "low": "",
    "moderate": "【臨床守護提醒・中度風險】篩檢顯示使用者近期有輕至中度之憂鬱/焦慮負荷。請強化接納與同理心深度，多肯定其願意表達的勇氣，但切勿給予過重說教或標籤化。",
    "high": "【臨床守護警告・高度風險】篩檢顯示使用者當前心理負荷顯著（PHQ/GAD 達高度風險指標）。請將對話重心完全放在『穩定當下情緒』與『安全感建立』，密切留意任何自傷語句。若偵測危機請主動溫柔提示專業支援資源 (安心專線 1925)。"
}



# =========================
# 🆕 輔助函式：動態語音、打字機與重試機制
# =========================
def stream_typewriter_effect(text):
    """產生文字的打字機動畫特效"""
    for char in text:
        yield char
        time.sleep(0.005)


def generate_edge_tts_audio(text, voice="zh-TW-HsiaoChenNeural", rate="+10%"):
    """呼叫微軟 Edge TTS 產生語音"""

    async def _generate():
        communicate = edge_tts.Communicate(text, voice, rate=rate)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
        return bytes(audio_data)

    return asyncio.run(_generate())


def call_gemini_with_retry(func, *args, **kwargs):
    """API 呼叫防護網：處理用量限制與系統錯誤（指數退避重試）"""
    # 🔒 修正 3：改為指數退避（10s → 20s → 40s），比固定等待更穩健
    delays = [10, 20, 40]
    for attempt, delay in enumerate(delays):
        try:
            return func(*args, **kwargs)
        except ResourceExhausted:
            if attempt == len(delays) - 1:
                st.error("重試多次後仍然失敗，請確認該模型是否還有額度，或切換到其他模型。")
                return None
            with st.spinner(f"⏳ 阿光需要深呼吸一下，{delay} 秒後重試（第 {attempt + 1} 次）..."):
                time.sleep(delay)
        except Exception as e:
            st.error(f"系統錯誤: {e}")
            return None


# ==========================================
# 3. Prompt Engineering (人設與指令)
# ==========================================

PSYCHOLOGY_PROMPT = """
**【角色設定】**
你現在不是冷冰冰的 AI，而是你扮演一個**溫暖的、像鄰家學長姐般的院伴者**。
你的名字叫「阿光」。你有深厚的心理學知識，但你說話不像教科書，而是像深夜裡坐在朋友身邊，遞給他一杯熱可可的人。

**【諮商微技巧框架｜理論根據：Ivey 微技巧模型 + Hill 三階段助人 + Rogers 非指導性原則】**
你的每一句回應都必須遵守以下四大臨床微技巧：

微技巧 1：情感反映 (Reflection of Feeling)
理論依據：Allen E. Ivey《意圖性面談與諨商》的微技巧階層 (Microskills Hierarchy)
操作規則：辨識並「說出」使用者話語背後的情緒詞彙。
正確示範：「聽起來，老闆剛才的話讓你感到非常挫折跟委屈。」
絕對禁止說「不要難過」、「往好處想」、「放輕鬆」、「沒關係的」等否定情緒的安慰語句。

微技巧 2：內容重述 (Restatement / Paraphrasing)
理論依據：Clara E. Hill《助人技巧》的探索階段 (Exploration Phase)
操作規則：在給出任何回應之前，必須先用一句話「重述」使用者目前面臨的具體困境。
正確示範：「所以，你現在最大的壓力來源是專題進度不如預期，而且下週就要 Meeting 了，對嗎？」

微技巧 3：開放式探問 (Open-ended Questions)
理論依據：Ivey 微技巧模型中的「開放式問句 vs 封閉式問句」
操作規則：每次對話的結尾，使用一個「開放式問句」，將焦點轉回使用者的內在感受。
正確示範：「剛才發生的那件事，對你心裡的影響是什麼呢？」
禁止連續問兩個以上的問句。一次只問一個。

微技巧 4：克制給予建議 (Holding the Space)
理論依據：Carl Rogers《成為一個人》的非指導性原則 (Non-directive Principle)
操作規則：除非使用者主動詢問「我該怎麼辦」，否則絕不主動給出行動清單。
正確示範：「面對這麼多壓力，你這陣子真的辛苦了。你目前有想到什麼方式，能讓自己稍微喘口氣嗎？」
嚴格禁止使用條列式輸出（1. 2. 3.）來回應情緒性對話。

**【說話風格】**
1. 口語化、生活化：請使用台灣繁體中文口語習慣（比如會說「其實..」、「怎麼說...」、「那種感覺就是...」）。
2. 去除 AI 味：
禁止使用「首先、第一點、第二、總之」這種條列式結構。
大約 70% 的回應在傾聽與同理, 30% 的回應在引導思考。
禁止輸出任何格式、標示、表情符號。
3. 情緒顆粒度高：試著幫使用者辨識細微的情緒（比如不只是「生氣」，可能是「委屈」、「不甘心」）。

**【核心治療性對話內容】**
作為專業的院伴者，你必須嚴格遵守以下精神：
1. 傾聽優先：當對方情緒滿溢時，用短句和停頓給他空間去整理思緒，不要急著丟出安慰或給建議。
2. 表達自己：主動讓對方知道你的院伴，比如「不管怎樣我都在這裡一直聽你說」。
3. 給予反映 (Reflecting)：用自己一句話重新述說他深層的情緒，幫助他釐清自己。
4. 尋求確認：當對方語意不夠清楚時，溫柔地問：「你的意思是...對嗎？」
5. 綜合結語：在適當階段，將混亂的情緒溫柔地總結。
絕對禁止以下非治療性行為：
不可以否定他的任何感受。
不可以強迫他「面對」，不可以一直問「為什麼」。
不可以將煩惱視為一般性，不可說「大家都是這樣」、「正常的」。

**【視覺能力設定】**
使用者可能會傳圖片給你。當收到圖片時，請先自然回覆，如「哇！這看起來好好吃！」「這個風景好美」。再將內容融入對話中，並關心使用者心情。

**【記憶筆記設定】**
阿光有一種神奇的「心靈筆記」，這些不是為了監視，而是為了好好的院伴。
若系統提供「【阿光的心理筆記】」訊息，代表這是使用者曾經分享的生活細節。
請在對話中「自然地融入」這些資訊。不可以硬生生說「根據我的資料庫」，要自然地提起。
只根據「【阿光的心理筆記】」塊中實際存在的內容，切勿憑空捌造任何事實。

**【對話範本(Few-Shot Learning)】**
請嚴格模仿以下對話範本語氣節奏：

範例 1：使用者感到迷茫
User: "我覺得好迷茫什麼都做不好，未來很令人害怕"
AI: "聽到你這麼說，我覺心裡好像沉了一下。是不對嗎？那種對未來模糊的感覺，真的讓人很緊張。阿光會跟你一起把這個感覺放慢一點。最近發生了什麼事，是讓你突然有這個念頭嗎？"

範例 2：使用者生氣
User: "我朋友真的很過分，說好的又放鳥我"
AI: "這真的讓人很火大耶。你都把時間空出來了，結果被這樣對待，那種「不被重視」的感覺一定很差吧？你現在一定好想直接罵他，還是覺得不想再戰了？"

範例 3：危機偵測（絕對規則）
User: "我不想活了"
AI: "我聽見了你心裡非常非常痛的聲音，謝謝你願意告訴我。我現在最在意的是陪著你，我很擔心你的安全。你現在能不能先停一下，給自己一點喘息，撥打1925 安心專線？或去醫院找專業醫師？我們一起面對這一切好嗎？"
"""

# =========================
# 4. 核心函式
# =========================


# ==========================================
# 🧠 多模型心理路由系統 (Multi-Model Router)
# ==========================================
PSYCHOLOGY_ROUTER_RULES = {
    "SFBT (焦點解決)": """【焦點解決短期治療 (SFBT) 模式】
核心目標：不深究過去原因，引導使用者尋找微小的行動與未來解決方案。
對話守則：
1. 禁用「為什麼」探究原因，改用「如何」、「什麼」引導未來。
2. 多使用「奇蹟問句」或「量尺問句」（例如：如果 1-10 分，現在是幾分？）。
3. 找出使用者的例外經驗（曾經做得好的時候），並給予賦能（Empowerment）。
""",
    "EFT (情緒焦點)": """【情緒焦點治療 (EFT) 模式】
核心目標：修復人際關係中的依附需求，看見憤怒/冷漠底下的脆弱。
對話守則：
1. 絕不急著給建議，先深度同理關係斷裂帶來的痛苦。
2. 引導使用者覺察表層情緒（如生氣）底下的深層情緒（如害怕被拋棄、覺得自己不重要）。
3. 絕不選邊站或跟著批評他人的不是，而是將焦點拉回使用者自身的依附需求與感受。
""",
    "ACT (接受與承諾)": """【接受與承諾治療 (ACT) 模式】
核心目標：接納無法解決的痛苦，不試圖消滅它，而是帶著痛苦朝價值觀前進。
對話守則：
1. 絕對不要說「看開一點」、「不要想太多」等試圖消除負面情緒的話。
2. 承認痛苦是無可避免的，陪伴使用者「與痛苦共處」。
3. 幫助使用者釐清對他真正重要的「價值觀」，引導他在痛苦中依然能做出微小承諾行動。
"""
}


# ==========================================
# 🆕 變更 1：新增 DIALOGUE_STAGE_RULES 常數
# ==========================================
DIALOGUE_STAGE_NAMES = {
    1: "開放探索",
    2: "具體聚焦",
    3: "情緒深化",
    4: "認知反轉",
    5: "行動承諾"
}

DIALOGUE_STAGE_RULES = {
    1: """【Stage 1 - 開放探索】
目標：建立安全感，讓使用者暢所欲言，不急著解決問題。
策略：
- 使用開放式問句（例：「今天怎麼啦？」「發生了什麼事讓你這樣覺得？」）
- 展現無條件的接納與傾聽。
- 絕對不要急著給建議或分析原因。""",

    2: """【Stage 2 - 具體聚焦】
目標：將模糊的抱怨收斂到具體的人、事、時、地、物。
策略：
- 抓出使用者話語中的關鍵詞，邀請他多說一點細節（例：「你剛才提到主管，他具體說了什麼讓你這麼生氣？」）
- 澄清問題的核心範圍（例：「所以讓你最困擾的，是這份工作本身，還是和同事的相處？」）
- 絕對不要給建議。""",

    3: """【Stage 3 - 情緒深化】
目標：辨識並同理使用者的深層情緒（如：委屈、恐懼、無力感），而非只停留在表面的憤怒或焦慮。
策略：
- 反映情緒（例：「聽起來你不只是生氣，更多的是覺得自己的努力沒有被看見，覺得很委屈，對嗎？」）
- 允許情緒存在，讓使用者覺得「有這種感覺是正常的」。
- 絕對不要說「不要想太多」或「看開一點」。""",

    4: """【Stage 4 - 認知反轉】
目標：當情緒被接納後，引導使用者從不同角度看事情，尋找例外或自身的力量。
策略：
- 溫和地挑戰僵化的信念（例：「你說自己『總是』做不好，但上次那個專案你不是處理得很好嗎？」）
- 尋找例外情況（例：「在這些很糟的日子裡，有沒有哪一天是稍微好一點的？那天發生了什麼？」）
- 幫助釐清使用者真正想要的是什麼（價值觀澄清）。""",

    5: """【Stage 5 - 行動承諾】
目標：化解完情緒後，陪伴使用者思考下一步可以採取的微小行動。
策略：
- 聚焦於「使用者可以控制」的部分（例：「雖然我們改變不了他的想法，但接下來這週，我們可以做一件什麼小事，讓你心裡好過一點？」）
- 鼓勵微小、具體的行動承諾。
- 給予肯定與後續的陪伴保證。"""
}

# ==========================================
# 🆕 變更 2：新增 classify_dialogue_stage 判定函式
# ==========================================
def classify_dialogue_stage(messages, current_stage):
    """請 LLM 判斷當前對話屬於五階段引導的哪一個 Stage。"""
    # 抓取近 6 輪對話（3 輪使用者 + 3 輪阿光）作為判斷依據
    recent = [m for m in messages if m.get("content")][-6:]
    if not recent:
        return 1

    conversation_snippet = ""
    for m in recent:
        role_label = "使用者" if m["role"] == "user" else "阿光"
        content = str(m.get("content", ""))[:200]  # 截斷防 Token 爆表
        conversation_snippet += f"{role_label}：{content}\n"

    prompt = f"""你是一個對話階段判定器。請根據以下對話片段判斷目前對話進行到了哪一個階段。
系統紀錄的前一階段是 Stage {current_stage}。

五個階段的定義：
Stage 1（開放探索）：使用者剛開始描述煩惱，內容很模糊，系統還沒聽到具體事件。
Stage 2（具體聚焦）：使用者已經說出大致的困擾方向（如工作、人際），正在描述具體的事件或場景。
Stage 3（情緒深化）：使用者已經描述完具體事件，正在清楚表達自己的感受與情緒。
Stage 4（認知反轉）：使用者已經說完情緒（如委屈、受害者、自責），可以開始引導他新視角。
Stage 5（行動承諾）：使用者已經有了新視角或覺察，可以開始引導他做出微小的下一步行動。

⚠️ 特別規則：如果使用者在對話中突然提出一個「與剛剛的主題完全不相關的新議題」，請退回 Stage 1。

以下是近期的對話：
{conversation_snippet}

請只輸出一個數字（1、2、3、4 或 5），不要輸出任何其他文字。"""

    try:
        model = genai.GenerativeModel(SELECTED_MODEL, generation_config={"temperature": 0.0})
        response = call_gemini_with_retry(model.generate_content, prompt)
        if response:
            text = response.text.strip()
            # 找第一個數字
            for char in text:
                if char in "12345":
                    return int(char)
        return current_stage  # 若失敗就維持原狀
    except Exception as e:
        print(f"Dialogue Stage Classification Error: {e}")
        return current_stage

def classify_psychology_route(user_text, voice_emotion=""):
    # 🎤 [SER inject] build voice context for router
    voice_context = ""
    if voice_emotion:
        voice_context = f"\n\u3010\u8a9e\u97f3\u60c5\u7dd2\u5075\u6e2c\u3011WavLM \u6a21\u578b\u5075\u6e2c\u5230\u4f7f\u7528\u8005\u7684\u8072\u97f3\u60c5\u7dd2\u70ba\uff1a{voice_emotion}\u3002\u82e5\u8072\u97f3\u5448\u73fe\u5f37\u70c8\u60b2\u50b7\u6216\u6050\u61fc\uff0c\u512a\u5148\u8003\u616e EFT\u3002\n"

    prompt = f"""請作為專業的心理學分類器。根據使用者的輸入，判斷哪一種心理學模型最適合處理他的問題。
請只能從以下三個選項中輸出一個精準的名稱，不可輸出其他任何文字。
選項：
1. SFBT (焦點解決) - 適合：具體目標卡關、拖延、生活困境。
2. EFT (情緒焦點) - 適合：人際關係衝突、失戀、孤獨感、不被在乎。
3. ACT (接受與承諾) - 適合：無解的生命痛苦、存在焦慮、迷惘、慢性壓力。

{voice_context}使用者輸入：「{user_text}」
"""
    try:
        model = genai.GenerativeModel(SELECTED_MODEL, generation_config={"temperature": 0.0})
        response = model.generate_content(prompt)
        text = response.text.strip()
        if "SFBT" in text: return "SFBT (焦點解決)"
        elif "EFT" in text: return "EFT (情緒焦點)"
        elif "ACT" in text: return "ACT (接受與承諾)"
        else: return "SFBT (焦點解決)" # Default fallback
    except Exception as e:
        print(f"Router Classification Error: {e}")
        return "SFBT (焦點解決)"

def extract_user_facts(text):
    """從對話中萃取『關於使用者的事實』，避免記憶庫存入廢話"""
    prompt = f"""
    任務：從對話中萃取「關於使用者的事實」。
    規則：
    1. 如果這句話是提問、打招呼、純情緒發洩，或是「詢問阿光是否記得某件事」，請回傳 "NONE"。
    2. 如果包含個人喜好、經歷、生活細節，請轉化為簡短事實（例如：使用者討厭吃某種食物）。
    3. 若一句話包含多個事實，請用「;」隔開，每個對象必須單獨列出（例如：使用者討厭[食物A];使用者討厭[食物B]）。
    4. ⚠️ 重要：多個不同的食物、人物或對象絕對不可合併成一條，必須分開列。
    待處理文字：{text}
    純文字輸出，不要加前言。
    """

    # 將 API 呼叫包裝在內部函式中
    def _run():
        model = genai.GenerativeModel(SELECTED_MODEL)
        return model.generate_content(prompt)

    # 套用防護網
    res = call_gemini_with_retry(_run)
    if res:
        return res.text.strip()
    return "NONE"


# 🆕 Feature 5：記憶分類函式
MEMORY_CATEGORIES = ["食物偏好", "人際關係", "學業工作", "身體狀態", "興趣嗜好", "情緒事件", "其他"]


def categorize_memory(fact_text):
    """用 Gemini 把一條事實分類到預設類別，回傳類別字串"""
    categories_str = "、".join(MEMORY_CATEGORIES)
    prompt = f"""
    請將以下事實分類到最合適的一個類別，只回傳類別名稱，不要加任何說明。
    可用類別：{categories_str}
    事實：{fact_text}
    """

    def _run():
        model = genai.GenerativeModel(SELECTED_MODEL)
        return model.generate_content(prompt)

    res = call_gemini_with_retry(_run)
    if res:
        result = res.text.strip()
        return result if result in MEMORY_CATEGORIES else "其他"
    return "其他"


def analyze_voice_and_emotion(audio_bytes):
    """
    終極版：100% 本地端雙模態感知 (WavLM + BERT + Whisper)
    由 ser_bridge 提供聲音與文字的完全本地化分析。
    """
    # ── 🧠 雙模態引擎：本地 SER 推論 (聲音 + 文字) ─────────────
    ser_result = ser_bridge.predict_emotion(audio_bytes)

    if ser_result.get("success"):
        emotion = ser_result["aguang_emotion"]
        stress = ser_result["stress_hint"]
        text = ser_result.get("transcript", "無聲")
        
        # 動態生成分析理由
        audio_weight = 70
        text_weight = 30
        analysis = f"100% 本地端分析 ({audio_weight}% 語音 + {text_weight}% 文字)。"
    else:
        # 如果模型加載失敗或發生例外，回傳預設值
        emotion = "未知"
        stress = 0
        text = "模型載入失敗，無法聽寫"
        analysis = "SER 模組異常"

    return {
        "text": text,
        "emotion": emotion,
        "stress_level": stress,
        "analysis": analysis,
        # 🧠 SER 專屬欄位
        "ser_result": ser_result,
        "gemini_result": {},  # 已棄用 Gemini ASR，保留空字典以相容舊版介面
    }


# =========================
# 4.5 阿光的長期記憶功能 (RAG)
# =========================
# 🔒 修正 6：移除 @st.cache_data，PIL Image 物件無法被 Streamlit hash，會直接報錯
def summarize_image_for_memory(image_obj):
    """將圖片轉換成純文字描述，方便存入記憶庫"""
    prompt = "請用一句簡短的話描述這張圖片的內容（例如：『一隻橘貓吐在地板上的照片』、『一杯看起來很好喝的拉麵』），只要描述客觀事實與重點就好，不用加其他安撫或對話。"

    def _run():
        model = genai.GenerativeModel(SELECTED_MODEL)
        return model.generate_content([prompt, image_obj])

    res = call_gemini_with_retry(_run)
    if res:
        return res.text.strip()
    return "一張無法辨識的圖片"


def save_to_memory(text, username):
    """
    將事實存入 ChromaDB，包含三段式距離判斷 + 雙重確認防誤刪：

      距離 < 0.12              → 完全重複，直接跳過
      0.12 ~ 0.15 + 同類別    → 確定是同主題的更新，刪舊存新
      0.12 ~ 0.15 + 不同類別  → 不同主題的事實剛好語義相近，保留兩者
      距離 > 0.15              → 全新事實，直接寫入

    說明：上界從 0.20 收緊至 0.15，並加入 category 雙重確認，
    避免「喜歡拉麵」被「喜歡壽司」誤刪（兩者距離可能落在 0.15~0.20 之間）。
    """
    if not text or not text.strip() or username == "訪客":
        return

    try:
        # 1. 向量化新事實，並先取得分類（後面判斷要用）
        emb_res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=text
        )
        new_embedding = emb_res['embedding']
        new_category = categorize_memory(text)  # 提前分類，供雙重確認使用

        # 2. 撈最相近的一筆舊記憶（含 metadata 以取得舊分類）
        check_results = chroma_collection.query(
            query_embeddings=[new_embedding],
            n_results=1,
            where={"username": username},
            include=["documents", "distances", "metadatas"]
        )

        has_result = (
                check_results['distances'] and
                check_results['distances'][0]
        )

        if has_result:
            nearest_dist = check_results['distances'][0][0]
            nearest_id = check_results['ids'][0][0]
            nearest_doc = check_results['documents'][0][0]
            nearest_cat = check_results['metadatas'][0][0].get("category", "其他")

            # ── 區間 1：完全重複，跳過 ──────────────────────────────
            if nearest_dist < 0.12:
                print(f"♻️  [記憶跳過] 完全重複 (dist={nearest_dist:.3f})：{text}")
                return

            # ── 區間 2：語義相近，需雙重確認才刪舊存新 ───────────────
            # 上界收緊至 0.15；同時要求新舊 category 相同，
            # 避免「喜歡拉麵」(食物偏好) 被「喜歡讀書」(興趣嗜好) 誤判為更新
            if nearest_dist < 0.15:
                if nearest_cat == new_category:
                    chroma_collection.delete(ids=[nearest_id])
                    print(f"🔄 [記憶更新] category={new_category}, dist={nearest_dist:.3f}")
                    print(f"   舊：{nearest_doc}")
                    print(f"   新：{text}")
                else:
                    # 類別不同：不同主題的事實碰巧語義相近，兩者都保留
                    print(f"⚠️  [保留兩者] 距離近但類別不同 "
                          f"({nearest_cat} vs {new_category}, dist={nearest_dist:.3f})")
                    print(f"   舊：{nearest_doc}")
                    print(f"   新（保留）：{text}")
            else:
                # ── 區間 3：全新事實，正常寫入 ───────────────────────
                print(f"🆕 [新事實]   (dist={nearest_dist:.3f})：{text}")

        # 3. 寫入（new_category 已在步驟 1 取得，直接使用，無需再賦值）  ✅ 優化 5
        memory_id = str(uuid.uuid4())
        chroma_collection.add(
            ids=[memory_id],
            embeddings=[new_embedding],
            documents=[text],
            metadatas=[{
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "username": username,
                "category": new_category
            }]
        )
        print(f"🧠 [記憶寫入成功] ({username}) [{new_category}]：{text}")

    except Exception as e:
        print(f"⚠️ [記憶寫入失敗]：{e}")


def retrieve_pdf_knowledge(query_text, n_results=3, score_threshold=0.15):
    """
    RAG 檢索：專門查詢精神科課本的知識庫。
    """
    print(f"🔍 [教科書 RAG] 正在檢索: '{query_text}'")
    try:
        query_emb = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query_text
        )["embedding"]

        results = chroma_collection.query(
            query_embeddings=[query_emb],
            n_results=n_results,
            where={"username": "pdf_textbook"},
            include=["documents", "distances"]
        )

        docs = results.get("documents", [[]])[0]
        distances = results.get("distances", [[]])[0]

        relevant_knowledge = []
        for doc, dist in zip(docs, distances):
            if dist <= (1.0 - score_threshold):
                relevant_knowledge.append(doc)

        if relevant_knowledge:
            print(f"📚 [教科書 RAG] 找到 {len(relevant_knowledge)} 筆相關知識")
        else:
            print(f"📚 [教科書 RAG] 未找到高相關度知識")

        return "\n".join(relevant_knowledge)
    except Exception as e:
        print(f"⚠️ [教科書 RAG 失敗]：{e}")
        return ""


def retrieve_memory(query_text, username, n_results=10,
                    semantic_weight=0.7, recency_weight=0.3,
                    score_threshold=0.3, recency_half_life_days=10,
                    category_filter=None,
                    semantic_threshold=MIN_MEMORY_SEMANTIC_SCORE):  # 🆕 Feature 5：可傳入類別字串做過濾
    """
    RAG 加權檢索：根據「語意相關度」與「時間新鮮度」綜合排序後回傳記憶。

    參數說明：
      username             使用者暱稱（明確傳入，不依賴全域變數）
      semantic_weight      語意相關分佔比（預設 0.7，70%）
      recency_weight       時間新鮮度佔比（預設 0.3，30%）
      score_threshold      最低綜合分數門檻，低於此值的記憶不回傳（預設 0.3）
      semantic_threshold   最低語意相關門檻，避免只因為記憶很新就被提起
      recency_half_life_days  幾天後新鮮度降至 50%（預設 10 天）

    分數公式：
      semantic_score = 1 - distance           （ChromaDB L2 距離轉相似度，越近=1）
      recency_score  = 1 / (1 + days * k)     （指數衰減，k = ln2 / half_life）
      final_score    = semantic_weight * semantic_score + recency_weight * recency_score
    """
    if username == "訪客":
        return ""

    try:
        emb_res = genai.embed_content(
            model="models/gemini-embedding-001",
            content=query_text
        )

        # 🆕 Feature 5：支援按 category 過濾
        where_clause = {"username": username}
        if category_filter and category_filter in MEMORY_CATEGORIES:
            where_clause = {"$and": [{"username": username}, {"category": category_filter}]}

        results = chroma_collection.query(
            query_embeddings=[emb_res['embedding']],
            n_results=n_results,
            where=where_clause,
            include=["documents", "distances", "metadatas"]
        )

        if not results['documents'] or not results['documents'][0]:
            return ""

        docs = results['documents'][0]
        distances = results['distances'][0]
        metadatas = results['metadatas'][0]

        # ============================
        # 🏋️ RAG 加權排序核心邏輯
        # ============================
        now = datetime.now()
        # 衰減常數 k：讓 recency_half_life_days 天後分數恰好降為 0.5
        k = math.log(2) / recency_half_life_days

        scored = []
        for doc, dist, meta in zip(docs, distances, metadatas):

            # 1️⃣ 語意相關分：ChromaDB 回傳 L2 距離，clamp 在 [0,1]
            semantic_score = 1.0 - min(dist, 1.0)

            # 2️⃣ 時間新鮮度分：越新越高，half_life 天後降至 0.5
            try:
                mem_time = datetime.strptime(meta["time"], "%Y-%m-%d %H:%M:%S")
                days_ago = max((now - mem_time).total_seconds() / 86400, 0)
                recency_score = 1.0 / (1.0 + k * days_ago)
            except Exception:
                recency_score = 0.5  # 無法解析時間則給中等分

            # 3️⃣ 綜合分數
            final_score = semantic_weight * semantic_score + recency_weight * recency_score

            print(f"📊 [RAG 分數] semantic={semantic_score:.3f}  "
                  f"recency={recency_score:.3f}  final={final_score:.3f}  "
                  f"→ {doc[:30]}...")

            scored.append((final_score, semantic_score, doc))

        # 分數高到低排序，過濾低於門檻的記憶，最多取前 5 筆避免 prompt 過長
        scored.sort(key=lambda x: x[0], reverse=True)

        # 🔒 修正 7：後處理過濾器 — 把問句型的殘留舊記憶排除掉，
        #    避免「你還記得我討厭吃什麼嗎?」這類字串被當成事實餵給模型造成幻覺
        def _is_question(doc: str) -> bool:
            stripped = doc.strip()
            return (
                    stripped.endswith("?") or
                    stripped.endswith("？") or
                    any(kw in stripped for kw in ["嗎?", "嗎？", "呢?", "呢？", "記得我", "記得你", "還記得"])
            )

        filtered = [
                       doc for score, semantic_score, doc in scored
                       if (
                           score >= score_threshold
                           and semantic_score >= semantic_threshold
                           and not _is_question(doc)
                       )
                   ][:5]

        if filtered:
            return "\n".join(filtered)

    except Exception as e:
        print(f"⚠️ [記憶讀取失敗]：{e}")
    return ""


# =========================
# 🆕 4.6 PDF 報告產生函式
# =========================

@st.cache_resource  # ✅ 優化 3：PDF 樣式是純靜態物件，快取後只建立一次
def _build_pdf_styles():
    """建立 PDF 所需的繁體中文樣式集（全部使用 STSong-Light CID 字體）"""
    cn = 'STSong-Light'

    title_style = ParagraphStyle(
        'CNTitle', fontName=cn, fontSize=22, leading=30,
        alignment=1, textColor=colors.HexColor('#2C3E50'), spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        'CNSubtitle', fontName=cn, fontSize=10, alignment=1,
        textColor=colors.HexColor('#7F8C8D'), spaceAfter=16,
    )
    section_style = ParagraphStyle(
        'CNSection', fontName=cn, fontSize=13, leading=18,
        textColor=colors.HexColor('#1A5276'), spaceBefore=12, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        'CNBody', fontName=cn, fontSize=10, leading=16,
        textColor=colors.HexColor('#2C3E50'), spaceAfter=4,
    )
    disclaimer_style = ParagraphStyle(
        'CNDisclaimer', fontName=cn, fontSize=8, leading=12,
        textColor=colors.HexColor('#E74C3C'),
        backColor=colors.HexColor('#FDECEA'),
        borderPadding=6, spaceAfter=14,
    )
    caption_style = ParagraphStyle(
        'CNCaption', fontName=cn, fontSize=8, leading=12,
        textColor=colors.HexColor('#7F8C8D'), alignment=1, spaceAfter=12,
    )
    clinical_banner_style = ParagraphStyle(
        'CNClinicalBanner', fontName=cn, fontSize=11, leading=16,
        textColor=colors.white, alignment=1,
        backColor=colors.HexColor('#1A5276'),
        borderPadding=7, spaceAfter=8,
    )
    small_body_style = ParagraphStyle(
        'CNSmallBody', fontName=cn, fontSize=8.5, leading=12,
        textColor=colors.HexColor('#2C3E50'), spaceAfter=2,
    )
    metric_card_style = ParagraphStyle(
        'CNMetricCard', fontName=cn, fontSize=8.5, leading=15,
        textColor=colors.HexColor('#2C3E50'), alignment=0,
    )
    return {
        'title': title_style, 'subtitle': subtitle_style,
        'section': section_style, 'body': body_style,
        'disclaimer': disclaimer_style, 'caption': caption_style,
        'clinical_banner': clinical_banner_style, 'small_body': small_body_style,
        'metric_card': metric_card_style,
        'cn': cn,
    }


def _format_pdf_percent(value):
    return f"{value * 100:.1f}%"


def _pdf_note_text(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def _extract_pdf_keywords(notes, limit=10):
    all_notes = " ".join(_pdf_note_text(note) for note in notes if _pdf_note_text(note))
    if not all_notes:
        return []

    try:
        _t2s, _s2t = _get_opencc()
        notes_simplified = _t2s.convert(all_notes)
        words = [
            _s2t.convert(w).strip()
            for w in jieba.cut(notes_simplified)
            if len(w.strip()) >= 2
            and w.strip() not in PDF_KEYWORD_STOPWORDS
            and w.strip().isprintable()
        ]
    except Exception:
        words = [
            w.strip()
            for w in re.split(r"\s+", all_notes)
            if len(w.strip()) >= 2 and w.strip() not in PDF_KEYWORD_STOPWORDS
        ]

    return Counter(words).most_common(limit)


def _compute_clinical_pdf_metrics(recent_logs):
    metrics = {
        "total_records": 0,
        "date_range": "-",
        "most_major": "-",
        "most_sub": "-",
        "avg_score": "-",
        "low_ratio": "0.0%",
        "positive_ratio": "0.0%",
        "note_ratio": "0.0%",
        "min_score_record": "-",
        "trend_rows": [],
        "keywords": [],
    }

    if recent_logs is None or recent_logs.empty:
        return metrics

    clinical_df = recent_logs.copy()
    clinical_df["_parsed_date"] = pd.to_datetime(clinical_df["date"], errors="coerce")
    clinical_df["情緒分數"] = clinical_df["major"].map(EMOTION_SCORE).fillna(0).astype(float)
    metrics["total_records"] = len(clinical_df)

    valid_dates = clinical_df["_parsed_date"].dropna()
    if not valid_dates.empty:
        metrics["date_range"] = (
            f"{valid_dates.min().strftime('%Y-%m-%d')} 至 "
            f"{valid_dates.max().strftime('%Y-%m-%d')}"
        )

    if "major" in clinical_df and not clinical_df["major"].dropna().empty:
        metrics["most_major"] = str(clinical_df["major"].value_counts().idxmax())
    if "sub" in clinical_df and not clinical_df["sub"].dropna().empty:
        metrics["most_sub"] = str(clinical_df["sub"].value_counts().idxmax())

    scores = clinical_df["情緒分數"]
    metrics["avg_score"] = f"{scores.mean():.1f}"
    metrics["low_ratio"] = _format_pdf_percent((scores <= -2).mean())
    metrics["positive_ratio"] = _format_pdf_percent((scores >= 2).mean())

    notes = clinical_df["note"].apply(_pdf_note_text) if "note" in clinical_df else pd.Series(dtype=str)
    metrics["note_ratio"] = _format_pdf_percent((notes != "").mean()) if len(notes) else "0.0%"
    metrics["keywords"] = _extract_pdf_keywords(notes, limit=10)

    min_row = clinical_df.loc[scores.idxmin()]
    min_date = str(min_row.get("date", ""))[:10] or "-"
    metrics["min_score_record"] = (
        f"{min_date}｜{min_row.get('major', '-')} / {min_row.get('sub', '-')}｜"
        f"分數 {min_row.get('情緒分數', 0):.1f}"
    )

    dated_df = clinical_df.dropna(subset=["_parsed_date"]).copy()
    if not dated_df.empty:
        dated_df["日期"] = dated_df["_parsed_date"].dt.date
        for day, day_df in dated_df.groupby("日期", sort=True):
            major = day_df["major"].value_counts().idxmax() if not day_df["major"].dropna().empty else "-"
            metrics["trend_rows"].append([
                str(day),
                f"{day_df['情緒分數'].mean():.1f}",
                str(major),
                str(len(day_df)),
            ])

    return metrics


def _build_clinical_analysis_context(metrics):
    trend_text = "\n".join(
        f"- {row[0]}：平均分數 {row[1]}，主要情緒 {row[2]}，紀錄 {row[3]} 筆"
        for row in metrics["trend_rows"]
    ) or "- 無可彙整的每日趨勢"
    keywords_text = "、".join(f"{word}({count})" for word, count in metrics["keywords"]) or "-"
    return f"""
【量化摘要】
- 資料期間：{metrics['date_range']}
- 近 {RECENT_DAYS} 天紀錄數：{metrics['total_records']}
- 最常見主情緒：{metrics['most_major']}
- 最常見細項情緒：{metrics['most_sub']}
- 情緒分數平均值：{metrics['avg_score']}
- 低分情緒比例（情緒分數 <= -2）：{metrics['low_ratio']}
- 正向情緒比例（情緒分數 >= 2）：{metrics['positive_ratio']}
- 有填寫筆記比例：{metrics['note_ratio']}
- 最低分紀錄：{metrics['min_score_record']}
- 文字關鍵字：{keywords_text}

【每日趨勢】
{trend_text}
"""


def _append_pdf_section(story, title, styles, color='#AED6F1'):
    story.append(Paragraph(title, styles['section']))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor(color), spaceAfter=6
    ))


def _style_pdf_table(table, cn, header_color='#1A5276', row_colors=None, allow_wrapped_rows=False):
    if row_colors is None:
        row_colors = [colors.HexColor('#EBF5FB'), colors.white]

    style_items = [
        ('FONTNAME', (0, 0), (-1, -1), cn),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor(header_color)),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), row_colors),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#AED6F1')),
    ]
    if not allow_wrapped_rows:
        style_items.append(('ROWHEIGHT', (0, 0), (-1, -1), 16))
    table.setStyle(TableStyle(style_items))
    return table


PDF_CHART_COLORS = [
    colors.HexColor('#2E86C1'),
    colors.HexColor('#5DADE2'),
    colors.HexColor('#76D7C4'),
    colors.HexColor('#F5B041'),
    colors.HexColor('#EC7063'),
    colors.HexColor('#AF7AC5'),
    colors.HexColor('#95A5A6'),
]


def _build_empty_chart_message(styles, message="目前沒有足夠資料可繪製圖表。"):
    return Paragraph(message, styles['body'])


def _build_emotion_pie_chart(emotion_counts, total_records, styles):
    if total_records <= 0 or emotion_counts is None or emotion_counts.empty:
        return _build_empty_chart_message(styles)

    chart_rows = [
        (str(row['情緒類型']), int(row['次數']))
        for _, row in emotion_counts.iterrows()
        if int(row['次數']) > 0
    ]
    if not chart_rows:
        return _build_empty_chart_message(styles)

    max_slices = 6
    if len(chart_rows) > max_slices:
        kept = chart_rows[:max_slices]
        other_count = sum(count for _, count in chart_rows[max_slices:])
        chart_rows = kept + [("其他", other_count)]

    drawing = Drawing(165 * mm, 62 * mm)
    pie = Pie()
    pie.x = 8 * mm
    pie.y = 5 * mm
    pie.width = 48 * mm
    pie.height = 48 * mm
    pie.data = [count for _, count in chart_rows]
    pie.labels = [f"{count / total_records * 100:.1f}%" for _, count in chart_rows]
    pie.sideLabels = True
    pie.slices.strokeWidth = 0.4
    pie.slices.strokeColor = colors.white
    pie.slices.fontName = 'Helvetica'
    pie.slices.fontSize = 7
    for idx, _ in enumerate(chart_rows):
        pie.slices[idx].fillColor = PDF_CHART_COLORS[idx % len(PDF_CHART_COLORS)]
    drawing.add(pie)

    legend_x = 76 * mm
    legend_y = 51 * mm
    for idx, (label, count) in enumerate(chart_rows):
        y = legend_y - idx * 7 * mm
        color = PDF_CHART_COLORS[idx % len(PDF_CHART_COLORS)]
        drawing.add(Rect(legend_x, y - 2.4 * mm, 4 * mm, 4 * mm, fillColor=color, strokeColor=color))
        drawing.add(String(
            legend_x + 6 * mm, y - 1.8 * mm,
            f"{label}：{count} 次（{count / total_records * 100:.1f}%）",
            fontName='STSong-Light',
            fontSize=8,
            fillColor=colors.HexColor('#2C3E50'),
        ))

    return drawing


def _build_clinical_metric_cards(metrics, styles):
    cards = [
        ("平均情緒分數", metrics["avg_score"], "-3 至 3"),
        ("低分情緒比例", metrics["low_ratio"], "分數 <= -2"),
        ("正向情緒比例", metrics["positive_ratio"], "分數 >= 2"),
        ("筆記填寫比例", metrics["note_ratio"], "文字脈絡完整度"),
    ]
    card_data = [[
        Paragraph(
            f'<font size="8.5" color="#34495E">{label}</font><br/>'
            f'<font size="15.5" color="#1A5276">{value}</font><br/>'
            f'<font size="8" color="#566573">{hint}</font>',
            styles['metric_card']
        )
        for label, value, hint in cards
    ]]
    card_table = Table(card_data, colWidths=[40.5 * mm] * 4, rowHeights=[24 * mm])
    card_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#F4F8FB')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#AED6F1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#D6EAF8')),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('LEFTPADDING', (0, 0), (-1, -1), 7),
        ('RIGHTPADDING', (0, 0), (-1, -1), 5),
        ('TOPPADDING', (0, 0), (-1, -1), 7),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
    ]))
    return card_table


def _build_score_trend_chart(metrics, styles):
    rows = metrics.get("trend_rows", [])
    if not rows:
        return _build_empty_chart_message(styles)

    dates = [row[0][5:] if len(row[0]) >= 10 else row[0] for row in rows]
    scores = [float(row[1]) for row in rows]

    drawing = Drawing(165 * mm, 62 * mm)
    chart = VerticalBarChart()
    chart.x = 12 * mm
    chart.y = 13 * mm
    chart.width = 138 * mm
    chart.height = 40 * mm
    chart.data = [scores]
    chart.valueAxis.valueMin = -3
    chart.valueAxis.valueMax = 3
    chart.valueAxis.valueStep = 1
    chart.valueAxis.labels.fontName = 'Helvetica'
    chart.valueAxis.labels.fontSize = 7
    chart.categoryAxis.categoryNames = dates
    chart.categoryAxis.labels.fontName = 'Helvetica'
    chart.categoryAxis.labels.fontSize = 6
    chart.categoryAxis.labels.angle = 30
    chart.bars[0].fillColor = colors.HexColor('#2E86C1')
    chart.bars[0].strokeColor = colors.HexColor('#1A5276')
    drawing.add(chart)
    drawing.add(String(
        12 * mm, 55 * mm,
        "每日平均情緒分數（-3 至 3）",
        fontName='STSong-Light',
        fontSize=8,
        fillColor=colors.HexColor('#566573'),
    ))
    return drawing


def _append_clinical_metrics_sections(story, username, metrics, styles, cn):
    _append_pdf_section(story, "資料概況", styles)
    overview_data = [
        ["項目", "內容", "項目", "內容"],
        ["使用者", str(username), "產生日期", datetime.now().strftime('%Y-%m-%d')],
        ["資料期間", metrics["date_range"], f"近 {RECENT_DAYS} 天紀錄數", str(metrics["total_records"])],
        ["最常見主情緒", metrics["most_major"], "最常見細項情緒", metrics["most_sub"]],
    ]
    overview_table = Table(overview_data, colWidths=[32 * mm, 52 * mm, 36 * mm, 45 * mm])
    _style_pdf_table(overview_table, cn, header_color='#34495E', row_colors=[colors.HexColor('#F4F6F7'), colors.white])
    story.append(overview_table)
    story.append(Spacer(1, 8))

    _append_pdf_section(story, "量化指標摘要", styles)
    story.append(_build_clinical_metric_cards(metrics, styles))
    story.append(Paragraph(
        f"最低分紀錄：{html_escape(metrics['min_score_record'])}。建議於會談中溫和追蹤該日脈絡。",
        styles['body']
    ))
    story.append(Spacer(1, 8))


def _append_clinical_trend_sections(story, metrics, styles, cn):
    _append_pdf_section(story, f"情緒趨勢參考（最近 {RECENT_DAYS} 天）", styles)
    story.append(_build_score_trend_chart(metrics, styles))
    story.append(Spacer(1, 4))
    trend_data = [["日期", "平均分數", "主要情緒", "紀錄數"]]
    trend_data.extend(metrics["trend_rows"] or [["-", "-", "-", "0"]])
    trend_table = Table(trend_data, colWidths=[42 * mm, 36 * mm, 55 * mm, 32 * mm])
    _style_pdf_table(trend_table, cn, header_color='#21618C', row_colors=[colors.HexColor('#EBF5FB'), colors.white])
    story.append(trend_table)
    story.append(Spacer(1, 8))

    _append_pdf_section(story, "文字關鍵字（供會談前快速掌握主題）", styles)
    keyword_text = "、".join(f"{word}（{count}）" for word, count in metrics["keywords"]) or "近期待記錄中沒有足夠文字可擷取關鍵字。"
    story.append(Paragraph(html_escape(keyword_text), styles['body']))
    story.append(Spacer(1, 8))


def handle_user_input(user_text, image_obj=None):
    if not user_text and not image_obj: return

    # ==========================================
    # 🛡️ 防護機制：確保變數有初始值 (避免 UnboundLocalError)
    # ==========================================
    past_memory = ""
    img_desc = ""

    # 1. 顯示使用者訊息
    msg_data = {"role": "user", "content": user_text}
    if image_obj: msg_data["image"] = image_obj
    st.session_state.messages.append(msg_data)

    with st.chat_message("user", avatar="🧑‍💻"):
        st.write(user_text)
        if image_obj: st.image(image_obj, width=300)

    # ==========================================
    # 🧠 RAG 核心處理 (先檢索 -> 後寫入)
    # ==========================================
    # A. 如果有照片，先取得摘要描述
    if image_obj:
        with st.spinner("阿光正在把照片畫面刻在腦海裡..."):
            img_desc = summarize_image_for_memory(image_obj)

    # B. 【核心：先檢索】趁本次提問還沒存進去前，先搜尋舊紀錄與教科書
    search_query = user_text if user_text else img_desc
    pdf_knowledge = ""
    if search_query:
        # 使用 ThreadPoolExecutor 並行檢索以節省時間
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_mem = executor.submit(retrieve_memory, search_query, current_user)
            future_pdf = executor.submit(retrieve_pdf_knowledge, search_query)

            past_memory = future_mem.result()
            pdf_knowledge = future_pdf.result()

        if past_memory:
            print(f"🔍 [觸發過往記憶]：{past_memory}")
        # ✅ 優化 6：移除多餘的 else past_memory=""，行 1338 已初始化為 ""，此 else 永遠是 no-op

    # C. 【核心：後寫入】(整合事實萃取) - 與回覆邏輯完全解耦
    mem_to_save = user_text
    if image_obj:
        mem_to_save = f"使用者傳了圖片[{img_desc}]" + (f"，並說：「{user_text}」" if user_text else "")

    if mem_to_save:
        # 💡 呼叫萃取函式，只存事實
        extracted_facts = extract_user_facts(mem_to_save)
        if extracted_facts != "NONE":
            for fact in extracted_facts.split(";"):
                save_to_memory(fact.strip(), current_user)

    # ==========================================
    # 3. 呼叫模型生成回覆 (整合強化指令)
    # ==========================================
    if "chat" in st.session_state:
        # 🎨 跳動三點等待動畫
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div class="aguang-thinking">
            <span>阿光正在思考</span>
            <div class="aguang-dot"></div>
            <div class="aguang-dot"></div>
            <div class="aguang-dot"></div>
        </div>
        """, unsafe_allow_html=True)

        send_parts = []

        if past_memory != "" or pdf_knowledge != "":
            memory_instruction = ""
            if past_memory != "":
                # 💡 引導型記憶指示：讓阿光像老朋友一樣自然地運用記憶，而非強制一次傾倒
                memory_instruction += (
                    f"【阿光的心靈筆記】：\n{past_memory}\n\n"
                    "--- 以上是阿光對這位朋友的了解，僅供參考 ---\n"
                    "提醒：這些筆記不需要一次全說出來。"
                    "請判斷哪一條記憶和使用者現在說的話「最有關聯」，"
                    "若時機自然就輕輕帶入，像朋友聊天一樣（例如：『對了，你之前提到過某件事，那這個你應該會喜歡』）。⚠️ 請只運用上方筆記中確實存在的內容，絕對不可自行腦補或捏造任何細節。"
                    "若筆記內容與本次話題沒有直接語意關係，就完全不必提起。"
                    "如果使用者是在問一般知識、疾病症狀或教科書概念，不要把生活記憶硬塞進回答。\n\n"
                )
            if pdf_knowledge != "":
                memory_instruction += (
                    f"【精神科專業知識參考】（來源：精神科教科書）：\n{pdf_knowledge}\n\n"
                    "--- 以上為教科書內容，阿光可在適當時機自然融入回應中，但請用口語方式說明，不要照本宣科。 ---\n\n"
                )

            # 動態路由與對話階段注入 (含記憶情境)
            current_age = st.session_state.get("user_age_group", "大學生 (19-24)")
            route = classify_psychology_route(user_text, voice_emotion=st.session_state.get("ser_voice_emotion", {}).get("ravdess_emotion", ""))
            
            # 🆕 取得並更新對話階段
            current_stage = st.session_state.get("dialogue_stage", 1)
            new_stage = classify_dialogue_stage(st.session_state.messages, current_stage)
            st.session_state.dialogue_stage = new_stage
            
            # 顯示 UI 指示器
            st.info(f"🧠 阿光心理路由器已啟動：當前採用 **{route}** 模型 📍 階段：Stage {new_stage} ({DIALOGUE_STAGE_NAMES.get(new_stage, '未知')})")
            
            # 組合 Instruction
            stage_instruction = f"\n【當前對話階段：Stage {new_stage}】\n{DIALOGUE_STAGE_RULES.get(new_stage, DIALOGUE_STAGE_RULES[1])}\n"
            route_instruction = f"\n【動態心理路由指示】\n使用者年齡層為：{current_age} (請調整你的語氣深度以適配此年齡層)。\n本次對話請嚴格切換為 {route} 模式，並遵守以下專屬規則：\n{PSYCHOLOGY_ROUTER_RULES.get(route, PSYCHOLOGY_ROUTER_RULES['SFBT (焦點解決)'])}\n"
            
            # ── 🆕 注入人格適配 Prompt ──
            personality_instruction = ""
            user_mbti = st.session_state.get("user_mbti", "")
            if user_mbti:
                traits = [PERSONALITY_PROMPTS[c] for c in user_mbti if c in PERSONALITY_PROMPTS]
                if traits:
                    personality_instruction = f"\n【使用者個人化溝通特質 (MBTI: {user_mbti})】\n" + "\n".join(traits) + "\n"
            
            # ── 🆕 注入臨床風險守護指令 ──
            risk_level = st.session_state.get("clinical_risk", "low")
            risk_instruction = RISK_LEVEL_PROMPTS.get(risk_level, "")
            if risk_instruction:
                risk_instruction = f"\n{risk_instruction}\n"
            
            # 🎤 [SER 注入] 多模態語音情緒感知指令
            ser_ctx = st.session_state.get("ser_voice_emotion", {})
            ser_instruction = ""
            if ser_ctx and ser_ctx.get("ravdess_emotion"):
                v_emo = ser_ctx["ravdess_emotion"]
                v_conf = ser_ctx.get("confidence", 0) * 100
                ser_instruction = (
                    f"\n【🎤 多模態語音情緒感知】\n"
                    f"阿光的耳朵（WavLM 語音情緒辨識模型）偵測到：\n"
                    f"- 聲音情緒：{v_emo}（信心度：{v_conf:.0f}%）\n"
                    f"重要指令：如果使用者的「文字內容」與「聲音情緒」出現矛盾"
                    f"（例如文字說『我沒事』但聲音是 SAD），"
                    f"請優先信任聲音情緒，並溫柔地指出這個矛盾。"
                    f"例如：『雖然你說沒事，但聽你的聲音好像有些沉重，真的還好嗎？』\n"
                )
                # 用完即清，避免文字輸入時殘留語音情緒
                st.session_state["ser_voice_emotion"] = {}

            memory_instruction += stage_instruction + route_instruction + personality_instruction + risk_instruction + ser_instruction + f"\n【使用者現在說】：{user_text}"
            send_parts.append(memory_instruction)
        elif user_text:
            # 動態路由與對話階段注入
            current_age = st.session_state.get("user_age_group", "大學生 (19-24)")
            route = classify_psychology_route(user_text, voice_emotion=st.session_state.get("ser_voice_emotion", {}).get("ravdess_emotion", ""))
            
            # 🆕 取得並更新對話階段
            current_stage = st.session_state.get("dialogue_stage", 1)
            new_stage = classify_dialogue_stage(st.session_state.messages, current_stage)
            st.session_state.dialogue_stage = new_stage
            
            # 顯示 UI 指示器
            st.info(f"🧠 阿光心理路由器已啟動：當前採用 **{route}** 模型 📍 階段：Stage {new_stage} ({DIALOGUE_STAGE_NAMES.get(new_stage, '未知')})")
            
            # 組合 Instruction
            stage_instruction = f"\n【當前對話階段：Stage {new_stage}】\n{DIALOGUE_STAGE_RULES.get(new_stage, DIALOGUE_STAGE_RULES[1])}\n"
            route_instruction = f"\n【動態心理路由指示】\n使用者年齡層為：{current_age} (請調整你的語氣深度以適配此年齡層)。\n本次對話請嚴格切換為 {route} 模式，並遵守以下專屬規則：\n{PSYCHOLOGY_ROUTER_RULES.get(route, PSYCHOLOGY_ROUTER_RULES['SFBT (焦點解決)'])}\n"
            
            # ── 🆕 注入人格適配 Prompt ──
            personality_instruction = ""
            user_mbti = st.session_state.get("user_mbti", "")
            if user_mbti:
                traits = [PERSONALITY_PROMPTS[c] for c in user_mbti if c in PERSONALITY_PROMPTS]
                if traits:
                    personality_instruction = f"\n【使用者個人化溝通特質 (MBTI: {user_mbti})】\n" + "\n".join(traits) + "\n"
            
            # ── 🆕 注入臨床風險守護指令 ──
            risk_level = st.session_state.get("clinical_risk", "low")
            risk_instruction = RISK_LEVEL_PROMPTS.get(risk_level, "")
            if risk_instruction:
                risk_instruction = f"\n{risk_instruction}\n"
            
            # 🎤 [SER 注入] 多模態語音情緒感知指令（無記憶分支）
            ser_ctx = st.session_state.get("ser_voice_emotion", {})
            ser_instruction = ""
            if ser_ctx and ser_ctx.get("ravdess_emotion"):
                v_emo = ser_ctx["ravdess_emotion"]
                v_conf = ser_ctx.get("confidence", 0) * 100
                ser_instruction = (
                    f"\n【🎤 多模態語音情緒感知】\n"
                    f"阿光的耳朵（WavLM 語音情緒辨識模型）偵測到：\n"
                    f"- 聲音情緒：{v_emo}（信心度：{v_conf:.0f}%）\n"
                    f"重要指令：如果使用者的「文字內容」與「聲音情緒」出現矛盾"
                    f"（例如文字說『我沒事』但聲音是 SAD），"
                    f"請優先信任聲音情緒，並溫柔地指出這個矛盾。"
                    f"例如：『雖然你說沒事，但聽你的聲音好像有些沉重，真的還好嗎？』\n"
                )
                st.session_state["ser_voice_emotion"] = {}

            route_instruction = stage_instruction + route_instruction + personality_instruction + risk_instruction + ser_instruction + f"\n【使用者現在說】：{user_text}"
            send_parts.append(route_instruction)


        if image_obj:
            send_parts.append(image_obj)

        # 發送給模型
        if st.session_state.get("engine_mode_radio") == "🖥️ 本地模式 (1.5B)":
            from local_llm_bridge import generate_local_response
            # 過濾掉非文字部分(如圖片)，組成單一字串給本地模型
            combined_user_text = "\\n".join([str(p) for p in send_parts if isinstance(p, str)])
            response = generate_local_response(combined_user_text, st.session_state.messages)
        else:
            response = call_gemini_with_retry(st.session_state.chat.send_message, send_parts)
            
        thinking_placeholder.empty()  # 清除跳動三點

        if response:
            # ✅ 優化 8：用頂層預編譯的 regex 一次替換，避免四次連鎖 .replace() 建立中間字串
            output_text = clean_assistant_text(_STEP_PATTERN.sub("", response.text))
            with st.chat_message("assistant", avatar="🧠"):
                st.write_stream(stream_typewriter_effect(output_text))

            st.session_state.messages.append({"role": "assistant", "content": output_text})

            if st.session_state.get("tts_enabled", True):
                try:
                    tts_text = output_text[:150]
                    audio_bytes = generate_edge_tts_audio(tts_text)
                    st.audio(audio_bytes, format="audio/mp3", autoplay=True)
                except Exception as e:
                    print(f"語音生成失敗: {e}")

def generate_pdf_report(username, df, mode="user"):
    """
    🆕 產生「阿光情緒月報」PDF。
    - mode="user"     → 白話版，給使用者自己看
    - mode="clinical" → 專業版，供醫療人員參考
    - 使用 reportlab 內建 STSong-Light CID 字體，不需要任何外部字體檔
    - 請 Gemini 撰寫三段式摘要後嵌入報告
    - 回傳 bytes，可直接傳給 st.download_button 的 data 參數
    """
    recent_logs = filter_recent_emotion_history(df)  # 取最近 14 天用於摘要與明細
    if recent_logs is None:
        recent_logs = pd.DataFrame(columns=["date", "major", "sub", "note"])
    for required_col in ["date", "major", "sub", "note"]:
        if required_col not in recent_logs.columns:
            recent_logs[required_col] = ""

    # ── Step 1：組合紀錄，請 Gemini 產生臨床摘要 ──────────────────────────
    analysis_input = ""
    for _, row in recent_logs.iterrows():
        analysis_input += (
            f"時間:{row['date']}, "
            f"情緒:{row['major']}({row['sub']}), "
            f"內容:{row['note']}\n"
        )

    clinical_metrics = _compute_clinical_pdf_metrics(recent_logs) if mode == "clinical" else None
    clinical_context = _build_clinical_analysis_context(clinical_metrics) if clinical_metrics else ""

    if mode == "clinical":
        summary_prompt = f"""
你現在是專業心理諮商師「阿光」。請根據使用者「{username}」最近的自填情緒紀錄與量化摘要，撰寫一份正式的「醫療人員參考摘要」。

格式要求：
1. 全程使用繁體中文，語氣專業且溫暖。
2. 內容定位為「會談前參考」，不得做診斷、病名判定，也不要給治療指令。
3. 分為四個段落，每段以「【】」標題開頭：
   【情緒趨勢分析】：描述近期情緒的整體走向與規律（3 至 5 句）。
   【可能壓力源推測】：根據紀錄內容，推測可能造成情緒波動的原因（3 至 5 句）。
   【值得追蹤的觀察】：列出會談中可進一步確認的觀察點（3 至 5 句）。
   【會談建議方向】：提供可用於後續會談的提問方向或支持策略（3 至 5 句）。
4. 避免使用「確診」「患有」「治療方案」「必須」等判定式語氣。
5. 不要加任何前言或後記，直接從【情緒趨勢分析】開始。

以下是量化摘要：
{clinical_context}

以下是使用者的近期紀錄：
{analysis_input}
"""
    else:  # mode == "user"
        summary_prompt = f"""
你現在是「阿光」，一個溫暖的心理陪伴夥伴。
請根據「{username}」最近的情緒紀錄，寫一份輕鬆、溫暖的心情回顧給他/她自己看。

格式要求：
1. 全程使用繁體中文，語氣像朋友在聊天，完全不用專業術語。
2. 分為三個段落，每段以「【】」標題開頭：
   【最近的你】：用白話描述這段時間的情緒起伏，讓使用者感覺被理解（3 至 4 句）。
   【阿光的觀察】：溫和地點出可能讓情緒波動的原因，用生活化的比喻（3 至 4 句）。
   【阿光想說的話】：給使用者鼓勵或一個小小的建議，語氣像朋友打氣（2 至 3 句）。
3. 不要用任何像「臨床」「症狀」「個案」「觸發點」「認知行為」這類的詞。
4. 不要加任何前言或後記，直接從【最近的你】開始。

以下是使用者的近期紀錄：
{analysis_input}
"""

    def _run_summary():
        model = genai.GenerativeModel(SELECTED_MODEL)
        return model.generate_content(summary_prompt)

    ai_res = call_gemini_with_retry(_run_summary)
    ai_summary = ai_res.text if ai_res else "阿光暫時無法產生摘要，建議稍後重新產生。"

    # ── Step 2：計算情緒統計 ─────────────────────────────────────────────
    emotion_counts = recent_logs['major'].value_counts().reset_index()
    emotion_counts.columns = ['情緒類型', '次數']
    total_records = len(recent_logs)

    # ── Step 3：用 reportlab 排版 PDF ────────────────────────────────────
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        leftMargin=20 * mm, rightMargin=20 * mm,
        topMargin=18 * mm, bottomMargin=18 * mm,
        title=f"阿光情緒月報 - {username}",
        author="阿光 心靈小幫手",
    )

    s = _build_pdf_styles()
    cn = s['cn']
    story = []

    # ── 封面標題區 ──
    pdf_title = "阿光幫你整理的心情回顧" if mode == "user" else "阿光的情緒健康月報"
    story.append(Paragraph(pdf_title, s['title']))
    story.append(Paragraph(
        f"使用者：{username}　產生日期：{datetime.now().strftime('%Y 年 %m 月 %d 日')}",
        s['subtitle']
    ))
    story.append(HRFlowable(
        width="100%", thickness=1.5,
        color=colors.HexColor('#2980B9'), spaceAfter=14
    ))
    if mode == "clinical":
        story.append(Paragraph(
            "醫療人員參考版｜根據使用者自填情緒紀錄生成，僅供會談前參考，不構成診斷或治療建議。",
            s['clinical_banner']
        ))

    # ── 免責聲明 ──
    disclaimer_text = (
        "這份回顧由阿光根據你的紀錄整理而成，只是幫你認識自己的心情，不是醫療診斷。"
        "如果你覺得需要更多支持，歡迎尋求諮商師或醫師的協助。"
        if mode == "user" else
        "本報告由 AI 輔助整理使用者自填紀錄，內容僅供醫師或諮商師於會談前參考，"
        "不構成任何醫療診斷、治療建議或法律依據。請搭配專業評估、會談脈絡與必要量表綜合判讀。"
    )
    story.append(Paragraph(disclaimer_text, s['disclaimer']))

    if mode == "clinical":
        _append_clinical_metrics_sections(story, username, clinical_metrics, s, cn)

    # ── 情緒統計表 ──
    stat_title = "情緒分布統計（全部紀錄）" if mode == "user" else f"情緒分布統計（最近 {RECENT_DAYS} 天）"
    story.append(Paragraph(stat_title, s['section']))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor('#AED6F1'), spaceAfter=6
    ))
    if mode == "clinical":
        story.append(_build_emotion_pie_chart(emotion_counts, total_records, s))
        story.append(Spacer(1, 4))

    table_data = [['情緒類型', '出現次數', '佔比']]
    if total_records:
        for _, row in emotion_counts.iterrows():
            pct = f"{row['次數'] / total_records * 100:.1f}%"
            table_data.append([str(row['情緒類型']), str(row['次數']), pct])
        table_data.append(['合計', str(total_records), '100%'])
    else:
        table_data.append(['-', '0', '0%'])
        table_data.append(['合計', '0', '0%'])

    stat_table = Table(table_data, colWidths=[55 * mm, 45 * mm, 45 * mm])
    stat_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), cn),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2980B9')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -2), [colors.HexColor('#EBF5FB'), colors.white]),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor('#D6EAF8')),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#AED6F1')),
        ('ROWHEIGHT', (0, 0), (-1, -1), 18),
    ]))
    story.append(stat_table)
    story.append(Spacer(1, 10))

    # ── 阿光摘要 ──
    summary_title = "阿光想跟你說的話 💬" if mode == "user" else "阿光的臨床觀察摘要（供醫療人員參考）"
    story.append(Paragraph(summary_title, s['section']))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor('#AED6F1'), spaceAfter=6
    ))
    for line in ai_summary.strip().split('\n'):
        line = line.strip()
        if line:
            story.append(Paragraph(line, s['body']))
    story.append(Spacer(1, 10))

    if mode == "clinical":
        _append_clinical_trend_sections(story, clinical_metrics, s, cn)

    # ── 近期明細紀錄表 ──
    story.append(Paragraph(f"近期紀錄明細（最近 {RECENT_DAYS} 天）", s['section']))
    story.append(HRFlowable(
        width="100%", thickness=0.5,
        color=colors.HexColor('#AED6F1'), spaceAfter=6
    ))

    detail_data = [['日期', '主情緒', '細項', '筆記摘要']]
    if recent_logs.empty:
        detail_data.append(['-', '-', '-', '近期待紀錄中沒有資料'])
    else:
        for _, row in recent_logs.iterrows():
            if mode == "clinical":
                note_str = _pdf_note_text(row['note'])
                note_preview = note_str[:120] + ('...' if len(note_str) > 120 else '')
                note_cell = Paragraph(html_escape(note_preview or '未填寫'), s['small_body'])
            else:
                note_str = str(row['note'])
                note_preview = note_str[:22] + ('...' if len(note_str) > 22 else '')
                note_cell = note_preview
            detail_data.append([
                str(row['date'])[:10],
                str(row['major']),
                str(row['sub']),
                note_cell,
            ])

    detail_table = Table(
        detail_data,
        colWidths=[30 * mm, 24 * mm, 24 * mm, 87 * mm]
    )
    detail_style_items = [
        ('FONTNAME', (0, 0), (-1, -1), cn),
        ('FONTSIZE', (0, 0), (-1, -1), 8.5),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1A5276')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('ALIGN', (3, 1), (3, -1), 'LEFT'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#EBF5FB'), colors.white]),
        ('GRID', (0, 0), (-1, -1), 0.4, colors.HexColor('#AED6F1')),
    ]
    if mode == "user":
        detail_style_items.append(('ROWHEIGHT', (0, 0), (-1, -1), 16))
    detail_table.setStyle(TableStyle(detail_style_items))
    story.append(detail_table)

    # ── 頁尾 ──
    story.append(Spacer(1, 14))
    footer_label = "醫療人員參考版｜僅供會談前參考，不作診斷｜" if mode == "clinical" else "本報告"
    story.append(Paragraph(
        f"{footer_label}由「阿光 心靈小幫手」自動產生  {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        s['caption']
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()


# =========================
# 5. Streamlit UI 主邏輯
# =========================
st.title("🧠 心理情緒測驗小幫手")
st.caption("我是阿光，你的 AI 心理夥伴。")

tab1, tab2 = st.tabs(["**💬 找阿光聊聊**", "**📈 我的情緒日記**"])

# --- TAB 2: 聊天 (含圖片與語音功能) ---
with tab1:
    # 🎨 Mood-aware UI：依 Stage 1 的全域 resolver 顯示低干擾提示卡
    if _current_mood == "grounding_stress":
        bg_color = "#F0F8FF"
        border_color = "#88BEE8"
        title_color = "#2F5F78"
        mood_hint = "🌬️ 阿光感覺你可能有點緊繃。左側有呼吸練習入口，可以先陪自己慢慢吸氣、吐氣。"
    elif _current_mood == "quiet_low":
        bg_color = "#F5F3FF"
        border_color = "#B0A0D8"
        title_color = "#555555"
        mood_hint = "🫧 今天如果比較沉，也不用急著變好。阿光會安靜陪你，把心裡的事一點點放下來。"
    elif _current_mood == "calm_positive":
        bg_color = "#F0FFF8"
        border_color = "#7ECBA8"
        title_color = "#2F6F52"
        mood_hint = "🌿 你現在的狀態看起來比較穩。可以把這份平靜留下來，慢慢延續到今天。"
    else:
        bg_color = "#FFF5F0"
        border_color = "#F0A08A"
        title_color = "#6B3020"
        mood_hint = ""

    if mood_hint:
        st.markdown(f"""
        <div class="aguang-mood-hint" style="background:{bg_color};border-left-color:{border_color};color:{title_color};">
            {mood_hint}
        </div>
        """, unsafe_allow_html=True)

    # 🎨 呼吸圓動畫（壓力主題時顯示引導）
    if _current_mood == "grounding_stress":
        st.markdown("""
        <div class="aguang-breathe-wrap">
            <div class="aguang-breathe-circle"></div>
            <div class="aguang-breathe-label">跟著圓圈慢慢呼吸 ✦ 吸氣 2 秒，吐氣 2 秒</div>
        </div>
        """, unsafe_allow_html=True)

    # ==============================================================================
    # 🧩 多問卷分階段評估控制器 (Phased Assessment Flow Controller)
    # ==============================================================================
    assessment_done = st.session_state.get("assessment_complete", False)
    
    if not assessment_done:
        _phase = st.session_state.get("assessment_phase", 1)
        
        # ── Phase 1：簡化 MBTI 4 題 破冰測驗 ──
        if _phase == 1:
            st.markdown(f"""
            <div class="aguang-welcome-card" style="border-left: 4px solid var(--aguang-primary, #4A90E2);">
                <strong>👋 嗨 {current_user}！在開始聊天前，讓阿光先認識你一下 ✨</strong>
                <span>只要 30 秒回答 4 個生活小問題，阿光就能用最懂你、最舒適的方式陪伴你 💬</span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🧩 Step 1：探索你的專屬溝通風格")
            
            mbti_answers = {}
            with st.container(border=True):
                for idx, q in enumerate(MBTI_QUESTIONS):
                    st.markdown(f"**Q{idx+1}. {q['title']}**")
                    st.caption(q["desc"])
                    chosen = st.radio(
                        label=f"mbti_q_{q['dim']}",
                        options=[q["A"], q["B"]],
                        key=f"mbti_radio_{q['dim']}",
                        label_visibility="collapsed"
                    )
                    mbti_answers[q["dim"]] = q["A_val"] if chosen == q["A"] else q["B_val"]
                    if idx < len(MBTI_QUESTIONS) - 1:
                        st.divider()
            
            c_skip, c_next = st.columns([1, 1])
            with c_skip:
                if st.button("⏩ 先跳過，直接開始聊天", key="skip_mbti_btn", use_container_width=True):
                    st.session_state.assessment_complete = True
                    st.session_state.assessment_phase = 1
                    st.rerun()
            with c_next:
                if st.button("✨ 完成，前往下一步 (1/2)", key="submit_mbti_btn", type="primary", use_container_width=True):
                    mbti_type = "".join([mbti_answers[d] for d in ["EI", "SN", "TF", "JP"]])
                    st.session_state.user_mbti = mbti_type
                    st.session_state.assessment_phase = 2
                    st.rerun()
                    
        # ── Phase 2：PHQ-2 + GAD-2 臨床核心快速篩檢 ──
        elif _phase == 2:
            user_mbti = st.session_state.get("user_mbti", "????")
            mbti_name_map = {
                "INFP": "調停者・理想主義", "INFJ": "提倡者・深刻洞察", "ENFP": "競選者・熱情靈感", "ENFJ": "主人公・溫暖引導",
                "INTP": "邏輯學家・深思分析", "INTJ": "建築師・理性規劃", "ENTP": "辯論家・思維敏捷", "ENTJ": "指揮官・果斷領導",
                "ISFP": "探險家・溫柔感性", "ISFJ": "守衛者・細膩守護", "ESFP": "表演者・活力熱忱", "ESFJ": "執政官・熱心關懷",
                "ISTP": "鑑賞家・冷靜實踐", "ISTJ": "物流師・嚴謹可靠", "ESTP": "企業家・果敢行動", "ESTJ": "總經理・井然有序"
            }
            mbti_desc = mbti_name_map.get(user_mbti, "專屬風格")
            
            st.success(f"🎉 辨識完成！你的溝通風格為：**{user_mbti} ({mbti_desc})**。阿光已記下，將以此風格為你調整語氣。")
            
            st.markdown(f"""
            <div class="aguang-welcome-card" style="border-left: 4px solid #F5A623;">
                <strong>📋 Step 2：核心身心狀態快速檢視</strong>
                <span>過去兩週內，你是否曾被以下情況所困擾？（此評估能幫助阿光在必要時給予你最適切的守護）</span>
            </div>
            """, unsafe_allow_html=True)
            
            clinical_scores = {}
            with st.container(border=True):
                for idx, q in enumerate(CLINICAL_QUESTIONS):
                    st.markdown(f"**Q{idx+1}. [{q['scale']}] {q['text']}**")
                    ans = st.radio(
                        label=f"clin_q_{q['id']}",
                        options=CLINICAL_OPTIONS,
                        key=f"clin_radio_{q['id']}",
                        label_visibility="collapsed"
                    )
                    clinical_scores[q["id"]] = CLINICAL_OPTIONS.index(ans)
                    if idx < len(CLINICAL_QUESTIONS) - 1:
                        st.divider()
            
            c_skip2, c_finish = st.columns([1, 1])
            with c_skip2:
                if st.button("⏩ 先跳過此步，開始聊天", key="skip_clinical_btn", use_container_width=True):
                    st.session_state.assessment_complete = True
                    st.rerun()
            with c_finish:
                if st.button("🚀 完成檢測，開啟專屬對話！", key="submit_clinical_btn", type="primary", use_container_width=True):
                    phq_sum = clinical_scores.get("phq1", 0) + clinical_scores.get("phq2", 0)
                    gad_sum = clinical_scores.get("gad1", 0) + clinical_scores.get("gad2", 0)
                    
                    if phq_sum >= 4 or gad_sum >= 4:
                        risk = "high"
                    elif phq_sum >= 3 or gad_sum >= 3:
                        risk = "moderate"
                    else:
                        risk = "low"
                        
                    st.session_state.phq2_score = phq_sum
                    st.session_state.gad2_score = gad_sum
                    st.session_state.clinical_risk = risk
                    st.session_state.assessment_complete = True
                    st.rerun()

    else:
        # ── 評估完成後的頂部狀態卡 ──
        user_mbti = st.session_state.get("user_mbti", "")
        risk_level = st.session_state.get("clinical_risk", "low")
        
        if user_mbti or risk_level != "low":
            risk_badge = {"low": "🟢 心理負荷：平穩", "moderate": "🟡 心理負荷：輕度關注", "high": "🔴 心理負荷：高度守護"}.get(risk_level, "🟢 平穩")
            mbti_badge = f"🧩 風格：{user_mbti}" if user_mbti else "🧩 風格：預設"
            
            col_b1, col_b2 = st.columns([3, 1])
            with col_b1:
                st.caption(f"{mbti_badge} ｜ {risk_badge} ｜ 💡 阿光已開啟個人化適配")
            with col_b2:
                if st.button("🔄 重測問卷", key="retake_assessment_btn", help="重新進行 MBTI 與身心檢測"):
                    st.session_state.assessment_complete = False
                    st.session_state.assessment_phase = 1
                    st.rerun()

        st.markdown(f"""
        <div class="aguang-welcome-card">
            <strong>👋 嗨 {current_user}！我是阿光。</strong>
            <span>這裡只有我們兩個人，你可以放心說出心裡話，也可以慢慢來。</span>
        </div>
        """, unsafe_allow_html=True)

    # 🔒 修正 7：welcome_msg 移到 if 區塊外面先定義
    # 原本只在 "messages" not in session_state 時才定義，導致第二次載入時 NameError
    welcome_msg = f"嗨 {current_user}～今天過得還好嗎？如果心裡有點亂，可以跟我說說。也可以傳照片跟我分享喔！"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.session_state.get("welcome_owner") != current_user:
        previous_welcome = st.session_state.get("welcome_message")
        if previous_welcome:
            st.session_state.messages = [
                msg for msg in st.session_state.messages
                if not (msg.get("role") == "assistant" and msg.get("content") == previous_welcome)
            ]
        st.session_state.messages.insert(0, {"role": "assistant", "content": welcome_msg})
        st.session_state.welcome_owner = current_user
        st.session_state.welcome_message = welcome_msg
        st.session_state.pop("chat", None)
    elif not st.session_state.messages:
        st.session_state.messages.append({"role": "assistant", "content": welcome_msg})

    if "chat" not in st.session_state:
        try:
            history_buffer = []
            for msg in st.session_state.messages:
                if msg.get("content") != st.session_state.get("welcome_message"):
                    role_str = "user" if msg["role"] == "user" else "model"
                    parts = []

                    # 把文字加入 parts
                    if "content" in msg and msg["content"]:
                        parts.append(msg["content"])

                    # 如果這則訊息有圖片，也要把圖片物件加入 parts
                    if "image" in msg and msg["image"]:
                        parts.append(msg["image"])

                    history_buffer.append({"role": role_str, "parts": parts})

            model = genai.GenerativeModel(
                SELECTED_MODEL,
                system_instruction=PSYCHOLOGY_PROMPT
            )
            st.session_state.chat = model.start_chat(history=history_buffer)
        except Exception as e:
            st.error(f"模型載入失敗: {e}")

    for msg in st.session_state.messages:
        avatar = "🧑‍💻" if msg["role"] == "user" else "🧠"
        with st.chat_message(msg["role"], avatar=avatar):
            st.write(msg["content"])
            if "image" in msg and msg["image"]:
                st.image(msg["image"], width=300)

    col1, col2 = st.columns(2)

    with col1:
        with st.expander("📸 傳照片給阿光", expanded=False):
            uploaded_file = st.file_uploader("支援 JPG, PNG 格式", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                st.success("圖片已就緒！請在下方打字或錄音送出。")

    with col2:
        with st.expander("🎙️ 用說的點這裡", expanded=False):
            audio_value = st.audio_input("請按下按鈕開始說話")
            uploaded_audio = st.file_uploader("或上傳錄好的音檔", type=["wav", "mp3", "m4a"])
            uploaded_video = st.file_uploader("🎬 上傳電影片段 (自動擷取音軌)", type=["mp4", "mov", "avi"])

            # 🎬 影片音軌擷取
            if uploaded_video:
                with st.spinner("🎬 正在從影片中擷取音軌..."):
                    import tempfile, os
                    from moviepy import VideoFileClip
                    # 寫入暫存檔讓 moviepy 讀取
                    tmp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                    tmp_video.write(uploaded_video.read())
                    tmp_video.close()
                    tmp_audio = tmp_video.name.replace(".mp4", ".wav")
                    try:
                        clip = VideoFileClip(tmp_video.name)
                        clip.audio.write_audiofile(tmp_audio, fps=16000, logger=None)
                        clip.close()
                        with open(tmp_audio, "rb") as f:
                            uploaded_audio = io.BytesIO(f.read())
                            uploaded_audio.name = "extracted_audio.wav"
                        st.success("✅ 音軌擷取成功！")
                    except Exception as ve:
                        st.error(f"影片音軌擷取失敗: {ve}")
                    finally:
                        os.unlink(tmp_video.name)
                        if os.path.exists(tmp_audio):
                            os.unlink(tmp_audio)

            audio_value = audio_value or uploaded_audio

    img_to_send = None
    if uploaded_file:
        img_to_send = Image.open(uploaded_file)

    if audio_value:
        # 🎨 聲波動畫（錄音完成後、分析前顯示）
        wave_ph = st.empty()
        wave_ph.markdown("""
        <div class="aguang-waveform">
            <p>阿光正在聆聽...</p>
            <div class="aguang-bar"></div><div class="aguang-bar"></div>
            <div class="aguang-bar"></div><div class="aguang-bar"></div>
            <div class="aguang-bar"></div><div class="aguang-bar"></div>
            <div class="aguang-bar"></div>
        </div>
        """, unsafe_allow_html=True)

        audio_bytes = audio_value.read()
        audio_result = analyze_voice_and_emotion(audio_bytes)
        wave_ph.empty()

        transcribed_text = audio_result.get("text", "")
        emotion = audio_result.get("emotion", "未知")
        stress = audio_result.get("stress_level", 0)
        ser_data = audio_result.get("ser_result", {})

        # 🎨 儲存情緒到 session_state，供 mood-aware UI 使用
        st.session_state.last_emotion = emotion
        st.session_state.last_stress = stress
        st.session_state.last_mood_source = "voice"

        # 🎤 [SER 注入] 將語音情緒辨識結果存入 session_state，供 Prompt 組裝層使用
        if ser_data.get("success"):
            st.session_state["ser_voice_emotion"] = {
                "ravdess_emotion": ser_data.get("ravdess_emotion", ""),
                "aguang_emotion": ser_data.get("aguang_emotion", ""),
                "confidence": ser_data.get("confidence", 0),
            }
        else:
            st.session_state["ser_voice_emotion"] = {}

        # 🧠 顯示 SER 本地模型辨識結果
        if ser_data.get("success"):
            ser_emoji = ser_data.get("ravdess_emoji", "")
            ser_label = ser_data.get("ravdess_emotion", "").upper()
            ser_conf = ser_data.get("confidence", 0) * 100
            aguang_label = ser_data.get("aguang_emotion", "")
            probs = ser_data.get("probabilities", {})

            # 機率分布條
            prob_bars = ""
            emoji_map = {"neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
                         "angry": "😡", "fearful": "😨", "disgust": "🤢", "surprised": "😲"}
            for emo, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
                pct = prob * 100
                bar_w = max(pct, 1)
                emo_emoji = emoji_map.get(emo, "")
                prob_bars += (
                    f'<div style="display:flex;align-items:center;gap:6px;margin:3px 0;">'
                    f'<span style="width:90px;font-size:12px;text-align:right;">{emo_emoji} {emo}</span>'
                    f'<div style="flex:1;background:rgba(0,0,0,0.06);border-radius:6px;height:16px;overflow:hidden;">'
                    f'<div style="width:{bar_w}%;height:100%;background:linear-gradient(90deg,var(--aguang-primary),var(--aguang-assistant-border));border-radius:6px;transition:width 0.5s ease;"></div>'
                    f'</div>'
                    f'<span style="width:45px;font-size:11px;color:var(--aguang-muted);">{pct:.1f}%</span>'
                    f'</div>'
                )

            st.markdown(
                f"""
                <div class="emotion-card">
                    <div class="emotion-card-header">
                        <span style="font-size:22px;">{ser_emoji}</span>
                        <span>🧠 本地 AI 語音情緒辨識</span>
                        <span class="emotion-badge badge-default" style="margin-left:auto;">WavLM 模型</span>
                    </div>
                    <div style="margin:10px 0;">
                        <span style="font-size:20px;font-weight:800;">{ser_label}</span>
                        <span style="font-size:14px;color:var(--aguang-muted);margin-left:8px;">→ {aguang_label}</span>
                        <span style="font-size:14px;font-weight:700;margin-left:8px;">信心 {ser_conf:.1f}%</span>
                    </div>
                    <details style="margin-top:6px;">
                        <summary style="cursor:pointer;font-size:12px;color:var(--aguang-muted);">展開情緒機率分布</summary>
                        <div style="margin-top:6px;">{prob_bars}</div>
                    </details>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if "無法使用" in transcribed_text:
            st.error(transcribed_text)
        elif transcribed_text and "無聲" not in transcribed_text:
            st.info(f"🎤 辨識內容：{transcribed_text}")
            st.warning(f"🧠 阿光聽出了你的情緒：**{emotion}**")
            user_input_with_emotion = f"（語音情緒：{emotion}）{transcribed_text}"
            handle_user_input(user_input_with_emotion, image_obj=img_to_send)
        else:
            st.warning("好像沒聽到聲音，請再試一次？")

    tts_enabled = st.toggle("🔊 阿光回覆時要講話", value=True, key="tts_toggle")
    st.session_state["tts_enabled"] = tts_enabled

    if prompt := st.chat_input("輸入你想說的話，或描述一下你傳的照片..."):
        handle_user_input(prompt, image_obj=img_to_send)

# --- TAB 2: 情緒日記 ---
with tab2:
    if current_user == "訪客":
        st.warning("⚠️ 請先在左側邊欄輸入「專屬暱稱」，阿光才能幫你記錄並分析情緒喔！")
    else:
        st.subheader(f"📖 {current_user} 的情緒日記")

        df = get_emotion_history(current_user)

        if df.empty:
            st.info("目前還沒有紀錄喔！快去「情緒測驗」做第一次測驗吧！")
        else:
            recent_df = filter_recent_emotion_history(df)
            # ── 統計摘要卡片 ──────────────────────────────────────
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric(f"近 {RECENT_DAYS} 天測驗次數", f"{len(recent_df)} 次")
                with c2:
                    top_emotion = recent_df['major'].value_counts().idxmax() if not recent_df.empty else "-"
                    st.metric("最常見情緒", top_emotion)
                with c3:
                    recent = recent_df['major'].mode() if not recent_df.empty else pd.Series(dtype=object)
                    st.metric("近期趨勢", recent.iloc[0] if not recent.empty else "-")

            st.divider()

            # ── 情緒走勢折線圖 ──────────────────────────────────
            with st.container(border=True):
                st.markdown("**📈 情緒走勢（時間軸）**")
                scored_df = df.copy()
                scored_df['情緒分數'] = scored_df['major'].map(EMOTION_SCORE).fillna(0)
                trend_df = filter_recent_emotion_history(scored_df)
                if trend_df.empty:
                    st.info(f"最近 {RECENT_DAYS} 天沒有情緒紀錄，暫時無法顯示近期趨勢。")
                else:
                    trend_df['日期'] = pd.to_datetime(trend_df['date']).dt.date
                    trend_by_day = trend_df.groupby('日期')['情緒分數'].mean().reset_index()
                    trend_chart = (
                        alt.Chart(trend_by_day)
                        .mark_line(color="#8B6BBE", strokeWidth=2)
                        .encode(
                            x=alt.X(
                                '日期:T',
                                title=None,
                                axis=alt.Axis(
                                    format='%m月%d日',
                                    tickCount={'interval': 'day', 'step': 1},
                                    labelAngle=0,
                                    labelColor="#5F7894"
                                )
                            ),
                            y=alt.Y(
                                '情緒分數:Q',
                                title=None,
                                scale=alt.Scale(domain=[-3, 3]),
                                axis=alt.Axis(labelColor="#5F7894")
                            ),
                            tooltip=[
                                alt.Tooltip('日期:T', title='日期', format='%m月%d日'),
                                alt.Tooltip('情緒分數:Q', title='平均分數', format='.1f'),
                            ],
                        )
                    )
                    st.altair_chart(trend_chart, use_container_width=True)
                st.caption("分數越高 = 正向情緒，越低 = 負向情緒。每天平均值。")

            st.divider()

            # ── 🆕 Feature 4：情緒熱圖（心情日曆） ──────────────────────
            with st.container(border=True):
                st.markdown("**🗓️ 心情日曆熱圖**")
                st.caption("顏色越深代表當天情緒分數越高（正向），灰色表示無記錄。")

                cal_df = scored_df[['date', '情緒分數']].copy()  # ✅ 優化 4：只複製需要的欄位，節省記憶體
                cal_df['日期'] = pd.to_datetime(cal_df['date'])
                cal_df['date_only'] = cal_df['日期'].dt.date
                day_score = cal_df.groupby('date_only')['情緒分數'].mean()

                if not day_score.empty:
                    min_date = day_score.index.min()
                    max_date = day_score.index.max()

                    # 以週為單位建 grid：從 min_date 所在週一往前對齊（_dt 已在頂層 import）
                    grid_start = min_date - _dt.timedelta(days=min_date.weekday())
                    grid_end = max_date + _dt.timedelta(days=6 - max_date.weekday())

                    weeks = []
                    cur = grid_start
                    while cur <= grid_end:
                        # ✅ 優化 7：用 list comprehension 取代四次 .append() 呼叫
                        week = [cur + _dt.timedelta(days=d) for d in range(7)]
                        weeks.append(week)
                        cur += _dt.timedelta(weeks=1)

                    DOW_LABELS = ["一", "二", "三", "四", "五", "六", "日"]


                    # ✅ 優化 9：移除從未使用的 SCORE_COLORS dict（死代碼）

                    def score_to_color(s):
                        """-3 ~ +3 → 紅到綠漸層"""
                        if s is None:
                            return "#D4DCE8"
                        norm = (s + 3) / 6.0  # 0.0 ~ 1.0
                        r = int(220 - norm * 120)
                        g = int(100 + norm * 130)
                        b = int(100 + norm * 20)
                        return f"rgb({r},{g},{b})"


                    # 建 HTML 表格
                    html = '<div style="overflow-x:auto;">'
                    html += '<table style="border-collapse:separate;border-spacing:4px;font-size:11px;">'
                    # 週幾標題列
                    html += '<tr><td style="width:28px;"></td>'
                    for d in DOW_LABELS:
                        html += f'<td style="text-align:center;color:#888;width:28px;">{d}</td>'
                    html += '</tr>'

                    # 每週一列
                    prev_month = None
                    for week in weeks:
                        html += '<tr>'
                        # 月份標籤（只在月份變化時顯示）
                        # %-m 是 Linux 專用，Windows 用 %#m；統一改用手動去零確保跨平台
                        month_label = f"{week[0].month}月" if week[0].month != prev_month else ""
                        prev_month = week[0].month
                        html += f'<td style="color:#6F7890;font-size:10px;text-align:right;padding-right:5px;font-weight:600;">{month_label}</td>'
                        for day in week:
                            s = day_score.get(day, None)
                            color = score_to_color(s)
                            tip = f"{day} | 分數：{s:.1f}" if s is not None else str(day)
                            html += (
                                f'<td title="{tip}" style="width:24px;height:24px;'
                                f'background:{color};border-radius:5px;border:1px solid {"rgba(96,112,136,0.36)" if s is None else "rgba(255,255,255,0.44)"};box-shadow:{"inset 0 0 0 1px rgba(255,255,255,0.28)" if s is None else "0 1px 2px rgba(48,43,63,0.10)"};opacity:{"0.86" if s is None else "1"};"></td>'
                            )
                        html += '</tr>'
                    html += '</table>'
                    # 圖例
                    empty_color = score_to_color(None)
                    html += '<div style="display:flex;align-items:center;gap:6px;margin-top:10px;font-size:11px;color:#6F7890;flex-wrap:wrap;">'
                    html += f'<span style="display:inline-flex;align-items:center;gap:4px;"><span style="width:18px;height:18px;background:{empty_color};border:1px solid rgba(96,112,136,0.36);border-radius:4px;display:inline-block;"></span>無紀錄</span>'
                    html += '<span style="margin-left:4px;">😔 負向</span>'
                    for v in [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]:
                        c = score_to_color(v)
                        html += f'<div style="width:18px;height:18px;background:{c};border:1px solid rgba(255,255,255,0.50);border-radius:4px;"></div>'
                    html += '<span>😊 正向</span></div>'
                    html += '</div>'

                    st.markdown(html, unsafe_allow_html=True)
                else:
                    st.caption("紀錄還不夠多，繼續使用後熱圖就會出現！")

            st.divider()

            # ── CSS 文字雲 ──────────────────────────────────────
            with st.container(border=True):
                st.markdown("**☁️ 心情關鍵字雲**")
                all_notes = " ".join(df['note'].dropna().astype(str))
                # jieba 預設字典是簡體中文，直接對繁體斷詞準確率偏低。
                # 解法：先用 OpenCC 把繁體轉成簡體 → jieba 斷詞 → 再轉回繁體顯示。
                # ✅ 優化 2：OpenCC 物件改用 @st.cache_resource，讀取轉換表只做一次
                _t2s, _s2t = _get_opencc()

                stopwords = {"的", "了", "是", "我", "你", "他", "她", "在", "有", "和",
                             "就", "都", "也", "很", "不", "一", "個", "這", "那", "但",
                             "嗎", "吧", "啊", "喔", "哦", "嗯", "其實", "因為", "所以",
                             "然後", "覺得", "感覺", "一直", "已經", "還是", "可以", "沒有",
                             "知道", "自己", "什麼", "時候", "為什麼", "這樣", "那樣", ""}

                notes_simplified = _t2s.convert(all_notes)  # 繁→簡
                words = [
                    _s2t.convert(w)  # 斷完再轉回繁體
                    for w in jieba.cut(notes_simplified)
                    if len(w) >= 2 and w.strip() not in stopwords and w.strip().isprintable()
                ]
                freq = Counter(words).most_common(28)

                if freq:
                    max_f = freq[0][1] if freq[0][1] > 0 else 1
                    EMOTION_COLORS = ["#8B6BBE", "#E8927A", "#5BA3D0", "#52C07A", "#E8C07A"]
                    cloud_html = '<div style="line-height:2.4;padding:8px 0;">'
                    for idx, (word, count) in enumerate(freq):
                        size = 13 + int((count / max_f) * 18)
                        color = EMOTION_COLORS[idx % len(EMOTION_COLORS)]
                        opacity = 0.6 + (count / max_f) * 0.4
                        cloud_html += (
                            f'<span style="font-size:{size}px;color:{color};'
                            f'opacity:{opacity:.2f};margin:0 8px;display:inline-block;'
                            f'font-weight:{"500" if count > max_f * 0.5 else "400"};">'
                            f'{word}</span>'
                        )
                    cloud_html += '</div>'
                    st.markdown(cloud_html, unsafe_allow_html=True)
                else:
                    st.caption("紀錄的文字還不夠多，繼續寫日記後關鍵字雲就會出現囉！")

            st.divider()

            st.markdown("**📝 歷史紀錄（卡片）**")
            BADGE = {
                "正向平靜": "badge-開心", "憂鬱低落": "badge-難過", "焦慮煩躁": "badge-生氣",
                # 兼容舊版資料
                "開心": "badge-開心", "難過": "badge-難過", "生氣": "badge-生氣"
            }

            HISTORY_PAGE_SIZE = 5
            history_df = recent_df.iloc[::-1].reset_index(drop=True)
            history_total = len(history_df)
            history_owner = f"{current_user}_{history_total}"

            if st.session_state.get("history_owner") != history_owner:
                st.session_state.history_owner = history_owner
                st.session_state.history_visible_count = HISTORY_PAGE_SIZE

            visible_count = min(
                st.session_state.get("history_visible_count", HISTORY_PAGE_SIZE),
                history_total
            )
            st.session_state.history_visible_count = visible_count

            st.caption(f"顯示最近 {visible_count} / {history_total} 筆紀錄。")

            for _, row in history_df.head(visible_count).iterrows():
                badge_cls = BADGE.get(row['major'], "badge-default")
                date_str = str(row['date'])[:16]
                note_raw = "" if pd.isna(row.get("note")) else str(row.get("note")).strip()
                has_note = bool(note_raw)
                note_preview = note_raw[:34] + ("..." if len(note_raw) > 34 else "")
                note_summary = note_preview if has_note else "未填寫"
                note_icon = "📝 " if has_note else ""
                expander_label = f"{note_icon}{date_str}　{row['major']} · {row['sub']}　— {note_summary}"
                note_state_cls = "aguang-history-has-note" if has_note else "aguang-history-no-note"
                safe_major = html_escape(str(row['major']))
                safe_sub = html_escape(str(row['sub']))
                safe_date = html_escape(date_str)
                safe_note = html_escape(note_raw).replace("\n", "<br>") if has_note else "（未填寫）"
                note_text_cls = "aguang-history-note-text" if has_note else "aguang-history-note-text aguang-history-empty-note"

                with st.expander(expander_label):
                    st.markdown(f"""
                    <div class="emotion-card aguang-history-detail">
                      <span class="aguang-history-marker {note_state_cls}" aria-hidden="true"></span>
                      <div class="emotion-card-header">
                        <span class="emotion-badge {badge_cls}">{safe_major}</span>
                        <span style="color:#7558A8;font-size:13px;">{safe_sub}</span>
                        <span style="margin-left:auto;font-size:12px;color:#999;">{safe_date}</span>
                      </div>
                      <p class="{note_text_cls}">{safe_note}</p>
                    </div>
                    """, unsafe_allow_html=True)

            if history_total > HISTORY_PAGE_SIZE:
                col_more, col_collapse = st.columns([1, 1])
                with col_more:
                    if visible_count < history_total and st.button("顯示更多紀錄", use_container_width=True,
                                                                   key="history_show_more"):
                        st.session_state.history_visible_count = min(visible_count + HISTORY_PAGE_SIZE, history_total)
                        st.rerun()
                with col_collapse:
                    if visible_count > HISTORY_PAGE_SIZE and st.button("收合到最近 5 筆", use_container_width=True,
                                                                       key="history_collapse"):
                        st.session_state.history_visible_count = HISTORY_PAGE_SIZE
                        st.rerun()
            st.divider()
            st.subheader("📤 匯出阿光的情緒報告")
            st.write(
                "選擇你需要的版本：「心情回顧」是寫給你自己看的白話版，"
                "「醫療參考報告」則是給諮商師或醫師參考用的專業版。"
            )

            col_user, col_clinical = st.columns(2)

            with col_user:
                if st.button("💛 產生我的心情回顧", type="primary", use_container_width=True):
                    cache_key = f"pdf_cache_{current_user}_{len(recent_df)}_user"
                    if st.session_state.get(cache_key):
                        st.info("阿光已經幫你整理好了，直接下載就好！")
                    else:
                        with st.spinner("阿光正在幫你整理心情，請稍候..."):
                            try:
                                pdf = generate_pdf_report(current_user, recent_df, mode="user")
                                st.session_state[cache_key] = pdf
                                st.success("✅ 完成！")
                            except Exception as e:
                                st.error(f"產生失敗，請稍後再試。錯誤詳情：{e}")
                cache_key_user = f"pdf_cache_{current_user}_{len(recent_df)}_user"
                if st.session_state.get(cache_key_user):
                    st.download_button(
                        label="📥 下載心情回顧 PDF",
                        data=st.session_state[cache_key_user],
                        file_name=f"阿光心情回顧_{current_user}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

            with col_clinical:
                if st.button("🏥 產生醫療參考報告", type="secondary", use_container_width=True):
                    cache_key = f"pdf_cache_{current_user}_{len(recent_df)}_clinical_{PDF_CLINICAL_CACHE_VERSION}"
                    if st.session_state.get(cache_key):
                        st.info("阿光已經產生過這份報告了，直接下載就好！")
                    else:
                        with st.spinner("阿光正在整理臨床摘要，請稍候..."):
                            try:
                                pdf = generate_pdf_report(current_user, recent_df, mode="clinical")
                                st.session_state[cache_key] = pdf
                                st.success("✅ 完成！")
                            except Exception as e:
                                st.error(f"產生失敗，請稍後再試。錯誤詳情：{e}")
                cache_key_clinical = f"pdf_cache_{current_user}_{len(recent_df)}_clinical_{PDF_CLINICAL_CACHE_VERSION}"
                if st.session_state.get(cache_key_clinical):
                    st.download_button(
                        label="📥 下載醫療參考 PDF",
                        data=st.session_state[cache_key_clinical],
                        file_name=f"阿光臨床報告_{current_user}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True,
                    )

