# app.py
# -*- coding: utf-8 -*-
"""
AI 습관 트래커 (달력 + SQLite + 멀티 API + AI 코치) — 달력 색상/범례 포함 완성본

✅ 추가 반영
- 달력 체크인 이벤트를 달성률 기반 색상으로 표시:
  🔴 <40, 🟡 40~79, 🟢 80+
- 달력 상단에 범례(legend) 표시

필수 requirements (Streamlit Cloud):
streamlit
pandas
requests
openai>=1.0.0
streamlit-calendar
"""

import os
import json
import sqlite3
from datetime import datetime, date, timedelta
from typing import Dict, Any, Optional, List, Tuple

import requests
import pandas as pd
import streamlit as st

# -----------------------------
# Optional imports (graceful)
# -----------------------------
OPENAI_AVAILABLE = True
try:
    from openai import OpenAI  # openai>=1.0.0
except Exception:
    OPENAI_AVAILABLE = False

CALENDAR_AVAILABLE = True
try:
    from streamlit_calendar import calendar  # pip install streamlit-calendar
except Exception:
    CALENDAR_AVAILABLE = False


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")
st.caption("체크인 → 컨텍스트(날씨/대기질/일출) → 기록(달력) → 통계 → AI 코치가 내일 미션까지 설계 🧠📅")


# =========================
# UI helpers
# =========================
def status_card(title: str, ok: bool, lines: List[str], kind: str = "info") -> None:
    with st.container(border=True):
        head = f"✅ {title}" if ok else f"⚠️ {title}"
        st.markdown(f"**{head}**")
        for ln in lines:
            st.write(ln)
        if not ok and kind == "error":
            st.caption("원인 메시지는 API 응답을 그대로 보여줍니다(키/권한/요금제/도시명/호출 제한 확인).")


def short_json(obj: Any, max_len: int = 800) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False, indent=2)
        return s if len(s) <= max_len else (s[:max_len] + "\n... (truncated)")
    except Exception:
        return str(obj)


def legend() -> None:
    """달력 상단 범례"""
    with st.container(border=True):
        st.markdown("**🎨 범례(달성률 색상)**")
        c1, c2, c3 = st.columns(3, gap="small")
        with c1:
            st.markdown("🔴 **< 40%**")
        with c2:
            st.markdown("🟡 **40 ~ 79%**")
        with c3:
            st.markdown("🟢 **80%+**")


def color_for_achievement(ach: int) -> Dict[str, str]:
    """
    FullCalendar event colors.
    규칙: 🔴 <40, 🟢 >=80, 🟡 40~79
    """
    if ach < 40:
        return {"backgroundColor": "#ef4444", "borderColor": "#ef4444", "textColor": "#ffffff"}  # red
    if ach >= 80:
        return {"backgroundColor": "#22c55e", "borderColor": "#22c55e", "textColor": "#ffffff"}  # green
    return {"backgroundColor": "#f59e0b", "borderColor": "#f59e0b", "textColor": "#111827"}      # yellow


# =========================
# Sidebar: API Keys / Settings
# =========================
with st.sidebar:
    st.header("🔑 API 설정")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
    )
    weather_key = st.text_input(
        "OpenWeatherMap API Key",
        value=os.getenv("OPENWEATHER_API_KEY", ""),
        type="password",
        placeholder="OpenWeatherMap key",
        help="날씨 + 대기질 + 일출/일몰에 사용합니다.",
    )

    st.divider()
    st.subheader("🧭 앱 옵션")
    db_path = st.text_input("DB 파일 경로", value="habit_tracker.db")
    debug = st.toggle("디버그 모드", value=False, help="실패 시 원인/응답을 더 보여줍니다.")
    st.session_state["debug_mode"] = debug

    if st.button("🔄 API 캐시 새로고침", use_container_width=True):
        try:
            st.cache_data.clear()
            st.success("캐시를 지웠어요. 다시 시도해보세요!")
        except Exception:
            st.warning("캐시 초기화에 실패했어요(환경에 따라 제한될 수 있어요).")


# =========================
# Constants
# =========================
HABITS = [
    ("🌅 기상 미션", "wake"),
    ("💧 물 마시기", "water"),
    ("📚 공부/독서", "study"),
    ("🏃 운동하기", "exercise"),
    ("😴 수면", "sleep"),
]

# 도시: 표시명, OpenWeatherMap query (City,KR)
CITIES = [
    ("서울 (Seoul)", "Seoul,KR"),
    ("부산 (Busan)", "Busan,KR"),
    ("인천 (Incheon)", "Incheon,KR"),
    ("대구 (Daegu)", "Daegu,KR"),
    ("대전 (Daejeon)", "Daejeon,KR"),
    ("광주 (Gwangju)", "Gwangju,KR"),
    ("수원 (Suwon)", "Suwon,KR"),
    ("울산 (Ulsan)", "Ulsan,KR"),
    ("세종 (Sejong)", "Sejong,KR"),
    ("제주 (Jeju City)", "Jeju City,KR"),
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]

SYSTEM_PROMPTS = {
    "스파르타 코치": (
        "너는 엄격한 스파르타 코치다. 군더더기 없이 직설적이고 기준이 높다. "
        "핑계는 바로 잡고, 내일 행동을 명확하게 지시한다. 하지만 인신공격은 금지."
    ),
    "따뜻한 멘토": (
        "너는 따뜻한 멘토다. 공감과 격려를 우선하고, 작은 성취를 잘 포착해 칭찬한다. "
        "부드럽게 개선점을 제안하고, 내일의 작은 실천을 설계한다."
    ),
    "게임 마스터": (
        "너는 RPG 게임 마스터다. 습관을 퀘스트/스탯/던전/보상 같은 게임 문법으로 해석한다. "
        "재미있고 몰입감 있게, 그러나 실천 가능한 미션을 준다."
    ),
}

REPORT_CONTRACT = """
너는 'AI 습관 트래커'의 코치다.
- 과장/단정 금지. 데이터가 없으면 '데이터 없음'이라고 말하고 추측하지 마라.
- 출력은 반드시 JSON 하나만 반환한다(설명 텍스트 금지).
- JSON 스키마:
{
  "condition_grade": "S|A|B|C|D",
  "habit_analysis": {
    "wins": ["...","..."],
    "gaps": ["...","..."]
  },
  "weather_comment": "...",
  "tomorrow_missions": [
    {"title":"...", "when":"YYYY-MM-DDTHH:MM", "duration_min": 10, "check_habit_key":"wake|water|study|exercise|sleep", "success_criteria":"..."},
    {"title":"...", "when":"YYYY-MM-DDTHH:MM", "duration_min": 10, "check_habit_key":"...", "success_criteria":"..."},
    {"title":"...", "when":"YYYY-MM-DDTHH:MM", "duration_min": 10, "check_habit_key":"...", "success_criteria":"..."}
  ],
  "one_liner": "..."
}
- tomorrow_missions는 반드시 3개.
- when은 사용자의 로컬 날짜 기준 '내일' 날짜로 작성한다.
- duration_min은 5~60 사이의 정수.
"""


# =========================
# SQLite (Persistence)
# =========================
def db_connect(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def db_init(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS checkins (
          day TEXT PRIMARY KEY,                 -- YYYY-MM-DD
          city TEXT,
          coach_style TEXT,
          mood INTEGER,
          habits_json TEXT,
          notes TEXT,
          weather_json TEXT,
          air_json TEXT,
          dog_json TEXT,
          quote_json TEXT,
          report_json TEXT,
          created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS missions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          day TEXT NOT NULL,
          title TEXT NOT NULL,
          start_at TEXT,
          duration_min INTEGER,
          habit_key TEXT,
          success_criteria TEXT,
          source TEXT DEFAULT 'ai',
          created_at TEXT DEFAULT (datetime('now')),
          FOREIGN KEY(day) REFERENCES checkins(day) ON DELETE CASCADE
        )
        """
    )
    conn.commit()


def upsert_checkin(
    conn: sqlite3.Connection,
    *,
    day: str,
    city: str,
    coach_style: str,
    mood: int,
    habits: Dict[str, bool],
    notes: str,
    weather: Optional[Dict[str, Any]],
    air: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, Any]],
    quote: Optional[Dict[str, Any]],
    report: Optional[Dict[str, Any]],
) -> None:
    conn.execute(
        """
        INSERT INTO checkins (day, city, coach_style, mood, habits_json, notes, weather_json, air_json, dog_json, quote_json, report_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(day) DO UPDATE SET
          city=excluded.city,
          coach_style=excluded.coach_style,
          mood=excluded.mood,
          habits_json=excluded.habits_json,
          notes=excluded.notes,
          weather_json=excluded.weather_json,
          air_json=excluded.air_json,
          dog_json=excluded.dog_json,
          quote_json=excluded.quote_json,
          report_json=excluded.report_json
        """,
        (
            day, city, coach_style, int(mood),
            json.dumps(habits, ensure_ascii=False),
            notes,
            json.dumps(weather, ensure_ascii=False) if weather else None,
            json.dumps(air, ensure_ascii=False) if air else None,
            json.dumps(dog, ensure_ascii=False) if dog else None,
            json.dumps(quote, ensure_ascii=False) if quote else None,
            json.dumps(report, ensure_ascii=False) if report else None,
        ),
    )
    conn.commit()


def replace_missions(conn: sqlite3.Connection, day: str, missions: List[Dict[str, Any]]) -> None:
    conn.execute("DELETE FROM missions WHERE day = ? AND source='ai'", (day,))
    for m in missions:
        conn.execute(
            """
            INSERT INTO missions (day, title, start_at, duration_min, habit_key, success_criteria, source)
            VALUES (?, ?, ?, ?, ?, ?, 'ai')
            """,
            (
                day,
                m.get("title") or "미션",
                m.get("when"),
                int(m.get("duration_min") or 10),
                m.get("check_habit_key"),
                m.get("success_criteria"),
            ),
        )
    conn.commit()


def load_checkin(conn: sqlite3.Connection, day: str) -> Optional[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT day, city, coach_style, mood, habits_json, notes, weather_json, air_json, dog_json, quote_json, report_json
        FROM checkins WHERE day=?
        """,
        (day,),
    )
    row = cur.fetchone()
    if not row:
        return None
    keys = ["day", "city", "coach_style", "mood", "habits_json", "notes", "weather_json", "air_json", "dog_json", "quote_json", "report_json"]
    data = dict(zip(keys, row))
    for k in ["habits_json", "weather_json", "air_json", "dog_json", "quote_json", "report_json"]:
        if data.get(k):
            try:
                data[k] = json.loads(data[k])
            except Exception:
                pass
    return data


def load_range(conn: sqlite3.Connection, start_day: str, end_day: str) -> pd.DataFrame:
    cur = conn.execute(
        """
        SELECT day, mood, habits_json, city, coach_style
        FROM checkins
        WHERE day BETWEEN ? AND ?
        ORDER BY day ASC
        """,
        (start_day, end_day),
    )
    rows = cur.fetchall()
    if not rows:
        return pd.DataFrame(columns=["day", "mood", "done", "achievement", "city", "coach_style"])
    out = []
    for day_s, mood, habits_json, city, coach_style in rows:
        try:
            habits = json.loads(habits_json) if habits_json else {}
        except Exception:
            habits = {}
        done = sum(1 for _, hk in HABITS if habits.get(hk))
        out.append({
            "day": day_s,
            "mood": int(mood) if mood is not None else None,
            "done": done,
            "achievement": round(done / len(HABITS) * 100) if len(HABITS) else 0,
            "city": city,
            "coach_style": coach_style,
        })
    return pd.DataFrame(out)


def load_missions(conn: sqlite3.Connection, start_day: str, end_day: str) -> List[Dict[str, Any]]:
    cur = conn.execute(
        """
        SELECT day, title, start_at, duration_min, habit_key, success_criteria, source
        FROM missions
        WHERE day BETWEEN ? AND ?
        ORDER BY day ASC, start_at ASC
        """,
        (start_day, end_day),
    )
    rows = cur.fetchall()
    missions = []
    for r in rows:
        missions.append({
            "day": r[0], "title": r[1], "start_at": r[2], "duration_min": r[3],
            "habit_key": r[4], "success_criteria": r[5], "source": r[6]
        })
    return missions


def compute_streak(conn: sqlite3.Connection, until_day: str) -> int:
    d = datetime.fromisoformat(until_day).date()
    streak = 0
    while True:
        day_s = d.isoformat()
        cur = conn.execute("SELECT 1 FROM checkins WHERE day=?", (day_s,))
        if cur.fetchone():
            streak += 1
            d = d - timedelta(days=1)
        else:
            break
    return streak


# =========================
# External APIs (Weather/Air/Sun + Dog + Quote)
# =========================
def safe_get_json(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        content_type = (r.headers.get("content-type") or "").lower()

        data = None
        if "application/json" in content_type:
            try:
                data = r.json()
            except Exception:
                data = None
        else:
            try:
                data = r.json()
            except Exception:
                data = None

        if r.status_code != 200:
            msg = None
            if isinstance(data, dict):
                msg = data.get("message") or data.get("error") or data.get("detail")
            if not msg:
                msg = (r.text or "").strip()[:300] or "요청 실패"
            err = {"status": r.status_code, "message": msg, "url": url, "params": params}
            return None, err

        if data is None:
            err = {"status": r.status_code, "message": "JSON 파싱 실패", "url": url, "params": params}
            return None, err

        return data, None

    except Exception as e:
        err = {"status": None, "message": f"요청 예외: {repr(e)}", "url": url, "params": params}
        return None, err


@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_weather_and_sun(city_query: str, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not api_key:
        return None, {"status": None, "message": "OpenWeatherMap API Key가 비어 있어요.", "url": None, "params": None}

    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"q": city_query, "appid": api_key, "units": "metric", "lang": "kr"}
    data, err = safe_get_json(url, params=params, timeout=10)
    if err or not data:
        return None, err

    weather_desc = (data.get("weather") or [{}])[0].get("description")
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    sys = data.get("sys") or {}

    sunrise = sys.get("sunrise")
    sunset = sys.get("sunset")

    def fmt_unix(ts: Optional[int]) -> Optional[str]:
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).strftime("%H:%M")
        except Exception:
            return None

    compact = {
        "city_query": city_query,
        "name": data.get("name"),
        "description": weather_desc,
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_mps": wind.get("speed"),
        "sunrise_hhmm": fmt_unix(sunrise),
        "sunset_hhmm": fmt_unix(sunset),
        "coord": data.get("coord"),
    }
    return compact, None


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_air_quality(lat: float, lon: float, api_key: str) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    if not api_key:
        return None, {"status": None, "message": "OpenWeatherMap API Key가 비어 있어요.", "url": None, "params": None}

    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": api_key}
    data, err = safe_get_json(url, params=params, timeout=10)
    if err or not data:
        return None, err

    item = (data.get("list") or [{}])[0]
    main = item.get("main") or {}
    comp = item.get("components") or {}

    aqi = main.get("aqi")
    aqi_map = {1: "매우 좋음", 2: "좋음", 3: "보통", 4: "나쁨", 5: "매우 나쁨"}

    compact = {
        "aqi": aqi,
        "aqi_label": aqi_map.get(aqi, "데이터 없음"),
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "o3": comp.get("o3"),
        "no2": comp.get("no2"),
    }
    return compact, None


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_dog_image() -> Optional[Dict[str, Any]]:
    data, err = safe_get_json("https://dog.ceo/api/breeds/image/random", timeout=10)
    if err or not data or data.get("status") != "success":
        return None
    url = data.get("message")
    breed = None
    try:
        parts = url.split("/breeds/")
        if len(parts) > 1:
            breed = parts[1].split("/")[0].replace("-", " ").strip()
    except Exception:
        breed = None
    return {"image_url": url, "breed": breed or "알 수 없음"}


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_quote() -> Optional[Dict[str, Any]]:
    data, err = safe_get_json("https://api.quotable.io/random", timeout=10)
    if err or not data:
        return None
    return {"content": data.get("content"), "author": data.get("author")}


def generate_ai_report(*, openai_api_key: str, coach_style: str, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not openai_api_key or not OPENAI_AVAILABLE:
        return None
    client = OpenAI(api_key=openai_api_key)
    sys_style = SYSTEM_PROMPTS.get(coach_style, SYSTEM_PROMPTS["따뜻한 멘토"])

    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": sys_style},
                {"role": "system", "content": REPORT_CONTRACT.strip()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        )
        text = getattr(resp, "output_text", None)
        if text:
            return json.loads(text)
    except Exception:
        pass

    try:
        chat = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": sys_style},
                {"role": "system", "content": REPORT_CONTRACT.strip()},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
            ],
        )
        text = chat.choices[0].message.content
        if text:
            return json.loads(text)
    except Exception:
        return None

    return None


def habits_summary(habits: Dict[str, bool]) -> Tuple[int, int, int]:
    done = sum(1 for _, hk in HABITS if habits.get(hk))
    total = len(HABITS)
    achievement = round(done / total * 100) if total else 0
    return done, total, achievement


def aqi_exercise_hint(aqi: Optional[int]) -> str:
    if not aqi:
        return "대기질 데이터 없음: 컨디션에 맞춰 운동 강도를 조절해요."
    if aqi >= 4:
        return "대기질이 좋지 않아요: 오늘 운동은 실내/저강도로 바꾸는 게 유리해요."
    if aqi == 3:
        return "대기질 보통: 격한 야외 운동보단 중강도 추천."
    return "대기질 양호: 야외 활동하기 좋은 편이에요."


# =========================
# Boot DB + defaults
# =========================
conn = db_connect(db_path)
db_init(conn)

today = date.today()
today_s = today.isoformat()

existing_today = load_checkin(conn, today_s)
default_city = existing_today["city"] if existing_today and existing_today.get("city") else "서울 (Seoul)"
default_style = existing_today["coach_style"] if existing_today and existing_today.get("coach_style") else "따뜻한 멘토"
default_mood = int(existing_today["mood"]) if existing_today and existing_today.get("mood") is not None else 6
default_notes = existing_today["notes"] if existing_today and existing_today.get("notes") else ""

default_habits = {hk: False for _, hk in HABITS}
if existing_today and isinstance(existing_today.get("habits_json"), dict):
    for _, hk in HABITS:
        default_habits[hk] = bool(existing_today["habits_json"].get(hk, False))


# =========================
# Tabs
# =========================
tab_checkin, tab_calendar, tab_stats = st.tabs(["✅ 체크인", "📅 달력", "📈 통계/회고"])


# =========================
# Tab: Check-in (간단 버전 - 달력/색상 요청의 핵심은 tab_calendar)
# =========================
with tab_checkin:
    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        st.subheader("오늘의 체크인")

        c1, c2 = st.columns(2, gap="medium")
        habits: Dict[str, bool] = {}
        for i, (label, hk) in enumerate(HABITS):
            with (c1 if i % 2 == 0 else c2):
                habits[hk] = st.checkbox(label, value=default_habits.get(hk, False), key=f"hb_{hk}")

        mood = st.slider("😶‍🌫️ 오늘 기분 점수", 1, 10, value=default_mood)

        city_display_list = [c[0] for c in CITIES]
        city_display = st.selectbox(
            "🌍 도시 선택",
            city_display_list,
            index=city_display_list.index(default_city) if default_city in city_display_list else 0,
        )
        city_query = dict(CITIES).get(city_display, "Seoul,KR")

        coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, index=COACH_STYLES.index(default_style) if default_style in COACH_STYLES else 1, horizontal=True)
        notes = st.text_area("📝 메모(선택)", value=default_notes, height=90)

        done, total, achievement = habits_summary(habits)
        m1, m2, m3, m4 = st.columns(4, gap="medium")
        m1.metric("달성률", f"{achievement}%")
        m2.metric("달성 습관", f"{done}/{total}")
        m3.metric("기분", f"{mood}/10")
        m4.metric("연속 기록(스트릭)", f"{compute_streak(conn, today_s)}일")

        if st.button("💾 오늘 체크인 저장", use_container_width=True):
            w, w_err = get_weather_and_sun(city_query, weather_key) if weather_key else (None, None)
            air, air_err = (None, None)
            if w and isinstance(w.get("coord"), dict) and weather_key:
                lat = w["coord"].get("lat")
                lon = w["coord"].get("lon")
                if lat is not None and lon is not None:
                    air, air_err = get_air_quality(float(lat), float(lon), weather_key)

            dog = get_dog_image()
            quote = get_quote()

            upsert_checkin(
                conn,
                day=today_s,
                city=city_display,
                coach_style=coach_style,
                mood=mood,
                habits=habits,
                notes=notes,
                weather=w,
                air=air,
                dog=dog,
                quote=quote,
                report=None,
            )
            st.success("저장 완료! 달력 탭에서 색상으로 확인해보세요. 🎨📅")

            if st.session_state.get("debug_mode") and (w_err or air_err):
                st.warning("디버그: API 오류 상세")
                if w_err:
                    st.code(short_json(w_err), language="json")
                if air_err:
                    st.code(short_json(air_err), language="json")

    with col_right:
        st.subheader("컨텍스트 카드(미리보기)")
        w_preview, w_err = (None, None)
        if weather_key:
            w_preview, w_err = get_weather_and_sun(city_query, weather_key)

        if not weather_key:
            status_card("날씨 비활성화", False, ["- 사이드바에 OpenWeatherMap API Key를 입력하세요."], kind="error")
        elif not w_preview:
            status_card("날씨 실패", False, [f"- status: {(w_err or {}).get('status')}", f"- message: {(w_err or {}).get('message')}"], kind="error")
        else:
            status_card("날씨 정상", True, [f"- {w_preview.get('description')} / {w_preview.get('temp_c')}℃"])


# =========================
# Tab: Calendar (✅ 색상 + ✅ 범례 포함 핵심)
# =========================
with tab_calendar:
    st.subheader("📅 달력 (기록 + 미션)")
    # ✅ 달력 상단 범례
    legend()

    # Load window
    start_day = (today - timedelta(days=45)).isoformat()
    end_day = (today + timedelta(days=14)).isoformat()
    df = load_range(conn, start_day, end_day)
    missions = load_missions(conn, start_day, end_day)

    events: List[Dict[str, Any]] = []

    # ✅ Checkin events with achievement-based color
    for _, row in df.iterrows():
        day_s = row["day"]
        ach = int(row["achievement"])
        mood_v = row["mood"]
        title = f"체크인 {ach}% · 기분 {mood_v}/10" if mood_v is not None else f"체크인 {ach}%"
        colors = color_for_achievement(ach)

        events.append({
            "title": title,
            "start": day_s,
            "allDay": True,
            **colors,
            "extendedProps": {
                "type": "checkin",
                "day": day_s,
                "achievement": ach,
                "mood": mood_v,
            },
        })

    # Mission events (고정색: 보라)
    for m in missions:
        start_at = m.get("start_at") or (m["day"] + "T09:00")
        try:
            dt0 = datetime.fromisoformat(start_at)
            dt1 = dt0 + timedelta(minutes=int(m.get("duration_min") or 10))
            end_at = dt1.isoformat(timespec="minutes")
        except Exception:
            end_at = None

        events.append({
            "title": f"🎯 {m.get('title')}",
            "start": start_at,
            **({"end": end_at} if end_at else {}),
            "allDay": False,
            "backgroundColor": "#8b5cf6",
            "borderColor": "#8b5cf6",
            "textColor": "#ffffff",
            "extendedProps": {
                "type": "mission",
                "day": m.get("day"),
                "habit_key": m.get("habit_key"),
                "success_criteria": m.get("success_criteria"),
                "source": m.get("source"),
            },
        })

    if CALENDAR_AVAILABLE:
        calendar_options = {
            "initialView": "dayGridMonth",
            "headerToolbar": {
                "left": "prev,next today",
                "center": "title",
                "right": "dayGridMonth,timeGridWeek,listWeek,multiMonthYear",
            },
            "selectable": True,
            "editable": False,
            "navLinks": True,
            "dayMaxEvents": True,
            "height": 700,
        }
        custom_css = """
        .fc-event { border-radius: 10px; padding: 2px 6px; }
        .fc .fc-toolbar-title { font-weight: 800; }
        """
        state = calendar(
            events=events,
            options=calendar_options,
            custom_css=custom_css,
            key="habit_calendar",
        )

        if st.session_state.get("debug_mode"):
            st.write("calendar state:", state)

        # Click handling
        if isinstance(state, dict) and state.get("callback") == "eventClick" and state.get("eventClick"):
            ev = state["eventClick"].get("event") or {}
            props = ev.get("extendedProps") or {}
            typ = props.get("type")
            day_clicked = props.get("day") or (ev.get("start") or "")[:10]

            st.markdown("### 🧾 선택한 항목")
            if typ == "checkin" and day_clicked:
                rec = load_checkin(conn, day_clicked)
                if not rec:
                    st.write("기록 없음")
                else:
                    st.write(f"- 날짜: **{day_clicked}**")
                    st.write(f"- 도시/코치: **{rec.get('city')} / {rec.get('coach_style')}**")
                    st.write(f"- 기분: **{rec.get('mood')}/10**")
                    st.write(f"- 메모: {rec.get('notes') or '없음'}")
            elif typ == "mission":
                st.write(f"- 미션 날짜: **{props.get('day') or day_clicked}**")
                st.write(f"- 습관 키: **{props.get('habit_key')}**")
                st.write(f"- 성공 기준: {props.get('success_criteria') or '없음'}")
                st.write(f"- 생성: {props.get('source')}")

    else:
        st.warning("달력 컴포넌트(streamlit-calendar)가 설치되어 있지 않아 폴백 UI로 표시합니다.")
        picked = st.date_input("날짜 선택", value=today)
        rec = load_checkin(conn, picked.isoformat())
        st.write(rec or "기록 없음")
        st.markdown("설치 안내:")
        st.code("pip install streamlit-calendar", language="bash")


# =========================
# Tab: Stats
# =========================
with tab_stats:
    st.subheader("📈 통계 / 회고 (최근 7일)")
    start7 = (today - timedelta(days=6)).isoformat()
    df7 = load_range(conn, start7, today_s)

    if df7.empty:
        st.info("아직 저장된 체크인이 없어요. '체크인' 탭에서 오늘 기록을 저장해보세요.")
    else:
        df7["date_label"] = df7["day"].apply(lambda x: x[5:])
        st.bar_chart(df7.set_index("date_label")[["achievement"]])

        mood_df = df7.set_index("date_label")[["mood"]].dropna()
        if not mood_df.empty:
            st.line_chart(mood_df)

        st.markdown("### 🧮 습관별 달성 횟수(7일)")
        cur = conn.execute(
            "SELECT day, habits_json FROM checkins WHERE day BETWEEN ? AND ? ORDER BY day ASC",
            (start7, today_s),
        )
        rows = cur.fetchall()
        counts = {hk: 0 for _, hk in HABITS}
        for _, hjson in rows:
            try:
                h = json.loads(hjson) if hjson else {}
            except Exception:
                h = {}
            for _, hk in HABITS:
                if h.get(hk):
                    counts[hk] += 1

        habit_count_df = pd.DataFrame([{"habit": label, "count": counts[hk]} for (label, hk) in HABITS]).set_index("habit")
        st.bar_chart(habit_count_df)

        weakest = min(counts.items(), key=lambda kv: kv[1])[0]
        weakest_label = {hk: label for label, hk in HABITS}.get(weakest, weakest)
        st.markdown("### 🧠 인사이트")
        st.write(f"- 최근 7일 평균 달성률: **{int(round(df7['achievement'].mean()))}%**")
        st.write(f"- 현재 연속 기록(스트릭): **{compute_streak(conn, today_s)}일**")
        st.write(f"- 가장 자주 빠진 습관: **{weakest_label}**")
