# app.py
# -*- coding: utf-8 -*-
"""
AI 습관 트래커 (달력 + SQLite + 멀티 API + AI 코치)
- Streamlit
- SQLite 영속 저장 (session_state 의존 X)
- 달력 UI: streamlit-calendar (없으면 자동 폴백)
- 외부 API: OpenWeatherMap(날씨+대기질+일출/일몰), Dog CEO(강아지), Quotable(명언, 키 없음)
- OpenAI: gpt-5-mini로 코치 리포트 + 내일 미션(캘린더 이벤트) 생성

필수 requirements 예시(= Streamlit Cloud에 필요):
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
    # pip install streamlit-calendar
    from streamlit_calendar import calendar
except Exception:
    CALENDAR_AVAILABLE = False


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")
st.title("📊 AI 습관 트래커")
st.caption("체크인 → 자동 컨텍스트(날씨/대기질/일출) → 기록(달력) → 통계 → AI 코치가 내일 미션까지 설계 🧠📅")


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
    db_path = st.text_input("DB 파일 경로", value="habit_tracker.db", help="로컬/클라우드 모두 기본값으로 동작하도록 설계.")
    debug = st.toggle("디버그 모드", value=False, help="API 실패 원인을 화면에 조금 더 보여줍니다.")


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

CITIES = [
    ("Seoul", 37.5665, 126.9780),
    ("Busan", 35.1796, 129.0756),
    ("Incheon", 37.4563, 126.7052),
    ("Daegu", 35.8722, 128.6025),
    ("Daejeon", 36.3504, 127.3845),
    ("Gwangju", 35.1595, 126.8526),
    ("Suwon", 37.2636, 127.0286),
    ("Ulsan", 35.5384, 129.3114),
    ("Sejong", 36.4800, 127.2890),
    ("Jeju", 33.4996, 126.5312),
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
          habits_json TEXT,                     -- {"wake": true, ...}
          notes TEXT,
          weather_json TEXT,                    -- weather payload (compact)
          air_json TEXT,                        -- air quality payload
          dog_json TEXT,                        -- dog payload
          quote_json TEXT,                      -- quote payload
          report_json TEXT,                     -- AI report JSON
          created_at TEXT DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS missions (
          id INTEGER PRIMARY KEY AUTOINCREMENT,
          day TEXT NOT NULL,                    -- YYYY-MM-DD (mission date)
          title TEXT NOT NULL,
          start_at TEXT,                        -- ISO datetime local string
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
            day,
            city,
            coach_style,
            int(mood),
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
    cur = conn.execute("SELECT day, city, coach_style, mood, habits_json, notes, weather_json, air_json, dog_json, quote_json, report_json FROM checkins WHERE day=?",
                       (day,))
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
    """
    '기록(체크인)'이 연속으로 존재한 일수(오늘 포함) 스트릭.
    """
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
def _safe_get(url: str, params: Optional[Dict[str, Any]] = None, timeout: int = 10) -> Optional[Dict[str, Any]]:
    try:
        r = requests.get(url, params=params, timeout=timeout)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


@st.cache_data(show_spinner=False, ttl=60 * 15)
def get_weather_and_sun(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap Current Weather: 한국어, 섭씨 + 일출/일몰 포함
    """
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/weather"
    data = _safe_get(url, params={"q": city, "appid": api_key, "units": "metric", "lang": "kr"}, timeout=10)
    if not data:
        return None

    weather_desc = (data.get("weather") or [{}])[0].get("description")
    main = data.get("main") or {}
    wind = data.get("wind") or {}
    sys = data.get("sys") or {}

    sunrise = sys.get("sunrise")
    sunset = sys.get("sunset")

    # UNIX -> local-ish string (서버 환경에 따라 timezone 다를 수 있어 설명용으로만)
    def fmt_unix(ts: Optional[int]) -> Optional[str]:
        if not ts:
            return None
        try:
            return datetime.fromtimestamp(ts).strftime("%H:%M")
        except Exception:
            return None

    return {
        "city": city,
        "description": weather_desc,
        "temp_c": main.get("temp"),
        "feels_like_c": main.get("feels_like"),
        "humidity": main.get("humidity"),
        "wind_mps": wind.get("speed"),
        "sunrise_hhmm": fmt_unix(sunrise),
        "sunset_hhmm": fmt_unix(sunset),
        "coord": data.get("coord"),  # lat/lon for air
    }


@st.cache_data(show_spinner=False, ttl=60 * 30)
def get_air_quality(lat: float, lon: float, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap Air Pollution API
    """
    if not api_key:
        return None
    url = "https://api.openweathermap.org/data/2.5/air_pollution"
    data = _safe_get(url, params={"lat": lat, "lon": lon, "appid": api_key}, timeout=10)
    if not data:
        return None
    item = (data.get("list") or [{}])[0]
    main = item.get("main") or {}
    comp = item.get("components") or {}

    aqi = main.get("aqi")  # 1~5
    aqi_map = {1: "매우 좋음", 2: "좋음", 3: "보통", 4: "나쁨", 5: "매우 나쁨"}

    return {
        "aqi": aqi,
        "aqi_label": aqi_map.get(aqi, "데이터 없음"),
        "pm2_5": comp.get("pm2_5"),
        "pm10": comp.get("pm10"),
        "o3": comp.get("o3"),
        "no2": comp.get("no2"),
    }


@st.cache_data(show_spinner=False, ttl=60 * 60)
def get_dog_image() -> Optional[Dict[str, Any]]:
    """
    Dog CEO 랜덤 이미지 + 품종 추출
    """
    data = _safe_get("https://dog.ceo/api/breeds/image/random", timeout=10)
    if not data or data.get("status") != "success":
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
    """
    Quotable 랜덤 명언(키 없음)
    """
    data = _safe_get("https://api.quotable.io/random", timeout=10)
    if not data:
        return None
    return {"content": data.get("content"), "author": data.get("author")}


# =========================
# OpenAI: Report + Missions
# =========================
def generate_ai_report(
    *,
    openai_api_key: str,
    coach_style: str,
    payload: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not openai_api_key or not OPENAI_AVAILABLE:
        return None

    client = OpenAI(api_key=openai_api_key)
    sys_style = SYSTEM_PROMPTS.get(coach_style, SYSTEM_PROMPTS["따뜻한 멘토"])

    # Responses API 우선 -> 실패 시 chat.completions 폴백
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


def normalize_habits(raw: Dict[str, Any]) -> Dict[str, bool]:
    out = {}
    for _, hk in HABITS:
        out[hk] = bool(raw.get(hk, False))
    return out


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
# App Boot: DB init
# =========================
conn = db_connect(db_path)
db_init(conn)

today = date.today()
today_s = today.isoformat()

# default load today's record (if exists)
existing_today = load_checkin(conn, today_s)
default_city = existing_today["city"] if existing_today and existing_today.get("city") else "Seoul"
default_style = existing_today["coach_style"] if existing_today and existing_today.get("coach_style") else "따뜻한 멘토"
default_mood = int(existing_today["mood"]) if existing_today and existing_today.get("mood") is not None else 6
default_notes = existing_today["notes"] if existing_today and existing_today.get("notes") else ""
default_habits = normalize_habits(existing_today["habits_json"]) if existing_today and isinstance(existing_today.get("habits_json"), dict) else {hk: False for _, hk in HABITS}

# =========================
# Top tabs
# =========================
tab_checkin, tab_calendar, tab_stats = st.tabs(["✅ 체크인", "📅 달력", "📈 통계/회고"])


# =========================
# Tab 1: Check-in
# =========================
with tab_checkin:
    col_left, col_right = st.columns([1.05, 0.95], gap="large")

    with col_left:
        st.subheader("오늘의 체크인")

        # habits in 2 columns
        c1, c2 = st.columns(2, gap="medium")
        habits: Dict[str, bool] = {}

        for i, (label, hk) in enumerate(HABITS):
            target = c1 if i % 2 == 0 else c2
            with target:
                habits[hk] = st.checkbox(label, value=bool(default_habits.get(hk, False)), key=f"hb_{hk}")

        mood = st.slider("😶‍🌫️ 오늘 기분 점수", 1, 10, value=default_mood, key="mood_slider")

        city = st.selectbox("🌍 도시 선택", [c[0] for c in CITIES], index=[c[0] for c in CITIES].index(default_city) if default_city in [c[0] for c in CITIES] else 0)
        coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, index=COACH_STYLES.index(default_style) if default_style in COACH_STYLES else 1, horizontal=True)

        notes = st.text_area("📝 메모(선택)", value=default_notes, placeholder="예: 오후에 집중이 잘 안 됐음 / 물을 더 마셔야 함", height=90)

        done, total, achievement = habits_summary(habits)

        m1, m2, m3, m4 = st.columns(4, gap="medium")
        m1.metric("달성률", f"{achievement}%")
        m2.metric("달성 습관", f"{done}/{total}")
        m3.metric("기분", f"{mood}/10")
        m4.metric("연속 기록(스트릭)", f"{compute_streak(conn, today_s)}일")

        st.caption("💡 체크인 저장은 '기록'의 본체입니다. 리포트는 그 위에 올라가는 보너스.")
        save_btn = st.button("💾 오늘 체크인 저장", use_container_width=True)

        if save_btn:
            # (1) Weather + Air + Sun
            w = get_weather_and_sun(city, weather_key)
            air = None
            if w and isinstance(w.get("coord"), dict):
                lat = w["coord"].get("lat")
                lon = w["coord"].get("lon")
                if lat is not None and lon is not None:
                    air = get_air_quality(float(lat), float(lon), weather_key)

            # (2) Dog + Quote
            dog = get_dog_image()
            quote = get_quote()

            # Save (no report yet)
            upsert_checkin(
                conn,
                day=today_s,
                city=city,
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
            st.success("오늘 체크인을 저장했어요! 이제 달력/통계에 반영됩니다. 📅")

    with col_right:
        st.subheader("컨텍스트 카드 (자동)")
        st.caption("날씨/대기질/강아지/명언은 체크인과 결합되어 AI가 '행동'으로 바꾸는 재료가 됩니다.")

        # preview cards (use cached live fetch)
        w_preview = get_weather_and_sun(city, weather_key) if weather_key else None
        air_preview = None
        if w_preview and isinstance(w_preview.get("coord"), dict) and weather_key:
            lat = w_preview["coord"].get("lat")
            lon = w_preview["coord"].get("lon")
            if lat is not None and lon is not None:
                air_preview = get_air_quality(float(lat), float(lon), weather_key)

        dog_preview = get_dog_image()
        quote_preview = get_quote()

        card1, card2 = st.columns(2, gap="large")

        with card1:
            st.markdown("#### 🌤️ 날씨")
            if not weather_key:
                st.info("사이드바에 OpenWeatherMap API Key를 넣으면 날씨/대기질이 활성화됩니다.")
            elif not w_preview:
                st.warning("날씨 데이터를 가져오지 못했어요. (도시/키/네트워크 확인)")
                if debug:
                    st.write({"city": city})
            else:
                st.write(f"- **도시:** {w_preview.get('city')}")
                st.write(f"- **상태:** {w_preview.get('description')}")
                st.write(f"- **기온/체감:** {w_preview.get('temp_c')}℃ / {w_preview.get('feels_like_c')}℃")
                st.write(f"- **습도/바람:** {w_preview.get('humidity')}% / {w_preview.get('wind_mps')} m/s")
                st.write(f"- **일출/일몰:** {w_preview.get('sunrise_hhmm')} / {w_preview.get('sunset_hhmm')}")

                st.markdown("#### 🌫️ 대기질")
                if not air_preview:
                    st.write("대기질 데이터 없음")
                else:
                    st.write(f"- **AQI:** {air_preview.get('aqi')} ({air_preview.get('aqi_label')})")
                    st.write(f"- **PM2.5 / PM10:** {air_preview.get('pm2_5')} / {air_preview.get('pm10')}")
                    st.caption(aqi_exercise_hint(air_preview.get("aqi")))

        with card2:
            st.markdown("#### 🐶 강아지 부스터")
            if dog_preview:
                st.write(f"- **품종:** {dog_preview.get('breed')}")
                st.image(dog_preview.get("image_url"), use_container_width=True)
            else:
                st.write("강아지 데이터 없음")

            st.markdown("#### ✨ 오늘의 한 줄 명언")
            if quote_preview and quote_preview.get("content"):
                st.write(f"“{quote_preview.get('content')}”")
                st.caption(f"- {quote_preview.get('author')}")
            else:
                st.write("명언 데이터 없음")

        st.divider()

        st.subheader("🧾 AI 코치 리포트 생성")
        if not OPENAI_AVAILABLE:
            st.error("openai 패키지가 설치되어 있지 않습니다. requirements.txt에 `openai>=1.0.0`를 추가하세요.")
        gen = st.button("⚡ 컨디션 리포트 생성 + 내일 미션(달력 등록)", type="primary", use_container_width=True)

        if gen:
            # Ensure today's record exists (save first with freshest context)
            w = get_weather_and_sun(city, weather_key) if weather_key else None
            air = None
            if w and isinstance(w.get("coord"), dict) and weather_key:
                lat = w["coord"].get("lat")
                lon = w["coord"].get("lon")
                if lat is not None and lon is not None:
                    air = get_air_quality(float(lat), float(lon), weather_key)

            dog = get_dog_image()
            quote = get_quote()

            # Load last 7 days summary to feed AI (pattern!)
            start7 = (today - timedelta(days=6)).isoformat()
            df7 = load_range(conn, start7, today_s)
            week_summary = df7.to_dict(orient="records") if not df7.empty else []

            payload = {
                "date_local": today_s,
                "tomorrow_local": (today + timedelta(days=1)).isoformat(),
                "mood_1_to_10": mood,
                "habits": habits,
                "habit_labels": {hk: label for label, hk in HABITS},
                "notes": notes or "",
                "city": city,
                "weather": w or "데이터 없음",
                "air_quality": air or "데이터 없음",
                "dog": dog or "데이터 없음",
                "quote": quote or "데이터 없음",
                "last_7_days": week_summary,
                "rules": [
                    "운동 미션은 대기질(AQI)이 나쁘면 실내 대체 루틴을 추천",
                    "수면/기상 미션은 일출/일몰과 연결 가능하면 한 문장 코멘트",
                    "미션은 측정 가능(시간/분/양)해야 함",
                ],
            }

            with st.spinner("AI 코치가 분석 중... 🧠"):
                report = generate_ai_report(openai_api_key=openai_key, coach_style=coach_style, payload=payload)

            if not report:
                st.error("리포트 생성 실패: OpenAI Key/모델 접근/네트워크를 확인해 주세요.")
                if debug:
                    st.write({"OPENAI_AVAILABLE": OPENAI_AVAILABLE, "coach_style": coach_style})
            else:
                # Save checkin + report
                upsert_checkin(
                    conn,
                    day=today_s,
                    city=city,
                    coach_style=coach_style,
                    mood=mood,
                    habits=habits,
                    notes=notes,
                    weather=w,
                    air=air,
                    dog=dog,
                    quote=quote,
                    report=report,
                )

                # Save missions to "tomorrow" (AI contract says tomorrow date)
                missions = report.get("tomorrow_missions") or []
                tomorrow_s = (today + timedelta(days=1)).isoformat()
                if isinstance(missions, list) and missions:
                    replace_missions(conn, tomorrow_s, missions)

                st.success("리포트 생성 완료! 내일 미션은 달력에 자동 등록됩니다. 📅⚔️")

                # Pretty render
                st.markdown("---")
                st.markdown("### 🧠 AI 리포트")
                grade = report.get("condition_grade", "—")
                st.metric("컨디션 등급", grade)

                ha = report.get("habit_analysis") or {}
                wins = ha.get("wins") or []
                gaps = ha.get("gaps") or []

                cA, cB = st.columns(2, gap="large")
                with cA:
                    st.markdown("#### ✅ 잘한 점")
                    if wins:
                        for x in wins:
                            st.write(f"- {x}")
                    else:
                        st.write("- 데이터 없음")

                with cB:
                    st.markdown("#### 🧩 개선 포인트")
                    if gaps:
                        for x in gaps:
                            st.write(f"- {x}")
                    else:
                        st.write("- 데이터 없음")

                st.markdown("#### 🌦️ 날씨 코멘트")
                st.write(report.get("weather_comment", "데이터 없음"))

                st.markdown("#### 🎯 내일 미션 (자동 캘린더 등록)")
                if missions:
                    for m in missions:
                        st.write(f"- **{m.get('title')}** · {m.get('when')} · {m.get('duration_min')}분")
                        st.caption(f"성공 기준: {m.get('success_criteria')}")
                else:
                    st.write("미션 데이터 없음")

                st.markdown("#### 🗣️ 오늘의 한마디")
                st.write(report.get("one_liner", "—"))

                # Share text
                st.markdown("### 📋 공유용 텍스트")
                done_labels = [label for (label, hk) in HABITS if habits.get(hk)]
                missed_labels = [label for (label, hk) in HABITS if not habits.get(hk)]

                share_lines = [
                    f"📊 AI 습관 트래커 ({today_s})",
                    f"🎭 코치: {coach_style}",
                    f"🙂 기분: {mood}/10",
                    f"✅ 달성: {done}/{total} ({achievement}%)",
                    f"✅ 완료: {', '.join(done_labels) if done_labels else '없음'}",
                    f"⬜ 미완료: {', '.join(missed_labels) if missed_labels else '없음'}",
                ]
                if w:
                    share_lines.append(f"🌤️ 날씨: {w.get('city')} / {w.get('description')} / {w.get('temp_c')}℃ (체감 {w.get('feels_like_c')}℃)")
                    share_lines.append(f"🌅 일출/일몰: {w.get('sunrise_hhmm')} / {w.get('sunset_hhmm')}")
                if air:
                    share_lines.append(f"🌫️ 대기질: AQI {air.get('aqi')} ({air.get('aqi_label')})")
                if dog:
                    share_lines.append(f"🐶 강아지: {dog.get('breed')}")
                if quote and quote.get("content"):
                    share_lines.append(f"✨ 명언: “{quote.get('content')}” - {quote.get('author')}")

                share_lines.append("")
                share_lines.append("🧾 AI 리포트(요약)")
                share_lines.append(f"[등급] {grade}")
                if report.get("one_liner"):
                    share_lines.append(f"[한마디] {report.get('one_liner')}")

                st.code("\n".join(share_lines), language="text")

    with st.expander("🔎 API/의존성 안내 (중요)"):
        st.markdown(
            """
- OpenAI 리포트를 쓰려면 `openai>=1.0.0` 설치가 필요합니다. (코드에서 `from openai import OpenAI` 사용)
- 달력 UI는 `streamlit-calendar`가 설치되어 있으면 FullCalendar 기반으로 동작합니다. 없으면 앱이 자동으로 폴백 UI를 사용합니다.
- OpenWeatherMap 키가 없으면 날씨/대기질/일출 정보는 비활성화됩니다(기록은 가능).
- 외부 API 호출은 모두 timeout=10, 실패 시 None 처리됩니다.
            """.strip()
        )


# =========================
# Tab 2: Calendar
# =========================
with tab_calendar:
    st.subheader("📅 달력 (기록 + 내일 미션)")

    # Load a reasonable window (past 45 days ~ next 14 days)
    start_day = (today - timedelta(days=45)).isoformat()
    end_day = (today + timedelta(days=14)).isoformat()
    df = load_range(conn, start_day, end_day)
    missions = load_missions(conn, start_day, end_day)

    # Build calendar events
    events: List[Dict[str, Any]] = []

    # Checkin events (all-day) with "achievement" info
    for _, row in df.iterrows():
        day_s = row["day"]
        ach = int(row["achievement"])
        mood_v = row["mood"]
        title = f"체크인 {ach}% · 기분 {mood_v}/10" if mood_v is not None else f"체크인 {ach}%"
        # FullCalendar expects ISO date for all-day: start="YYYY-MM-DD"
        events.append({
            "title": title,
            "start": day_s,
            "allDay": True,
            "extendedProps": {
                "type": "checkin",
                "day": day_s,
                "achievement": ach,
                "mood": mood_v,
            },
        })

    # Mission events
    for m in missions:
        start_at = m.get("start_at") or (m["day"] + "T09:00")
        # compute end if possible
        try:
            dt0 = datetime.fromisoformat(start_at)
            dt1 = dt0 + timedelta(minutes=int(m.get("duration_min") or 10))
            end_at = dt1.isoformat(timespec="minutes")
        except Exception:
            end_at = None

        title = f"🎯 {m.get('title')}"
        events.append({
            "title": title,
            "start": start_at,
            **({"end": end_at} if end_at else {}),
            "allDay": False,
            "extendedProps": {
                "type": "mission",
                "day": m.get("day"),
                "habit_key": m.get("habit_key"),
                "success_criteria": m.get("success_criteria"),
                "source": m.get("source"),
            },
        })

    # Calendar UI
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
            "weekNumbers": False,
            "dayMaxEvents": True,
            "height": 680,
        }

        custom_css = """
        .fc-event { border-radius: 8px; padding: 2px 4px; }
        .fc .fc-toolbar-title { font-weight: 700; }
        """

        state = calendar(
            events=events,
            options=calendar_options,
            custom_css=custom_css,
            key="habit_calendar",
        )

        # Handle clicks
        if isinstance(state, dict) and state.get("callback"):
            cb = state.get("callback")
            st.info(f"달력 이벤트: {cb}")

            # eventClick
            if cb == "eventClick" and state.get("eventClick"):
                ev = state["eventClick"].get("event") or {}
                props = ev.get("extendedProps") or {}
                typ = props.get("type")
                day_clicked = props.get("day") or (ev.get("start") or "")[:10]

                if typ == "checkin" and day_clicked:
                    st.markdown("### 🗂️ 선택한 날짜의 기록")
                    rec = load_checkin(conn, day_clicked)
                    if not rec:
                        st.write("기록 없음")
                    else:
                        habits_json = rec.get("habits_json") if isinstance(rec.get("habits_json"), dict) else {}
                        done, total, ach = habits_summary(habits_json)
                        st.write(f"- 날짜: **{day_clicked}**")
                        st.write(f"- 도시/코치: **{rec.get('city')} / {rec.get('coach_style')}**")
                        st.write(f"- 기분: **{rec.get('mood')}/10**")
                        st.write(f"- 달성: **{done}/{total} ({ach}%)**")
                        st.write(f"- 메모: {rec.get('notes') or '없음'}")

                        # Context
                        w = rec.get("weather_json")
                        air = rec.get("air_json")
                        dog = rec.get("dog_json")
                        quote = rec.get("quote_json")

                        cA, cB = st.columns(2, gap="large")
                        with cA:
                            st.markdown("#### 🌤️ 날씨/일출")
                            if isinstance(w, dict):
                                st.write(f"- {w.get('description')} / {w.get('temp_c')}℃ (체감 {w.get('feels_like_c')}℃)")
                                st.write(f"- 일출/일몰: {w.get('sunrise_hhmm')} / {w.get('sunset_hhmm')}")
                            else:
                                st.write("데이터 없음")

                            st.markdown("#### 🌫️ 대기질")
                            if isinstance(air, dict):
                                st.write(f"- AQI {air.get('aqi')} ({air.get('aqi_label')})")
                                st.write(f"- PM2.5/PM10: {air.get('pm2_5')} / {air.get('pm10')}")
                            else:
                                st.write("데이터 없음")

                        with cB:
                            st.markdown("#### 🐶 강아지")
                            if isinstance(dog, dict) and dog.get("image_url"):
                                st.write(f"- 품종: {dog.get('breed')}")
                                st.image(dog.get("image_url"), use_container_width=True)
                            else:
                                st.write("데이터 없음")

                            st.markdown("#### ✨ 명언")
                            if isinstance(quote, dict) and quote.get("content"):
                                st.write(f"“{quote.get('content')}”")
                                st.caption(f"- {quote.get('author')}")
                            else:
                                st.write("데이터 없음")

                        # AI report
                        rep = rec.get("report_json")
                        st.markdown("#### 🧠 AI 리포트")
                        if isinstance(rep, dict):
                            st.write(f"- 등급: **{rep.get('condition_grade')}**")
                            st.write(f"- 한마디: {rep.get('one_liner')}")
                            st.write(rep)
                        else:
                            st.write("리포트 없음 (체크인만 저장된 상태일 수 있어요)")

                elif typ == "mission":
                    st.markdown("### 🎯 선택한 미션")
                    st.write(f"- 날짜: **{props.get('day') or day_clicked}**")
                    st.write(f"- 습관 키: **{props.get('habit_key')}**")
                    st.write(f"- 성공 기준: {props.get('success_criteria') or '없음'}")
                    st.write(f"- 생성: {props.get('source')}")

            # dateClick (optional): show quick create UI hint
            if cb == "dateClick" and state.get("dateClick"):
                d = state["dateClick"].get("date")  # ISO date
                if d:
                    st.markdown("### 🧷 날짜 클릭")
                    st.write(f"선택한 날짜: **{d[:10]}**")
                    st.caption("해당 날짜로 체크인을 바꾸려면, 상단 '체크인' 탭에서 날짜 선택 기능을 확장해도 좋아요(다음 개선 포인트).")

    else:
        st.warning("달력 컴포넌트(streamlit-calendar)가 설치되어 있지 않아 폴백 UI로 표시합니다.")
        # Fallback: simple date picker + records view
        picked = st.date_input("날짜 선택", value=today)
        picked_s = picked.isoformat()
        rec = load_checkin(conn, picked_s)
        if not rec:
            st.write("기록 없음")
        else:
            st.write(rec)

        st.markdown("#### 설치 안내")
        st.code("pip install streamlit-calendar", language="bash")


# =========================
# Tab 3: Stats / Review
# =========================
with tab_stats:
    st.subheader("📈 통계 / 회고 (최근 7일)")
    start7 = (today - timedelta(days=6)).isoformat()
    df7 = load_range(conn, start7, today_s)

    if df7.empty:
        st.info("아직 저장된 체크인이 없어요. '체크인' 탭에서 오늘 기록을 저장해보세요.")
    else:
        # Chart data
        df7["date_label"] = df7["day"].apply(lambda x: x[5:])  # MM-DD
        chart_df = df7.set_index("date_label")[["achievement"]]
        st.bar_chart(chart_df)

        # Mood line-ish (Streamlit default line chart)
        mood_df = df7.set_index("date_label")[["mood"]].dropna()
        if not mood_df.empty:
            st.line_chart(mood_df)

        # Habit hit counts
        st.markdown("### 🧮 습관별 달성 횟수(7일)")
        # Re-load habits per day for accurate count
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

        habit_count_df = pd.DataFrame(
            [{"habit": label, "count": counts[hk]} for (label, hk) in HABITS]
        ).set_index("habit")
        st.bar_chart(habit_count_df)

        # Insights
        st.markdown("### 🧠 자동 인사이트(규칙 기반)")
        avg_ach = int(round(df7["achievement"].mean()))
        avg_mood = float(df7["mood"].dropna().mean()) if df7["mood"].notna().any() else None
        weakest = min(counts.items(), key=lambda kv: kv[1])[0]
        weakest_label = {hk: label for label, hk in HABITS}.get(weakest, weakest)

        st.write(f"- 최근 7일 평균 달성률: **{avg_ach}%**")
        if avg_mood is not None:
            st.write(f"- 최근 7일 평균 기분: **{avg_mood:.1f}/10**")
        st.write(f"- 가장 자주 빠진 습관: **{weakest_label}**")

        # Streak
        st.write(f"- 현재 연속 기록(스트릭): **{compute_streak(conn, today_s)}일**")

        st.divider()
        st.subheader("🗃️ 데이터 내보내기")
        export_btn = st.button("CSV로 내보내기", use_container_width=True)
        if export_btn:
            df_all = load_range(conn, "2000-01-01", "2100-01-01")
            csv = df_all.to_csv(index=False).encode("utf-8")
            st.download_button("다운로드", data=csv, file_name="habit_tracker_export.csv", mime="text/csv", use_container_width=True)

    with st.expander("🔧 고급 팁 / 다음 확장 아이디어"):
        st.markdown(
            """
- 체크인 날짜를 '오늘' 고정이 아니라, 달력에서 날짜를 눌러 편집하는 UX로 확장 가능
- 미션 완료 체크(미션→습관 반영) 흐름을 추가하면 진짜 '게임화'가 됩니다
- 장기 시계열(30/90/365일) 통계, 요일별 패턴 분석, 월간 리포트 자동 생성도 가능
            """.strip()
        )


# =========================
# Footer
# =========================
st.divider()
with st.expander("📌 문제 해결(에러가 날 때)"):
    st.markdown(
        """
**1) ModuleNotFoundError: openai**
- requirements.txt에 `openai>=1.0.0` 추가하세요. (Streamlit Cloud는 requirements.txt가 설치 기준입니다)

**2) 날씨/대기질이 안 나와요**
- OpenWeatherMap 키가 필요합니다.
- 키/도시명(영문)/무료 플랜 제한을 확인하세요.

**3) 달력이 안 보여요**
- `streamlit-calendar` 설치가 필요합니다.
- 설치가 없으면 앱은 폴백 UI로 동작합니다.

**4) AI 리포트가 실패해요**
- OpenAI Key, 모델 접근 권한, 네트워크를 확인하세요.
- 디버그 모드를 켜면 원인 파악이 조금 더 쉬워집니다.
        """.strip()
    )
