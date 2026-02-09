# app.py
import os
import time
import json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List

import requests
import pandas as pd
import streamlit as st

# OpenAI SDK (python)
# pip install openai
from openai import OpenAI


# =========================
# Page Config
# =========================
st.set_page_config(page_title="AI 습관 트래커", page_icon="📊", layout="wide")

st.title("📊 AI 습관 트래커")
st.caption("오늘의 습관 + 기분 + 날씨 + 강아지 = AI 코치 컨디션 리포트 🧠🐶🌤️")


# =========================
# Sidebar: API Keys
# =========================
with st.sidebar:
    st.header("🔑 API 설정")

    openai_key = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        placeholder="sk-...",
        help="환경변수 OPENAI_API_KEY가 있으면 자동으로 채워집니다.",
    )
    weather_key = st.text_input(
        "OpenWeatherMap API Key",
        value=os.getenv("OPENWEATHER_API_KEY", ""),
        type="password",
        placeholder="OpenWeatherMap key",
        help="환경변수 OPENWEATHER_API_KEY가 있으면 자동으로 채워집니다.",
    )

    st.divider()
    st.caption("키는 세션에만 사용되며, 이 앱은 저장소에 키를 기록하지 않도록 설계되어 있습니다.")


# =========================
# Helpers: APIs
# =========================
def get_weather(city: str, api_key: str) -> Optional[Dict[str, Any]]:
    """
    OpenWeatherMap에서 현재 날씨 가져오기 (한국어, 섭씨).
    실패 시 None 반환. timeout=10
    """
    if not api_key:
        return None

    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": city,
            "appid": api_key,
            "units": "metric",   # 섭씨
            "lang": "kr",        # 한국어 설명
        }
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()

        # 핵심 정보만 정리
        weather_desc = (data.get("weather") or [{}])[0].get("description")
        temp = (data.get("main") or {}).get("temp")
        feels_like = (data.get("main") or {}).get("feels_like")
        humidity = (data.get("main") or {}).get("humidity")
        wind = (data.get("wind") or {}).get("speed")

        return {
            "city": city,
            "description": weather_desc,
            "temp_c": temp,
            "feels_like_c": feels_like,
            "humidity": humidity,
            "wind_mps": wind,
            "raw": data,
        }
    except Exception:
        return None


def _breed_from_dog_url(url: str) -> Optional[str]:
    """
    Dog CEO 이미지 URL에서 품종(breed) 추출 시도.
    예) https://images.dog.ceo/breeds/hound-afghan/n02088094_1003.jpg -> hound-afghan
    """
    try:
        parts = url.split("/breeds/")
        if len(parts) < 2:
            return None
        tail = parts[1]
        breed = tail.split("/")[0]
        return breed.replace("-", " ").strip() if breed else None
    except Exception:
        return None


def get_dog_image() -> Optional[Dict[str, Any]]:
    """
    Dog CEO에서 랜덤 강아지 사진 URL과 품종 가져오기.
    실패 시 None 반환. timeout=10
    """
    try:
        url = "https://dog.ceo/api/breeds/image/random"
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return None
        data = r.json()
        if data.get("status") != "success":
            return None

        img_url = data.get("message")
        if not img_url:
            return None

        breed = _breed_from_dog_url(img_url)
        return {"image_url": img_url, "breed": breed}
    except Exception:
        return None


# =========================
# AI Coach
# =========================
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

REPORT_FORMAT_GUIDE = """출력은 반드시 아래 섹션을 지켜라(순서 유지). 각 섹션 제목은 그대로 쓰고, 내용은 한국어로 작성.

[컨디션 등급] S/A/B/C/D 중 하나
[습관 분석] (잘한 점 2~3개 + 아쉬운 점 1~2개, 구체적으로)
[날씨 코멘트] (오늘 날씨/체감과 컨디션 연결, 과장 금지)
[내일 미션] 체크박스 습관을 기반으로 3가지 미션(각각 매우 구체적/측정 가능)
[오늘의 한마디] 한 줄 (코치 스타일 반영)
"""


def generate_report(
    *,
    openai_api_key: str,
    coach_style: str,
    habits: Dict[str, bool],
    mood: int,
    weather: Optional[Dict[str, Any]],
    dog: Optional[Dict[str, Any]],
) -> Optional[str]:
    """
    습관+기분+날씨+강아지 품종을 모아서 OpenAI에 전달.
    모델: gpt-5-mini
    실패 시 None 반환.
    """
    if not openai_api_key:
        return None

    client = OpenAI(api_key=openai_api_key)

    habit_done = [k for k, v in habits.items() if v]
    habit_miss = [k for k, v in habits.items() if not v]

    weather_brief = None
    if weather:
        weather_brief = {
            "도시": weather.get("city"),
            "설명": weather.get("description"),
            "기온(℃)": weather.get("temp_c"),
            "체감(℃)": weather.get("feels_like_c"),
            "습도(%)": weather.get("humidity"),
            "바람(m/s)": weather.get("wind_mps"),
        }

    dog_brief = None
    if dog:
        dog_brief = {
            "품종": dog.get("breed") or "알 수 없음",
            "이미지": dog.get("image_url"),
        }

    user_payload = {
        "date_local": datetime.now().strftime("%Y-%m-%d"),
        "mood_1_to_10": mood,
        "habits_done": habit_done,
        "habits_missed": habit_miss,
        "weather": weather_brief,
        "dog": dog_brief,
        "notes": "과장/단정 금지. 데이터가 없으면 '데이터 없음'으로 처리하고 추측하지 말 것.",
    }

    system_prompt = SYSTEM_PROMPTS.get(coach_style, SYSTEM_PROMPTS["따뜻한 멘토"])

    # Responses API 우선 사용(최신 SDK). 실패하면 Chat Completions로 폴백.
    try:
        resp = client.responses.create(
            model="gpt-5-mini",
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": REPORT_FORMAT_GUIDE},
                {
                    "role": "user",
                    "content": "아래 JSON을 기반으로 오늘의 리포트를 작성해줘.\n\n"
                               + json.dumps(user_payload, ensure_ascii=False, indent=2),
                },
            ],
        )
        text = getattr(resp, "output_text", None)
        if text and text.strip():
            return text.strip()
    except Exception:
        pass

    try:
        chat = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "system", "content": REPORT_FORMAT_GUIDE},
                {
                    "role": "user",
                    "content": "아래 JSON을 기반으로 오늘의 리포트를 작성해줘.\n\n"
                               + json.dumps(user_payload, ensure_ascii=False, indent=2),
                },
            ],
        )
        text = chat.choices[0].message.content
        return text.strip() if text else None
    except Exception:
        return None


# =========================
# Session State: 7-day data
# =========================
HABIT_KEYS = [
    ("🌅 기상 미션", "wake"),
    ("💧 물 마시기", "water"),
    ("📚 공부/독서", "study"),
    ("🏃 운동하기", "exercise"),
    ("😴 수면", "sleep"),
]

CITIES = [
    "Seoul", "Busan", "Incheon", "Daegu", "Daejeon",
    "Gwangju", "Suwon", "Ulsan", "Sejong", "Jeju",
]

COACH_STYLES = ["스파르타 코치", "따뜻한 멘토", "게임 마스터"]


def _init_sample_history():
    # 데모용 6일 샘플 + 오늘은 UI로 저장
    today = datetime.now().date()
    days = [today - timedelta(days=i) for i in range(6, 0, -1)]  # 6일 전 ~ 1일 전

    # 적당히 변화 있게 샘플 생성(고정)
    samples = []
    pattern = [
        (3, 6),
        (4, 7),
        (2, 5),
        (5, 8),
        (3, 6),
        (4, 7),
    ]  # (done_count, mood)
    for d, (done_count, mood) in zip(days, pattern):
        row = {"date": d.strftime("%m/%d"), "mood": mood}
        for idx, (_, key) in enumerate(HABIT_KEYS):
            row[key] = 1 if idx < done_count else 0
        samples.append(row)
    return samples


if "history" not in st.session_state:
    st.session_state.history = _init_sample_history()  # 최근 6일
if "today_saved" not in st.session_state:
    st.session_state.today_saved = False


def save_today_record(habits_bool: Dict[str, bool], mood: int):
    today_label = datetime.now().date().strftime("%m/%d")
    row = {"date": today_label, "mood": int(mood)}
    for _, key in HABIT_KEYS:
        row[key] = 1 if habits_bool.get(key, False) else 0

    # 이미 오늘 데이터가 있으면 교체
    history = st.session_state.history[:]
    idx = next((i for i, r in enumerate(history) if r.get("date") == today_label), None)
    if idx is None:
        history.append(row)
    else:
        history[idx] = row

    # 7일 유지(마지막 7개)
    history = history[-7:]
    st.session_state.history = history
    st.session_state.today_saved = True


# =========================
# UI: Check-in
# =========================
left, right = st.columns([1.1, 0.9], gap="large")

with left:
    st.subheader("✅ 오늘의 체크인")

    col_a, col_b = st.columns(2, gap="medium")

    habit_values: Dict[str, bool] = {}
    for i, (label, key) in enumerate(HABIT_KEYS):
        target_col = col_a if i % 2 == 0 else col_b
        with target_col:
            habit_values[key] = st.checkbox(label, value=False)

    mood = st.slider("😶‍🌫️ 오늘 기분 점수", min_value=1, max_value=10, value=6)

    c1, c2 = st.columns([1, 1], gap="medium")
    with c1:
        city = st.selectbox("🌍 도시 선택", CITIES, index=0)
    with c2:
        coach_style = st.radio("🎭 코치 스타일", COACH_STYLES, index=1, horizontal=True)

    # 달성률 계산
    done_count = sum(1 for v in habit_values.values() if v)
    total = len(HABIT_KEYS)
    achievement = round((done_count / total) * 100)

    st.divider()

    # Metrics (3개)
    m1, m2, m3 = st.columns(3, gap="medium")
    m1.metric("달성률", f"{achievement}%")
    m2.metric("달성 습관", f"{done_count}/{total}")
    m3.metric("기분", f"{mood}/10")

    # 오늘 기록 저장
    save_col1, save_col2 = st.columns([1, 2], gap="small")
    with save_col1:
        if st.button("💾 오늘 기록 저장", use_container_width=True):
            save_today_record(habit_values, mood)
            st.success("오늘 기록을 저장했어요! (7일 차트에 반영)")

    with save_col2:
        if not st.session_state.today_saved:
            st.info("체크 후 **오늘 기록 저장**을 누르면 7일 차트에 오늘 데이터가 들어가요.")


with right:
    st.subheader("📈 7일 달성 추이")

    # 6일 샘플 + 오늘(저장된 경우 포함)으로 7일 바 차트
    df = pd.DataFrame(st.session_state.history)

    # 오늘 데이터가 아직 저장 안 되었으면, 미리보기로 오늘 값을 덧붙여 보여주기(차트에 포함)
    today_label = datetime.now().date().strftime("%m/%d")
    if today_label not in df["date"].astype(str).tolist():
        preview_row = {"date": today_label, "mood": int(mood)}
        for _, key in HABIT_KEYS:
            preview_row[key] = 1 if habit_values.get(key, False) else 0
        df = pd.concat([df, pd.DataFrame([preview_row])], ignore_index=True)
        df = df.tail(7)

    df["done"] = df[[k for _, k in HABIT_KEYS]].sum(axis=1)
    df["achievement"] = (df["done"] / len(HABIT_KEYS) * 100).round(0).astype(int)

    chart_df = df.set_index("date")[["achievement"]]
    st.bar_chart(chart_df)

    st.caption("※ 데모용 과거 6일 샘플 + 오늘(저장 전이면 미리보기)로 구성됩니다.")


# =========================
# Weather + Dog + Report
# =========================
st.divider()
st.subheader("🧾 AI 코치 컨디션 리포트")

gen = st.button("컨디션 리포트 생성", type="primary", use_container_width=True)

weather_data = None
dog_data = None
report_text = None

if gen:
    # 기록도 같이 저장해두면 사용성이 좋아서 자동 저장
    save_today_record(habit_values, mood)

    with st.spinner("날씨와 강아지를 소환 중... 🧙‍♂️"):
        weather_data = get_weather(city, weather_key)
        dog_data = get_dog_image()

    with st.spinner("AI 코치가 분석 중... 🧠"):
        report_text = generate_report(
            openai_api_key=openai_key,
            coach_style=coach_style,
            habits=habit_values,
            mood=mood,
            weather=weather_data,
            dog=dog_data,
        )

    if report_text is None:
        st.error(
            "리포트 생성에 실패했어요. "
            "OpenAI API Key가 올바른지/모델 접근 권한이 있는지, 네트워크 상태를 확인해 주세요."
        )

# 결과 표시(버튼 눌렀을 때만)
if gen:
    # 2열 카드: 날씨 / 강아지
    c_weather, c_dog = st.columns(2, gap="large")

    with c_weather:
        st.markdown("#### 🌤️ 오늘의 날씨")
        if weather_data is None:
            st.warning("날씨 데이터를 가져오지 못했어요. (API Key/도시/네트워크 확인)")
        else:
            st.metric("기온(℃)", f"{weather_data.get('temp_c', '—')}")
            st.write(f"- **도시:** {weather_data.get('city')}")
            st.write(f"- **상태:** {weather_data.get('description')}")
            st.write(f"- **체감:** {weather_data.get('feels_like_c')}℃")
            st.write(f"- **습도:** {weather_data.get('humidity')}%")
            st.write(f"- **바람:** {weather_data.get('wind_mps')} m/s")

    with c_dog:
        st.markdown("#### 🐶 오늘의 강아지 부스터")
        if dog_data is None:
            st.warning("강아지 이미지를 가져오지 못했어요. (네트워크 확인)")
        else:
            breed = dog_data.get("breed") or "알 수 없음"
            st.write(f"- **품종:** {breed}")
            st.image(dog_data.get("image_url"), use_container_width=True)

    st.markdown("#### 🧠 AI 리포트")
    if report_text:
        st.markdown(report_text)

    # 공유용 텍스트
    st.markdown("#### 📋 공유용 텍스트")
    done_labels = [label for (label, key) in HABIT_KEYS if habit_values.get(key)]
    missed_labels = [label for (label, key) in HABIT_KEYS if not habit_values.get(key)]
    share_lines = [
        f"📊 AI 습관 트래커 ({datetime.now().strftime('%Y-%m-%d')})",
        f"🎭 코치: {coach_style}",
        f"🙂 기분: {mood}/10",
        f"✅ 달성: {len(done_labels)}/{len(HABIT_KEYS)} ({round(len(done_labels)/len(HABIT_KEYS)*100)}%)",
        f"✅ 완료: " + (", ".join(done_labels) if done_labels else "없음"),
        f"⬜ 미완료: " + (", ".join(missed_labels) if missed_labels else "없음"),
    ]
    if weather_data:
        share_lines.append(
            f"🌤️ 날씨: {weather_data.get('city')} / {weather_data.get('description')} / {weather_data.get('temp_c')}℃"
        )
    if dog_data:
        share_lines.append(f"🐶 강아지: {dog_data.get('breed') or '알 수 없음'}")

    if report_text:
        share_lines.append("")
        share_lines.append("🧾 AI 코치 리포트")
        share_lines.append(report_text.strip())

    st.code("\n".join(share_lines), language="text")


# =========================
# Footer: API 안내
# =========================
with st.expander("🔎 API 안내 / 문제 해결"):
    st.markdown(
        """
- **OpenAI API Key**: OpenAI 플랫폼에서 발급한 키를 넣어주세요.
- **OpenWeatherMap API Key**: OpenWeatherMap에서 발급한 키를 넣어주세요. (현재날씨 API 사용)
- **Dog CEO API**: 키 없이 사용 가능한 공개 API입니다.

**자주 겪는 이슈**
- 날씨가 안 뜸: OpenWeatherMap 키가 비었거나, 도시명(영문)이 맞지 않거나, 무료 플랜 제한일 수 있어요.
- 리포트가 안 뜸: OpenAI 키가 비었거나, `gpt-5-mini` 모델 접근 권한이 없거나, 네트워크 문제일 수 있어요.
- 시간 초과/실패: 외부 API는 `timeout=10`으로 설정되어 있어요. 네트워크 상태에 따라 실패 시 `None` 처리됩니다.
        """.strip()
    )
