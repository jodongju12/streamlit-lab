import streamlit as st
from google import genai
from google.genai import types

# ==========================================
# 기본 설정
# ==========================================
MODEL_NAME = "gemini-3.5-flash-lite"

SYSTEM_PROMPT = """
너는 '폴리봇'이야.
인공지능소프트웨어과 1학년의 파이썬 및 AI 공부 도우미 챗봇이야.

규칙 1. 항상 존댓말로 3문장 이내로 답한다.
규칙 2. 파이썬 및 AI 공부와 관련 없는 질문에는
'저는 공부 도우미라서 그 질문에는 답하기 어려워요.'라고 답한다.
규칙 3. 학교 일정처럼 모르는 내용은 지어내지 말고
학과 사무실에 문의하라고 안내한다.
규칙 4. 이름, 학번, 전화번호 같은 개인정보를 묻지 않는다.
"""

# ==========================================
# Gemini 연결
# ==========================================
@st.cache_resource
def get_client():
    return genai.Client(
        api_key=st.secrets["GEMINI_API_KEY"],
        http_options=types.HttpOptions(
            timeout=120_000,
            retry_options=types.HttpRetryOptions(
                http_status_codes=[500, 503]
            ),
        ),
    )


def explain_error(error):
    code = getattr(error, "status_code", None)

    if code == 429:
        return "⏳ 무료 사용 한도를 넘었습니다. 잠시 후 다시 시도하세요."

    if code in (400, 403):
        return "❌ API 키를 확인해 주세요."

    if code == 402:
        return "💳 API 프로젝트의 결제 설정을 확인해 주세요."

    return f"❌ 오류 ({code}): {error}"


# ==========================================
# 대화 기록 처리
# ==========================================
def step_text(step):
    content = step.get("content")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        return "".join(
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        )

    return ""


def trim_history(history, max_turns):
    user_positions = [
        index
        for index, step in enumerate(history)
        if step["type"] == "user_input"
    ]

    if len(user_positions) <= max_turns:
        return history

    return history[user_positions[-max_turns]:]


# ==========================================
# API 키 확인
# ==========================================
try:
    client = get_client()

except Exception:
    st.error(
        "🔑 GEMINI_API_KEY를 찾지 못했습니다. "
        "Streamlit의 Manage app → Settings → Secrets를 확인하세요."
    )
    st.stop()


# ==========================================
# 화면 구성
# ==========================================
st.set_page_config(
    page_title="폴리봇 AI 공부 도우미",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 폴리봇")
st.subheader("인공지능소프트웨어과 AI 공부 도우미")

st.caption(
    "파이썬과 인공지능 공부를 도와주는 AI 챗봇입니다."
)

st.info(
    "💡 파이썬, 머신러닝, 딥러닝, 프로그래밍에 대해 질문해 보세요!"
)


# ==========================================
# 대화 기록 초기화
# ==========================================
if "history" not in st.session_state:
    st.session_state.history = []


# ==========================================
# 사이드바 설정
# ==========================================
with st.sidebar:
    st.header("⚙️ 챗봇 설정")

    max_turns = st.slider(
        "기억할 대화 턴 수",
        min_value=1,
        max_value=10,
        value=6
    )

    if st.button("🧹 새 대화 시작", use_container_width=True):
        st.session_state.history = []
        st.rerun()

    turns = sum(
        1 for step in st.session_state.history
        if step["type"] == "user_input"
    )

    st.metric(
        "기억 중인 대화",
        f"{turns} / {max_turns}"
    )

    st.divider()

    st.caption("모델: " + MODEL_NAME)
    st.caption("개인정보는 입력하지 마세요.")


# ==========================================
# 이전 대화 화면에 표시
# ==========================================
for step in st.session_state.history:

    if step["type"] == "user_input":
        with st.chat_message("user"):
            st.markdown(step_text(step))

    elif step["type"] == "model_output":
        with st.chat_message("assistant"):
            st.markdown(step_text(step))


# ==========================================
# 사용자 메시지 입력 및 AI 응답
# ==========================================
if message := st.chat_input("폴리봇에게 질문해 보세요!"):

    with st.chat_message("user"):
        st.markdown(message)

    send_list = trim_history(
        st.session_state.history + [
            {
                "type": "user_input",
                "content": [
                    {
                        "type": "text",
                        "text": message
                    }
                ]
            }
        ],
        max_turns
    )

    with st.chat_message("assistant"):

        with st.spinner("🤖 폴리봇이 답변을 작성하고 있습니다..."):

            try:
                reply = client.interactions.create(
                    model=MODEL_NAME,
                    store=False,
                    system_instruction=SYSTEM_PROMPT,
                    input=send_list
                )

            except Exception as error:
                st.error("Gemini 요청 중 오류가 발생했습니다.")
                st.code(f"{type(error).__name__}: {error}")

            else:
                st.markdown(reply.output_text)

                st.session_state.history = (
                    send_list
                    + [
                        step.model_dump()
                        for step in reply.steps
                    ]
                )

                st.rerun()
