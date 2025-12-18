import streamlit as st
import cv2
import time
from deepface import DeepFace
import matplotlib.pyplot as plt
from collections import deque

# -----------------------------
# CONFIG
# -----------------------------
EMOTIONS = ["angry", "happy", "sad", "surprise", "neutral"]
emotion_history = deque(maxlen=20)
ANALYZE_EVERY = 2.0  # seconds

CHATBOT_RESPONSES = {
    "happy": "You look happy 😄 Keep spreading positivity!",
    "sad": "I’m here for you ❤️ Take a short break.",
    "angry": "Try deep breathing 🌿 It helps calm the mind.",
    "surprise": "Something interesting happened 👀",
    "neutral": "All good 😌 Stay focused."
}

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(
    page_title="Emotion Analytics Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    "<h1 style='text-align:center;'>😊 Emotion Analytics Dashboard</h1>",
    unsafe_allow_html=True
)

st.sidebar.title("⚙ Controls")
run = st.sidebar.toggle("Start Camera", False)
st.sidebar.info("Emotion analysis runs every 2 seconds for better performance")

col1, col2 = st.columns([3, 2])

FRAME_WINDOW = col1.image([])
status_text = col2.empty()
chatbot_box = col2.empty()
graph_area = col2.empty()

# -----------------------------
# SESSION STATE
# -----------------------------
if "last_analysis" not in st.session_state:
    st.session_state.last_analysis = 0
    st.session_state.top_emotion = "neutral"
    st.session_state.confidence = 0

# -----------------------------
# CAMERA
# -----------------------------
cap = cv2.VideoCapture(0) if run else None

while run:
    ret, frame = cap.read()
    if not ret:
        st.error("Camera not accessible")
        break

    frame = cv2.flip(frame, 1)
    now = time.time()

    # -----------------------------
    # EMOTION ANALYSIS (THROTTLED)
    # -----------------------------
    if now - st.session_state.last_analysis > ANALYZE_EVERY:
        try:
            result = DeepFace.analyze(
                frame,
                actions=["emotion"],
                enforce_detection=False
            )

            emo_scores = result[0]["emotion"]
            st.session_state.top_emotion = max(
                EMOTIONS, key=lambda e: emo_scores.get(e, 0)
            )
            st.session_state.confidence = int(
                emo_scores[st.session_state.top_emotion]
            )

            emotion_history.append(st.session_state.top_emotion)
            st.session_state.last_analysis = now

        except:
            pass

    # -----------------------------
    # UI UPDATE
    # -----------------------------
    cv2.putText(
        frame,
        f"{st.session_state.top_emotion.upper()} ({st.session_state.confidence}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 0, 255),
        2
    )

    FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    status_text.markdown(
        f"### 🧠 Detected Emotion\n**{st.session_state.top_emotion.upper()} ({st.session_state.confidence}%)**"
    )

    chatbot_box.markdown(
        f"### 🤖 AI Assistant\n{CHATBOT_RESPONSES[st.session_state.top_emotion]}"
    )

    # -----------------------------
    # GRAPH (LIGHTWEIGHT)
    # -----------------------------
    fig, ax = plt.subplots(figsize=(4, 2))
    y_vals = [EMOTIONS.index(e) for e in emotion_history]
    ax.plot(y_vals, marker="o", linewidth=2)
    ax.set_yticks(range(len(EMOTIONS)))
    ax.set_yticklabels(EMOTIONS)
    ax.set_title("Emotion History")
    graph_area.pyplot(fig)

    time.sleep(0.05)

if cap:
    cap.release()
#python -m streamlit run app.py