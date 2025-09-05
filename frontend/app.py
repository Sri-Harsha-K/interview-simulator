import os, requests, streamlit as st
BACKEND = os.getenv("BACKEND_BASE", "http://127.0.0.1:8000")

st.set_page_config(page_title="Interview Simulator", page_icon="💬")
st.title("💬 Interview Simulator")

with st.sidebar:
    url = st.text_input("Backend URL", value=BACKEND)
    BACKEND = url or BACKEND
    st.caption("Uses POST /ask and POST /evaluate")

if "persona" not in st.session_state: st.session_state.persona = None
if "question" not in st.session_state: st.session_state.question = None
if "context" not in st.session_state: st.session_state.context = "Backend Engineer; APIs & scalability"

st.subheader("Context")
st.session_state.context = st.text_area("Describe the role/scenario:", st.session_state.context, height=100)

c1, c2 = st.columns(2)
with c1:
    if st.button("Ask me a question", use_container_width=True):
        try:
            r = requests.post(f"{BACKEND}/ask", json={"context": st.session_state.context}, timeout=60)
            r.raise_for_status()
            data = r.json()
            st.session_state.persona = data["persona"]
            st.session_state.question = data["question"]
            st.success(f"**{st.session_state.persona.upper()}** asks:")
        except Exception as e:
            st.error(f"/ask failed: {e}")
with c2:
    if st.button("Clear", use_container_width=True):
        st.session_state.persona = None
        st.session_state.question = None

if st.session_state.question:
    st.markdown(f"### ❓ Question\n{st.session_state.question}")
    st.subheader("Your answer")
    answer = st.text_area("Type your answer:", height=160, key="answer_box")

    c3, c4 = st.columns(2)
    with c3:
        if st.button("Submit answer", type="primary", use_container_width=True):
            try:
                r = requests.post(f"{BACKEND}/evaluate",
                                  json={"persona": st.session_state.persona,
                                        "question": st.session_state.question,
                                        "answer": answer or ""}, timeout=90)
                r.raise_for_status()
                st.markdown("### 📝 Feedback")
                st.write(r.json().get("feedback","").strip() or "_(no feedback)_")
            except Exception as e:
                st.error(f"/evaluate failed: {e}")
    with c4:
        if st.button("Begin the interview", use_container_width=True):
            try:
                r = requests.post(f"{BACKEND}/ask", json={"context": st.session_state.context}, timeout=60)
                r.raise_for_status()
                data = r.json()
                st.session_state.persona = data["persona"]
                st.session_state.question = data["question"]
                st.success(f"**{st.session_state.persona.upper()}** asks:")
            except Exception as e:
                st.error(f"/ask failed: {e}")
