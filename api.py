"""Streamlit dashboard for the Automatic Question Generation API."""
from __future__ import annotations

import requests
import streamlit as st


st.set_page_config(
    page_title="Question Foundry",
    page_icon="Q",
    layout="wide",
    initial_sidebar_state="expanded",
)


API_TIMEOUT_SECONDS = 30
QUESTION_TYPES = {
    "Multiple choice": "multiple_choice",
    "True / false": "true_false",
    "Short answer": "short_answer",
    "Fill in the blank": "fill_in_blank",
}


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink: #17212b; --muted: #68737d; --coral: #e86b4f; --mint: #dcefe9; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: var(--ink); }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: 0; }
    [data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif; color: var(--coral); }
    [data-testid="stSidebar"] { background: #17212b; }
    [data-testid="stSidebar"] * { color: #eef4f1; }
    .eyebrow { color: var(--coral); font-size: .78rem; font-weight: 700; letter-spacing: .12em; text-transform: uppercase; }
    .hero { background: linear-gradient(120deg, #dcefe9 0%, #f7eee5 100%); padding: 1.5rem 1.8rem; border-radius: 10px; margin-bottom: 1.4rem; }
    .hero p { color: #52606b; margin-bottom: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


with st.sidebar:
    st.markdown("## Question Foundry")
    st.caption("Live control room for your learning materials")
    api_url = st.text_input("FastAPI URL", value="http://127.0.0.1:8000").rstrip("/")
    auto_refresh = st.toggle("Live refresh", value=True)
    refresh_seconds = st.select_slider("Refresh every", options=[5, 10, 20, 30], value=10)
    st.divider()
    st.caption("The dashboard reads and writes through the FastAPI service.")


def api_request(method: str, path: str, **kwargs):
    try:
        response = requests.request(
            method,
            f"{api_url}{path}",
            timeout=API_TIMEOUT_SECONDS,
            **kwargs,
        )
        if response.ok:
            return response.json(), None
        try:
            detail = response.json().get("detail", response.text)
        except ValueError:
            detail = response.text
        return None, f"{response.status_code}: {detail}"
    except requests.RequestException as exc:
        return None, f"API unavailable: {exc}"


def load_dashboard_data():
    health, health_error = api_request("GET", "/health")
    materials, materials_error = api_request("GET", "/materials")
    questions, questions_error = api_request("GET", "/questions")
    errors = [error for error in (health_error, materials_error, questions_error) if error]
    return health, materials, questions, errors


def render_live_dashboard():
    health, materials_data, questions_data, errors = load_dashboard_data()

    if errors:
        for error in errors:
            st.error(error)
        st.info("Start the API with: uvicorn app.main:app --reload")
        return

    materials = materials_data["materials"]
    questions = questions_data["questions"]
    question_types = {}
    difficulties = {}
    for question in questions:
        question_types[question["question_type"]] = question_types.get(question["question_type"], 0) + 1
        difficulties[question["difficulty"]] = difficulties.get(question["difficulty"], 0) + 1

    st.markdown('<div class="eyebrow">Live learning operations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><h1>Question Foundry</h1><p>Turn fresh study material into a visible, usable question bank.</p></div>', unsafe_allow_html=True)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Materials", len(materials))
    metric_columns[1].metric("Questions", len(questions))
    metric_columns[2].metric("Words indexed", f"{sum(item['word_count'] for item in materials):,}")
    metric_columns[3].metric("Engine", health["question_generation_mode"].replace("_", " ").title())

    chart_columns = st.columns([1.15, 1, 1])
    with chart_columns[0]:
        st.subheader("Question mix")
        st.bar_chart(question_types, color="#e86b4f", height=220)
    with chart_columns[1]:
        st.subheader("Difficulty")
        st.bar_chart(difficulties, color="#4c8c7a", height=220)
    with chart_columns[2]:
        st.subheader("Service status")
        st.success(f"API {health['status'].upper()}")
        st.write(f"Database: **{health['database']}**")
        st.write(f"Refresh: **{'on' if auto_refresh else 'off'}**")

    st.divider()
    left, right = st.columns([1, 1.35])
    with left:
        st.subheader("Add material")
        with st.form("upload_form", clear_on_submit=True):
            uploaded_file = st.file_uploader("PDF, DOCX, or TXT", type=["pdf", "docx", "txt"])
            upload_submitted = st.form_submit_button("Upload material", type="primary", use_container_width=True)
        if upload_submitted:
            if uploaded_file is None:
                st.warning("Choose a file first.")
            else:
                with st.spinner("Extracting and indexing text..."):
                    result, error = api_request(
                        "POST",
                        "/upload",
                        files={"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)},
                    )
                if error:
                    st.error(error)
                else:
                    st.success(f"Uploaded {result['original_filename']} ({result['word_count']:,} words).")
                    st.rerun()

        st.subheader("Generate questions")
        if not materials:
            st.caption("Upload a material to unlock generation.")
        else:
            material_options = {
                f"{material['original_filename']} · {material['word_count']:,} words": material["id"]
                for material in materials
            }
            with st.form("generate_form"):
                selected_label = st.selectbox("Material", list(material_options))
                count = st.slider("Number of questions", min_value=1, max_value=20, value=5)
                difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"], index=1)
                selected_types = st.multiselect(
                    "Question formats",
                    list(QUESTION_TYPES),
                    default=list(QUESTION_TYPES),
                )
                generate_submitted = st.form_submit_button("Generate now", type="primary", use_container_width=True)
            if generate_submitted:
                if not selected_types:
                    st.warning("Choose at least one question format.")
                else:
                    with st.spinner("Generating question set..."):
                        result, error = api_request(
                            "POST",
                            f"/generate/{material_options[selected_label]}",
                            json={
                                "num_questions": count,
                                "difficulty": difficulty,
                                "question_types": [QUESTION_TYPES[label] for label in selected_types],
                            },
                        )
                    if error:
                        st.error(error)
                    else:
                        st.success(f"Generated {result['generated_count']} questions.")
                        st.rerun()

    with right:
        st.subheader("Recent question bank")
        if not questions:
            st.info("No questions generated yet.")
        else:
            material_names = {material["id"]: material["original_filename"] for material in materials}
            table_rows = [
                {
                    "Question": question["question_text"],
                    "Type": question["question_type"].replace("_", " ").title(),
                    "Level": question["difficulty"].title(),
                    "Material": material_names.get(question["material_id"], "Unknown"),
                }
                for question in questions[:25]
            ]
            st.dataframe(table_rows, use_container_width=True, hide_index=True)
            st.caption(f"Showing {len(table_rows)} of {len(questions)} generated questions.")


if auto_refresh:
    st.fragment(run_every=refresh_seconds)(render_live_dashboard)()
else:
    render_live_dashboard()
