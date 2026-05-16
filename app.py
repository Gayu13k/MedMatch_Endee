import streamlit as st
from rag_pipeline import index_symptoms, discover

st.set_page_config(
    page_title="MedMatch",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MedMatch — Symptom Research Discovery")
st.caption("Describe symptoms → discover real PubMed research → AI-powered insights")
from vector_store import is_endee_available
if is_endee_available():
    st.success("🟢 Endee Vector DB connected — localhost:8080")
else:
    st.warning("🟡 Endee not connected — run Docker locally for full functionality")
# Sidebar
with st.sidebar:
    st.header("📚 Step 1: Index Research")
    st.markdown("Enter a medical topic to fetch & index papers from PubMed.")
    topic = st.text_input("Medical topic", placeholder="e.g. chest pain shortness of breath")
    if st.button("🔄 Fetch & Index from PubMed"):
        if topic:
            with st.spinner("Fetching from PubMed and indexing into Endee..."):
                count = index_symptoms(topic)
            if count > 0:
                st.success(f"✅ Indexed {count} papers into Endee!")
            else:
                st.error("No articles found. Try a different topic.")
        else:
            st.warning("Please enter a topic first.")
    st.markdown("---")
    st.markdown("**Try these topics:**")
    st.markdown("- chest pain shortness of breath")
    st.markdown("- type 2 diabetes fatigue")
    st.markdown("- migraine with aura")
    st.markdown("- anxiety sleep disorder")

# TWO TABS
tab1, tab2 = st.tabs(["🔍 Standard Search", "🤖 Agentic AI Mode"])

# ─────────────────────────────────────────────
# TAB 1 — Standard Search
# ─────────────────────────────────────────────
with tab1:
    st.header("🔍 Step 2: Describe Your Symptoms")
    st.info("⚠️ For research discovery only — not medical advice.")

    user_query = st.text_area(
        "Describe your symptoms in plain English",
        placeholder="e.g. I have persistent headaches behind my eyes, sensitivity to light and nausea for a week...",
        height=120,
        key="standard_query"
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        top_k = st.selectbox("Results", [3, 5, 8, 10], index=1)
    with col2:
        search_btn = st.button("🧬 Discover Research", type="primary")

    if search_btn and user_query:
        with st.spinner("Searching Endee + generating insights with LLaMA 3.3..."):
            answer, results, metrics = discover(user_query, top_k=top_k)

        # Metrics Dashboard
        st.subheader("📊 Retrieval Metrics")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("⏱️ Query Latency", f"{metrics['latency_ms']} ms")
        with col2:
            st.metric("🎯 Avg Similarity", f"{metrics['avg_similarity']}%")
        with col3:
            st.metric("📈 Precision@3", f"{metrics['precision_at_3']}%")
        with col4:
            st.metric("📈 Precision@5", f"{metrics['precision_at_5']}%")

        st.markdown("---")

        # AI Summary
        st.subheader("🤖 AI Research Summary")
        lines = answer.strip().split('\n')
        parsed = {}
        for line in lines:
            for key in ["CONDITIONS", "FINDINGS", "SEVERITY", "CAVEATS"]:
                if line.startswith(f"{key}:"):
                    parsed[key] = line.replace(f"{key}:", "").strip()

        if parsed:
            if "CONDITIONS" in parsed:
                st.markdown("**🏷️ Possible Related Conditions:**")
                conditions = parsed["CONDITIONS"].strip("[]").split(",")
                cols = st.columns(len(conditions))
                for i, cond in enumerate(conditions):
                    cols[i].info(cond.strip())

            if "SEVERITY" in parsed:
                severity_text = parsed["SEVERITY"]
                if "High" in severity_text:
                    st.error(f"🔴 Severity: {severity_text}")
                elif "Medium" in severity_text:
                    st.warning(f"🟡 Severity: {severity_text}")
                else:
                    st.success(f"🟢 Severity: {severity_text}")

            if "FINDINGS" in parsed:
                st.markdown("**🔬 Key Findings:**")
                st.write(parsed["FINDINGS"])

            if "CAVEATS" in parsed:
                st.markdown("**⚠️ Caveats:**")
                st.write(parsed["CAVEATS"])
        else:
            st.markdown(answer)

        st.markdown("---")

        # Retrieved Studies
        st.subheader(f"📄 Top {len(results)} Studies Retrieved from Endee")
        for i, r in enumerate(results, 1):
            meta = r.get("meta", {})
            sim = r.get("similarity", 0)
            sim_pct = round(sim * 100, 1)

            if sim_pct >= 70:
                tag = "🟢"
            elif sim_pct >= 50:
                tag = "🟡"
            else:
                tag = "🔴"

            with st.expander(f"{tag} {i}. {meta.get('title', 'Untitled')} — {sim_pct}% match"):
                st.progress(sim, text=f"Semantic similarity: {sim_pct}%")
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(f"**Journal:** {meta.get('journal', 'N/A')}")
                    st.markdown(f"**Date:** {meta.get('date', 'N/A')}")
                with col_b:
                    st.markdown(f"**Authors:** {meta.get('authors', 'N/A')}")
                    st.markdown(f"[🔗 View on PubMed]({meta.get('url', '#')})")
                st.write(meta.get("abstract_snippet", ""))

    elif search_btn:
        st.warning("Please describe your symptoms first.")

# ─────────────────────────────────────────────
# TAB 2 — Agentic AI Mode
# ─────────────────────────────────────────────
with tab2:
    st.markdown("### 🤖 Agentic AI Mode")
    st.info("The agent autonomously: assesses urgency → expands symptoms to medical terms → searches PubMed → synthesizes findings. No manual steps needed!")

    agent_query = st.text_area(
        "Describe your symptoms — the agent handles everything automatically",
        placeholder="e.g. I wake up with severe headaches, blurred vision and neck stiffness...",
        height=120,
        key="agent_query"
    )

    if st.button("🚀 Run Medical Agent", type="primary"):
        if agent_query:
            from medical_agent import run_agent

            with st.spinner("🤖 Agent is autonomously reasoning through your symptoms..."):
                steps, results, urgency, expanded = run_agent(
                    agent_query,
                    search_fn=lambda q: discover(q, top_k=5)[1],
                    index_fn=index_symptoms
                )

            # Urgency alert — shown first
            st.subheader("🚨 Urgency Assessment")
            if "EMERGENCY" in urgency:
                st.error("🚨 EMERGENCY — Seek immediate medical care!")
            elif "HIGH" in urgency:
                st.warning("⚠️ HIGH urgency — See a doctor soon")
            elif "MEDIUM" in urgency:
                st.warning("🟡 MEDIUM — Monitor closely, consider seeing a doctor")
            else:
                st.success("🟢 LOW urgency — Monitor symptoms")

            # Agent reasoning steps
            st.subheader("🧠 Agent Reasoning Steps")
            for i, step in enumerate(steps, 1):
                with st.expander(f"Step {i} — {step['action'].replace('_', ' ').title()}"):
                    st.code(step['result'])

            # Results
            st.subheader(f"📄 {len(results)} Studies Found by Agent")
            for i, r in enumerate(results, 1):
                meta = r.get("meta", {})
                sim = round(r.get("similarity", 0) * 100, 1)
                with st.expander(f"{i}. {meta.get('title', 'Untitled')} — {sim}% match"):
                    st.progress(r.get("similarity", 0))
                    st.markdown(f"**Journal:** {meta.get('journal', 'N/A')} | **Date:** {meta.get('date', 'N/A')}")
                    st.markdown(f"[🔗 View on PubMed]({meta.get('url', '#')})")
                    st.write(meta.get("abstract_snippet", ""))
        else:
            st.warning("Please describe your symptoms first.")