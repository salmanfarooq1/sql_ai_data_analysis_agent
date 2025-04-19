# 1. Load environment variables (must come before any Phidata/OpenAI import)
from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY from .env into os.environ

# 2. Standard library & 3rd‑party imports
import os
import json
import tempfile
import csv
import streamlit as st
import pandas as pd

# 3. Phidata imports (updated wrappers)
from phi.model.openai import OpenAIChat
from phi.agent import Agent
from phi.tools.duckdb import DuckDbTools

# 4. File preprocessing function
def preprocess_and_save(file):
    try:
        # 4.1 Read into DataFrame, handling CSV or Excel
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # 4.2 Escape internal quotes on string cols
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # 4.3 Auto‑cast date/numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

        # 4.4 Write out a fully‑quoted temp CSV for DuckDB
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
            df.to_csv(tmp.name, index=False, quoting=csv.QUOTE_ALL)
            temp_path = tmp.name

        return temp_path, df.columns.tolist(), df

    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# 5. Initialize Streamlit session state
for var in ('dataframe', 'file_path', 'columns', 'last_file_name', 'query_history'):
    if var not in st.session_state:
        st.session_state[var] = None if var != 'query_history' else []

# 6. App header
st.title("📊 AI Data Analyst")
st.markdown("Upload your data and ask questions in natural language to analyze it.")

# 7. Sidebar: API configuration + examples + history
with st.sidebar:
    st.header("API Configuration")

    if "OPENAI_API_KEY" in os.environ:
        st.success("✅ OpenAI API key loaded from environment")
        openai_api_key = os.environ["OPENAI_API_KEY"]
    else:
        openai_api_key = st.text_input("Enter your OpenAI API key:", type="password")
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
            st.success("✅ OpenAI API key saved to environment")
    if openai_api_key:
        st.session_state.openai_api_key = openai_api_key

    st.header("Example Queries")
    st.markdown("""
    - Show me the top 5 rows of the dataset  
    - What is the average value of column X?  
    - Find correlations between columns X and Y  
    - Create a summary of the data  
    - Show me rows between the 5th and 95th percentile of a column
    """)

    if st.session_state.query_history:
        st.header("Recent Queries")
        for q in st.session_state.query_history[-5:]:
            st.text(f"• {q}")

# 8. Main: File upload
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])
if uploaded_file:
    # 8.1 Preprocess only on new file
    if (st.session_state.dataframe is None
        or uploaded_file.name != st.session_state.last_file_name):
        with st.spinner("Processing data…"):
            path, cols, df = preprocess_and_save(uploaded_file)
            if path and cols and df is not None:
                st.session_state.dataframe = df
                st.session_state.file_path = path
                st.session_state.columns = cols
                st.session_state.last_file_name = uploaded_file.name
                st.session_state.query_history = []
                st.success(f"✅ Processed {uploaded_file.name}")
            else:
                st.error("Failed to preprocess. Check file format.")

    # 8.2 Show preview & metrics
    if st.session_state.dataframe is not None:
        st.subheader("Data Preview")
        st.dataframe(st.session_state.dataframe.head(10))

        c1, c2 = st.columns(2)
        with c1:
            st.metric("Rows", f"{len(st.session_state.dataframe):,}")
        with c2:
            st.metric("Columns", len(st.session_state.dataframe.columns))

        # 9. Query interface (only if API key is set)
        if hasattr(st.session_state, 'openai_api_key'):
            st.subheader("Ask Questions About Your Data")
            user_query = st.text_area(
                "Enter your query in natural language:",
                placeholder="e.g. Show me the top 5 rows with highest values in column X"
            )

            if st.button("Run Analysis"):
                if not user_query.strip():
                    st.warning("Please enter a query first.")
                else:
                    st.session_state.query_history.append(user_query)
                    try:
                        with st.spinner("Analyzing…"):
                            # 9.1 Instantiate the OpenAIChat LLM
                            llm = OpenAIChat(id="gpt-3.5-turbo")

                            # 9.2 Build and run the Agent
                            agent = Agent(
                                llm=llm,
                                tools=[DuckDbTools()],
                                show_tool_calls=True,
                                system_prompt=f"""
You are an expert data analyst. Use SQL on the CSV at {st.session_state.file_path}.
Dataset columns: {', '.join(st.session_state.columns)}
Provide both SQL and results, plus any simple viz guidance.
"""
                            )
                            response = agent.run(user_query)

                        # 9.3 Display results
                        st.subheader("Analysis Results")
                        content = getattr(response, "content", str(response))
                        st.markdown(content)

                    except Exception as e:
                        st.error(f"Analysis error: {e}")
                        st.info("Try simplifying your query or checking your column names.")
                        st.expander("Debug info", expanded=False).write(str(e))
        else:
            st.warning("🔑 Please set your OpenAI API key in the sidebar.")
else:
    st.info("👋 Upload a CSV or Excel file to get started.")
