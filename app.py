import streamlit as st
import pandas as pd
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_experimental.agents import create_pandas_dataframe_agent

# 1. Page Configuration
st.set_page_config(page_title="Enterprise GenAI Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Enterprise GenAI Data Assistant")

# 2. Sidebar Setup
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Enter your API Key", type="password")
    uploaded_file = st.file_uploader("Upload your Data file", type=["csv", "xlsx"])

# 3. Main Logic
if api_key and uploaded_file:
    st.success("✅ Setup complete! Ready for deep data analysis.")
    
    try:
        # Read the uploaded data file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
            
        st.subheader("📊 Data Preview")
        st.dataframe(df.head())
        
        # 4. Enterprise AI Agent Setup (The core engine)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
        
        # Create the Pandas Agent with improved error handling & iteration limit
        agent = create_pandas_dataframe_agent(
            llm, 
            df, 
            verbose=True, 
            allow_dangerous_code=True,
            handle_parsing_errors=True,
            max_iterations=3  
        )
        
        st.subheader("💬 Chat with your Data")
        user_query = st.text_input("Ask a complex question (e.g., 'What is the maximum selling price?' or 'How many rows are there?')")
        
        if st.button("Analyze Data") and user_query:
            with st.spinner("AI is analyzing your data..."):
                # Querying the agent directly
                response = agent.invoke(user_query)
                
                # The agent returns the final answer
                st.info(f"**AI says:** {response['output']}")
                
    except Exception as e:
        st.error(f"Error during analysis: {e}")

else:
    st.info("👉 Please enter your API Key and upload a data file in the sidebar to proceed.")