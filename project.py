

import streamlit as st
import google.generativeai as genai
import mysql.connector
import pandas as pd
import io
from mysql.connector import Error
import networkx as nx
import plotly.express as px
import speech_recognition as sr
import tempfile
import os
from gtts import gTTS
import requests
import random 
import json
from datetime import datetime
import time
import plotly.graph_objects as go
import pyperclip


import requests
from bs4 import BeautifulSoup




import sqlite3  # Add this import for SQLite support
import shutil   # Add this for temporary file copying

# Configure the API Key
GOOGLE_API_KEY = "AIzaSyDX8os0xVmRfWDXfKR-rnaoks3_GfwtwFg"
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

YOUTUBE_API_KEY = "AIzaSyD2k2-kMObtxWrX_RQVQhFhUhXzS3ZbVk8"



# Helper function to manage databases/ folder and list .db files
def get_db_files():
    db_dir = "databases"
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
        except OSError as e:
            st.error(f"Failed to create databases directory: {e}")
            return []
    return [f for f in os.listdir(db_dir) if f.endswith(".db")]



def create_sample_database():
    db_dir = "databases"
    if not os.path.exists(db_dir):
        try:
            os.makedirs(db_dir)
        except OSError as e:
            st.error(f"Failed to create databases directory: {e}")
            return
    
    db_path = os.path.join(db_dir, "sales_demo.db")
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        
        # Create sample tables
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS customers (
                customer_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                email TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS products (
                product_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                price REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS orders (
                order_id INTEGER PRIMARY KEY,
                customer_id INTEGER,
                product_id INTEGER,
                order_date TEXT,
                FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
                FOREIGN KEY (product_id) REFERENCES products(product_id)
            )
        """)
        
        # Insert sample data
        cursor.execute("INSERT OR IGNORE INTO customers (customer_id, name, email) VALUES (1, 'Alice', 'alice@example.com')")
        cursor.execute("INSERT OR IGNORE INTO customers (customer_id, name, email) VALUES (2, 'Bob', 'bob@example.com')")
        cursor.execute("INSERT OR IGNORE INTO products (product_id, name, price) VALUES (1, 'Laptop', 999.99)")
        cursor.execute("INSERT OR IGNORE INTO products (product_id, name, price) VALUES (2, 'Phone', 499.99)")
        cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_id, order_date) VALUES (1, 1, 1, '2023-10-01')")
        cursor.execute("INSERT OR IGNORE INTO orders (order_id, customer_id, product_id, order_date) VALUES (2, 2, 2, '2023-10-02')")
        
        conn.commit()
        # Verify tables were created
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
        conn.close()
        
        if set(tables) != {"customers", "products", "orders"}:
            st.error(f"Failed to create all expected tables in {db_path}. Created tables: {tables}")
    except sqlite3.Error as e:
        st.error(f"Failed to create sample database: {e}")



def search_web(query):
    try:
        url = f"https://duckduckgo.com/html/?q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for item in soup.select(".result__body")[:5]:
            title_elem = item.select_one(".result__title a")
            snippet_elem = item.select_one(".result__snippet")
            if title_elem and snippet_elem:
                title = title_elem.text
                snippet = snippet_elem.text  # Full snippet, no truncation
                results.append(f"- **{title}**: {snippet}")
      
        return results if results else ["- No relevant results found."]
    except requests.RequestException as e:
        return [f"- Error fetching web data: {str(e)}"]


def agent_response(user_input, db_params, model, chat_history, uploaded_file_content=None, uploaded_image_content=None, use_web=False, use_deep_search=False):
    
    schema = fetch_tables(db_params) if db_params.get("database") else "No schema available"
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history])
    file_content = uploaded_file_content if uploaded_file_content else ""
    image_description = "User uploaded an image. Describe or analyze it based on the input." if uploaded_image_content else ""

    # Web browsing logic
    web_info = ""
    if use_web or use_deep_search:
        web_results = search_web(user_input)
        web_info = f"Web Search Results (as of March 31, 2025) for '{user_input}':\n" + "\n".join(web_results)
        
   
    

    # Deep Search logic
    deep_search_info = ""
    if use_deep_search:
        schema_display = "\n  - " + "\n  - ".join(schema["Table Name"].tolist()) if isinstance(schema, pd.DataFrame) else "No DB connected"
        deep_search_info = f"\nDeep Search Enabled: Analyzing '{user_input}' with all available data:\n" \
                           f"- Database Schema: {schema_display}\n" \
                           f"- Uploaded File: {file_content[:100] + '...' if file_content else 'None'}\n" \
                           f"- Uploaded Image: {image_description if image_description else 'None'}\n" \
                           f"- Chat History Summary: {context[:100] + '...' if context else 'None'}\n" \
                           f"- {web_info if web_info else 'Web Data: No web data available.'}\n" \
                           
      
    

    prompt = f"""
    
    You are an AI SQL Assistant with a friendly, conversational tone like Grok or ChatGPT.
    Use the following context to respond, tailoring the response strictly based on the enabled features:

    Database Schema: {schema if isinstance(schema, str) else schema.to_string(index=False)}
    Chat History: {context}
    Uploaded File Content: {file_content}
    Image Description: {image_description}
    {web_info if use_web and not use_deep_search else ""}
    {deep_search_info if use_deep_search else ""}
    User Input: {user_input}
    
    ### Instructions:
    - If deep search is enabled (use_deep_search=True), YOU MUST:
      1. Start with the exact 'Deep Search Enabled' section provided above as a structured list.
      2.Follow it with a detailed synthesis focusing on the current User Input. Use the data sources (schema, web, files, images, history) to inform your response, but only reference chat history if it directly enhances the answer to the current input (e.g., for context or follow-up). Use complete sentences and specific examples from the data.
      3. Suggest SQL queries or follow-ups, using generic examples if no schema is provided.
    - If deep search is disabled (use_deep_search=False):
      - **DO NOT** include the 'Deep Search Enabled' section or any deep search formatting, regardless of what’s in the chat history.
      - Ignore any prior deep search structure in the chat history—focus only on the current User Input.
      - Provide a concise, knowledge-based answer using only the schema, uploaded content, and general knowledge, without external web or X data unless use_web is enabled.
    - If web browsing is enabled (but not deep search), include the exact 'Web Search Results' provided above, followed by a summary in your own words.
    - If neither web nor deep search is enabled, give a simple, direct answer without external references or formatting.
    - If the user uploaded a SQL file, analyze its content.
    - If the user uploaded an image, describe it or use it to inform your response if relevant.
    - If the user asks a SQL-related question, generate a query, explain it, and execute it if possible.
    - If the user asks a general question, provide a clear, detailed answer.
    - If the user asks for analysis, suggest advanced SQL like window functions (e.g., ROW_NUMBER, RANK), CTEs, or aggregations (e.g., AVG, STDDEV).

    - Be proactive: suggest follow-up actions or related queries based on the current input and available data.
    - Use a conversational tone, and add a touch of humor where appropriate.
    """

    response = model.generate_content(prompt).text.strip()
    

    if "select" in user_input.lower() or "query" in user_input.lower() or (file_content and "select" in file_content.lower()):
        try:
            sql_query = response.split("```sql\n")[1].split("\n```")[0] if "```sql\n" in response else None
            if not sql_query and file_content:
                sql_query = file_content.strip()
            if sql_query and db_params.get("database"):
                result = execute_query(sql_query, db_params)
                response += f"\n\n**Query Results**:\n{result.get('data', 'No results')}"
        except Exception as e:
            response += f"\n\n**Error executing query**: {str(e)}"

    # Strictly control note addition
    final_response = response
    if use_deep_search:
        final_response += "\n\n*Note: This is a deep search result integrating multiple data sources!*"
        
    elif use_web:
        final_response += "\n\n*Note: This response includes real-time info from web browsing!*"
       

   
    return str(final_response)


def generate_suggestions(db_params):
    schema = fetch_tables(db_params) if db_params.get("database") else None
    if isinstance(schema, pd.DataFrame) and not schema.empty:
        tables = schema["Table Name"].tolist()
        suggestions = []
        for table in tables[:2]:  # Limit to first 2 tables for brevity
            table_data = fetch_table_data(db_params, table)
            cols = table_data.columns.tolist()
            suggestions.extend([
                f"Show all records from {table}",
                f"Count rows in {table}",
                f"Select {cols[0]} from {table} where {cols[1]} > 10" if len(cols) > 1 else f"Select * from {table}"
            ])
        return suggestions
    return ["Try connecting to a database for tailored suggestions!"]


def render_download_buttons(report_data):
    download_format = st.selectbox(
        "Choose download format for the query report:",
        ["CSV", "JSON", "Text", "PDF"]
    )
    if download_format == "CSV":
        # Convert DataFrame results to string for CSV compatibility
        report_data_csv = report_data.copy()
        if "results" in report_data_csv.columns:
            report_data_csv["results"] = report_data_csv["results"].apply(
                lambda x: x.to_string(index=False) if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else ""
            )
        st.download_button(
            label="Download CSV",
            data=report_data_csv.to_csv(index=False),
            file_name="sql_query_report.csv",
            mime="text/csv",
            key="csv_download"
        )
    elif download_format == "JSON":
        # Convert DataFrame results to list of dictionaries for JSON
        report_data_json = report_data.copy()
        if "results" in report_data_json.columns:
            report_data_json["results"] = report_data_json["results"].apply(
                lambda x: x.to_dict(orient="records") if isinstance(x, pd.DataFrame) else str(x) if pd.notna(x) else None
            )
        st.download_button(
            label="Download JSON",
            data=report_data_json.to_json(orient="records"),
            file_name="sql_query_report.json",
            mime="application/json",
            key="json_download"
        )
    elif download_format == "Text":
        text_content = ""
        for _, row in report_data.iterrows():
            text_content += f"Prompt:\n{row['prompt']}\n\n"
            text_content += f"Query:\n{row['query']}\n\n"
            text_content += f"Explanation:\n{row['explanation']}\n\n"
            text_content += f"Expected Output:\n{row['output']}\n\n"
            if "results" in row and row["results"] is not None:
                text_content += "Executed Query Results:\n"
                if isinstance(row["results"], pd.DataFrame):
                    text_content += row["results"].to_string(index=False) + "\n"
                else:
                    text_content += str(row["results"]) + "\n"
            text_content += "-" * 50 + "\n"
        st.download_button(
            label="Download Text",
            data=text_content,
            file_name="sql_query_report.txt",
            mime="text/plain",
            key="text_download"
        )
    elif download_format == "PDF":
        pdf_content = generate_pdf_report(report_data)
        st.download_button(
            label="Download PDF",
            data=pdf_content,
            file_name="sql_query_report.pdf",
            mime="application/pdf",
            key="pdf_download"
        )




def fetch_databases(host, user, password):
    # Note: This function will use demo_mode from main(), so it needs to be adjusted there
    # For now, we'll assume demo_mode is passed or accessed globally if needed
    try:
        conn = mysql.connector.connect(host=host, user=user, password=password)
        cursor = conn.cursor()
        cursor.execute("SHOW DATABASES")
        databases = [db[0] for db in cursor.fetchall()]
        conn.close()
        return databases
    except Error as e:
        return []

        
# Function to get topic from Gemini API
def get_sql_topic_from_gemini(query):
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")  # Replace with the desired model
        response = model.generate_content(query)
        topic = response.text.strip()  # Get the SQL topic from Gemini's response
        return topic
    except Exception as e:
        st.error(f"Error getting topic from Gemini API: {e}")
        return None

    
# Function to search YouTube videos based on the query
def search_youtube_videos(api_key, search_query):
    youtube_search_url = "https://www.googleapis.com/youtube/v3/search"
    params = {
        "part": "snippet",
        "q": f"SQL {search_query} tutorial",  # More descriptive search query
        "type": "video",
        "key": api_key,
        "maxResults": 7,
    }

    try:
        response = requests.get(youtube_search_url, params=params)
        response.raise_for_status()
        response_data = response.json()

        # Extract video details
        video_results = []
        for video in response_data.get("items", []):
            video_id = video["id"]["videoId"]
            title = video["snippet"]["title"]
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            video_results.append({"title": title, "url": video_url})
        
        return video_results
    except requests.exceptions.RequestException as e:
        st.error(f"Error with YouTube API: {e}")
        return []
    except KeyError as e:
        st.error(f"Unexpected response format: {e}")
        return []


# Function to determine the modified table (can be enhanced based on your use case)
def get_modified_table_from_query(query):
    # Identify the modified table based on query keywords (e.g., INSERT INTO, UPDATE, DELETE FROM)
    query_lower = query.strip().lower()
    if query_lower.startswith("insert into"):
        return query.split()[2]  # Assuming table name follows 'INSERT INTO'
    elif query_lower.startswith("update"):
        return query.split()[1]  # Assuming table name follows 'UPDATE'
    elif query_lower.startswith("delete from"):
        return query.split()[2]  # Assuming table name follows 'DELETE FROM'
    elif query_lower.startswith(("create", "alter", "drop")):
        # For DDL queries, the table is usually specified after the command
        return query.split()[2] if len(query.split()) > 2 else None
    return None
    
def execute_query(query, db_params):
    try:
        if not db_params.get("database") and not query.strip().lower().startswith("create database"):
            return {"status": "error", "message": "Database connection details are missing."}

        if db_params.get("type") == "sqlite":
            conn = sqlite3.connect(db_params["database"])
            cursor = conn.cursor()
            cursor.execute("PRAGMA foreign_keys = ON")  # Enable foreign keys for SQLite
            cursor.execute(query)
            
            if query.strip().lower().startswith(("create", "alter", "drop")):
                conn.commit()
                tables = fetch_tables(db_params)
                conn.close()
                return {"status": "success", "message": "DDL query executed.", "data": tables}
            elif query.strip().lower().startswith("select"):
                data = pd.read_sql_query(query, conn)
                conn.close()
                return {"status": "success", "data": data}
            else:  # INSERT, UPDATE, DELETE
                conn.commit()
                modified_table = get_modified_table_from_query(query)
                if modified_table:
                    table_data = fetch_table_data(db_params, modified_table)
                    conn.close()
                    return {"status": "success", "data": table_data}
                conn.close()
                return {"status": "success", "data": pd.DataFrame()}
        else:  # MySQL
            if query.strip().lower().startswith(("create database", "drop database")):
                conn = mysql.connector.connect(
                    host=db_params.get("host", "localhost"),
                    user=db_params.get("user", "root"),
                    password=db_params.get("password", "")
                )
                cursor = conn.cursor()
                cursor.execute(query)
                conn.close()
                return {"status": "success", "message": f"Database operation successful."}
            
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute(query)
            
            if query.strip().lower().startswith(("create", "alter", "drop")):
                conn.commit()
                tables = fetch_tables(db_params)
                conn.close()
                return {"status": "success", "message": "DDL query executed.", "data": tables}
            elif query.strip().lower().startswith("select"):
                columns = [col[0] for col in cursor.description]
                data = cursor.fetchall()
                conn.close()
                return {"status": "success", "data": pd.DataFrame(data, columns=columns)}
            else:
                conn.commit()
                modified_table = get_modified_table_from_query(query)
                if modified_table:
                    table_data = fetch_table_data(db_params, modified_table)
                    conn.close()
                    return {"status": "success", "data": table_data}
                conn.close()
                return {"status": "success", "data": pd.DataFrame()}
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error executing query: {str(e)}"}



# Function to process SQL file content
def process_sql_file(file_content):
    queries = file_content.strip().split(";")
    return [query.strip() for query in queries if query.strip()]


def transcribe_audio_to_text():
    """
    Captures audio from the microphone and transcribes it into text.
    Returns the transcribed text or an error message.
    """
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        st.info("Listening... Please speak into the microphone.")
        try:
            audio = recognizer.listen(source, timeout=5)
            st.info("Processing the audio...")
            text = recognizer.recognize_google(audio)  # Using Google Web Speech API
            return text
        except sr.WaitTimeoutError:
            return "Error: Timeout. Please speak louder or check your microphone."
        except sr.UnknownValueError:
            return "Error: Could not understand the audio."
        except sr.RequestError as e:
            return f"Error: Could not request results from the speech recognition service; {e}"

# Function to generate explanation for SQL query
def generate_explanation(sql_query):
    explanation_template = """
        Explain the SQL Query snippet:
        {sql_query}
        Please provide the simplest explanation:
    """
    explanation_formatted = explanation_template.format(sql_query=sql_query)
    explanation_response = model.generate_content(explanation_formatted)
    explanation = explanation_response.text.strip()

    return explanation

def text_to_speech(text):
    if not text.strip():
        st.warning("No text to convert to speech.")
        return None,None
    try:
        tts = gTTS(text=text, lang='en')
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as tmp_file:
            tts.save(tmp_file.name)
            tmp_file.seek(0)
            audio_bytes = tmp_file.read()
            tmp_file_path = tmp_file.name
        return audio_bytes, tmp_file_path  # Return bytes and path for cleanup
    except Exception as e:
        st.error(f"Error generating audio: {e}")
        return None, None


# Function to generate expected output for SQL query
def generate_expected_output(sql_query):
    expected_output_template = """
        What would be the expected output of the SQL Query snippet:
        {sql_query}
        Provide a sample tabular response with no explanation:
    """
    expected_output_formatted = expected_output_template.format(sql_query=sql_query)
    expected_output_response = model.generate_content(expected_output_formatted)
    expected_output = expected_output_response.text.strip()
    return expected_output

# Fetch tables from the database
def fetch_tables(db_params):
    try:
        if db_params.get("type") == "sqlite":
            if not db_params.get("database"):
                return {"status": "error", "message": "No SQLite database selected."}
            if not os.path.exists(db_params["database"]):
                return {"status": "error", "message": f"Database file {db_params['database']} does not exist."}
            # Retry connection up to 3 times with delay to ensure file is ready
            for attempt in range(3):
                try:
                    conn = sqlite3.connect(db_params["database"])
                    cursor = conn.cursor()
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
                    tables = [table[0] for table in cursor.fetchall() if table[0] != "sqlite_sequence"]
                    conn.close()
                    return pd.DataFrame(tables, columns=["Table Name"])
                except sqlite3.Error as e:
                    if attempt < 2:
                        time.sleep(0.5)  # Wait before retrying
                        continue
                    return {"status": "error", "message": f"Error fetching tables: {str(e)} - Path: {db_params['database']}"}
        else:
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute("SHOW TABLES")
            tables = [table[0] for table in cursor.fetchall()]
            conn.close()
            return pd.DataFrame(tables, columns=["Table Name"])
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error fetching tables: {str(e)} - Path: {db_params.get('database', 'N/A')}"}
def fetch_table_data(db_params, table_name):
    try:
        if db_params.get("type") == "sqlite":
            conn = sqlite3.connect(db_params["database"])
            query = f"SELECT * FROM {table_name}"
            data = pd.read_sql_query(query, conn)
            conn.close()
            return data
        else:
            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table_name}")
            columns = [col[0] for col in cursor.description]
            data = cursor.fetchall()
            conn.close()
            return pd.DataFrame(data, columns=columns)
    except (Error, sqlite3.Error) as e:
        return {"status": "error", "message": f"Error fetching data from table {table_name}: {str(e)}"}

    
# Function to generate a PDF report from the generated data
def generate_pdf_report(report_data):
    import io
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # Add title
    pdf.cell(200, 10, txt="SQL Query Report", ln=True, align="C")
    pdf.ln(10)

    # Iterate through the data and add it to the PDF
    for _, row in report_data.iterrows():
        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Prompt:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["prompt"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Query:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["query"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Explanation:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["explanation"]))

        pdf.set_font("Arial", style="B", size=12)
        pdf.multi_cell(0, 10, f"Expected Output:")
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, str(row["output"]))
        

        # Include executed results if available (fixed condition)
        if "results" in row and row["results"] is not None:
            pdf.set_font("Arial", style="B", size=12)
            pdf.multi_cell(0, 10, "Executed Query Results:")
            pdf.set_font("Arial", size=12)
            if isinstance(row["results"], pd.DataFrame):
                results_text = row["results"].to_string(index=False)
            else:
                results_text = str(row["results"])
            pdf.multi_cell(0, 10, results_text)

        pdf.ln(5)

    pdf_output = io.BytesIO()
    pdf_content = pdf.output(dest="S").encode("latin1")
    pdf_output.write(pdf_content)
    pdf_output.seek(0)
    return pdf_output   

def main():
    st.set_page_config(page_title="SQL Query Assistant", page_icon="🔍", layout="wide")
   
   # Create sample database for demo mode
    create_sample_database()

    st.markdown(
        """
        <div style="text-align: center; padding: 20px;">
            <h1>SQL Query Assistant 🤖</h1>
            <h3>Generate SQL queries effortlessly ✨</h3>
            <h4>Get explanations, expected outputs, and optional execution 📚</h4>
        </div>
        """,
        unsafe_allow_html=True
    )

   # Initialize session state at the very start
    if "chat_sessions" not in st.session_state:
        st.session_state.chat_sessions = {
            "Chat 1": [
                {"role": "agent", "content": "Hi there! How can I help you with SQL today? 😊", "timestamp": datetime.now().strftime("%H:%M:%S")}
            ]
        }
      
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    if "use_emojis" not in st.session_state:
        st.session_state.use_emojis = True
    if "active_chat" not in st.session_state:
        st.session_state.active_chat = "Chat 1"
    if "editing_message" not in st.session_state:
        st.session_state.editing_message = None  # Tracks which message is being edited
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = []

    # Sidebar for Database Connection
    st.sidebar.header("Database Connection (Optional)")
    db_type = st.sidebar.radio("Database Type", ["Demo (SQLite)", "MySQL", "SQLite"], index=0)  # Default to Demo

    if db_type == "Demo (SQLite)":
        db_path = os.path.join("databases", "sales_demo.db")
        if not os.path.exists(db_path):
            st.sidebar.error("Sample database not found. Attempting to recreate...")
            create_sample_database()
        st.sidebar.info("Demo Mode: Using a sample SQLite sales database with 'orders', 'customers', and 'products' tables.")
        db_params = {"database": db_path, "type": "sqlite"}
    elif db_type == "SQLite":
        st.sidebar.subheader("Manage SQLite Databases")
        # Create new .db file
        new_db_name = st.sidebar.text_input("Create New Database (e.g., new_database.db)", "")
        if new_db_name:
            if not new_db_name.endswith(".db"):
                new_db_name += ".db"
            new_db_path = os.path.join("databases", new_db_name)
            if st.sidebar.button("Create Database"):
                if os.path.exists(new_db_path):
                    st.sidebar.error(f"Database '{new_db_name}' already exists! Choose another name.")
                else:
                    try:
                        conn = sqlite3.connect(new_db_path)
                        conn.close()
                        st.sidebar.success(f"Database '{new_db_name}' created successfully!")
                        st.rerun()  # Refresh to update dropdown
                    except sqlite3.Error as e:
                        st.sidebar.error(f"Failed to create database: {e}")

        # Select existing .db file
        db_files = get_db_files()
        if not db_files:
            st.sidebar.warning("No .db files found. Create one or upload a file.")
            selected_db = None
        else:
            selected_db = st.sidebar.selectbox("Select Existing Database", [""] + db_files)

        # Upload .db file
        sqlite_file = st.sidebar.file_uploader("Upload SQLite Database File (.db)", type=["db"])
        if sqlite_file:
            new_db_name = sqlite_file.name
            new_db_path = os.path.join("databases", new_db_name)
            if os.path.exists(new_db_path):
                st.sidebar.error(f"File '{new_db_name}' already exists! Rename or delete the existing file.")
            else:
                with open(new_db_path, "wb") as f:
                    f.write(sqlite_file.read())
                st.sidebar.success(f"Uploaded '{new_db_name}' to databases/")
                st.rerun()  # Refresh to update dropdown

        # Remove .db file
        if selected_db and selected_db != "sales_demo.db":
            st.sidebar.warning(f"Delete '{selected_db}'? This cannot be undone.")
            if st.sidebar.button("Delete Database"):
                try:
                    os.remove(os.path.join("databases", selected_db))
                    st.sidebar.success(f"Database '{selected_db}' deleted successfully!")
                    st.rerun()  # Refresh to update dropdown
                except OSError as e:
                    st.sidebar.error(f"Failed to delete database: {e}")

        # Set db_params based on selection or upload
        if selected_db:
            db_params = {"database": os.path.join("databases", selected_db), "type": "sqlite"}
        else:
            db_params = {"database": "", "type": "sqlite"}  # No database selected
    else:  # MySQL
        host = st.sidebar.text_input("Host", "localhost")
        user = st.sidebar.text_input("Username", "root")
        password = st.sidebar.text_input("Password", type="password")
        databases = fetch_databases(host, user, password) or []
        database = st.sidebar.selectbox(
            "Select Database",
            databases if databases else ["Enter credentials to fetch databases"],
            disabled=not bool(databases)
        )
        db_params = {
            "host": host,
            "user": user,
            "password": password,
            "database": database if database != "Enter credentials to fetch databases" else "",
            "type": "mysql"
        }

    if st.sidebar.button("Test Connection"):
        try:
            if db_params.get("type") == "sqlite" and db_params.get("database"):
                conn = sqlite3.connect(db_params["database"])
            elif db_params.get("type") == "mysql" and db_params.get("database"):
                conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
            else:
                st.sidebar.error("Please select or create a database first.")
                return
            conn.close()
            st.sidebar.success("Database connection successful!")
        except (Error, sqlite3.Error) as e:
            st.sidebar.error(f"Connection failed: {e}")

    

    if db_params.get("database"):  # Check if database is set instead of all(db_params.values())
        tables = fetch_tables(db_params)
        if isinstance(tables, pd.DataFrame):
            st.sidebar.subheader("Tables in Database")
            selected_table = st.sidebar.selectbox("Select a table to view:", tables["Table Name"].tolist())
            if selected_table:
                table_data = fetch_table_data(db_params, selected_table)
                if isinstance(table_data, pd.DataFrame):
                    st.sidebar.write(f"Contents of `{selected_table}`:")
                    st.sidebar.dataframe(table_data)
                else:
                    st.sidebar.error(table_data["message"])
        else:
            st.sidebar.error(tables["message"])


    st.sidebar.header("Upload SQL Files")
    uploaded_files = st.sidebar.file_uploader("Upload one or more .sql files", type=["sql"], accept_multiple_files=True)

    generated_data = []  # To store generated prompts, queries, and outputs
    # Use session state to persist generated_data across interactions
    if "generated_data" not in st.session_state:
        st.session_state.generated_data = []
    tabs = st.selectbox(
        "Choose a feature",
        ["Generate Query from English", "Upload SQL Files", "Visualize Data", "Database Schema", "Learn SQL","AI Agent","Query History","SQL Playground"]
    )

    # Handling Tabs
    if tabs == "Generate Query from English":
        st.header("Generate SQL Query from Plain English")
        if "voice_input" not in st.session_state:
            st.session_state["voice_input"] = ""

        
        text_input = st.text_area(
            "Enter your SQL query in plain English:",
            placeholder="E.g., Show all employees in the IT department",
            value=st.session_state.get("voice_input", "")
        )

        if st.button("Record Audio 🎙️"):
            voice_text = transcribe_audio_to_text()
            if "Error" not in voice_text:
                st.session_state["voice_input"] = voice_text
                st.success(f"Transcribed Text: {voice_text}")
                text_input = voice_text
            else:
                st.error(voice_text)

        if st.button("Generate SQL Query"):
            # Clear previous generated_data to avoid duplicates
            st.session_state.generated_data = []

            if text_input.strip() == "":
                st.warning("Please enter a valid plain English query.")
                return

            with st.spinner("Generating SQL Query..."):
                try:
                    template = """
                        Create a SQL Query snippet using the below text:
                        {text_input}
                        I just want a SQL Query.
                    """
                    formatted_template = template.format(text_input=text_input)
                    response = model.generate_content(formatted_template)
                    sql_query = response.text.strip().lstrip("```sql").rstrip("```")

                    explanation = generate_explanation(sql_query)
                    expected_output = generate_expected_output(sql_query)

                    st.success("SQL Query Generated Successfully! Here is your query:")
                    st.code(sql_query, language="sql")

                    

                    st.success("Explanation of the SQL Query:")
                    st.markdown(explanation)
                    if explanation:
                        # st.audio(text_to_speech(explanation), format="audio/mp3")
                        audio_bytes, tmp_file_path = text_to_speech(explanation)
                        if audio_bytes:
                         st.audio(audio_bytes, format="audio/mp3")
                        os.unlink(tmp_file_path)  # Clean up

                    if expected_output:
                        st.success("Expected Output of the SQL Query:")
                        st.markdown(expected_output)


                    # Execute query if database connection is provided
                    executed_results = None
                    if all(db_params.values()):
                        st.info("Executing query...")
                        result = execute_query(sql_query, db_params)
                        if result["status"] == "success":
                            if "data" in result and not result["data"].empty:
                                st.success("Query executed successfully! Displaying results:")
                                st.dataframe(result["data"])
                                executed_results = result["data"]  # Store the DataFrame

                                st.session_state.query_history.append({"query": sql_query, "result": result["data"]})
                            else:
                                st.success(result["message"])
                                executed_results = result["message"]  # Store the message as string
                                st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                        else:
                            st.error(f"Error executing query: {result['message']}")
                            executed_results = result["message"]  # Store error message
                            st.session_state.query_history.append({"query": sql_query, "result": result["message"]})
                    st.download_button(
                        label="Download SQL Query",
                        data=sql_query,
                        file_name="generated_query.sql",
                        mime="text/sql"
                    )

                    st.session_state.generated_data.append({
                        "prompt": text_input,
                        "query": sql_query,
                        "explanation": explanation,
                        "output": expected_output,
                        "results": executed_results
                    })

                   
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        # # Download Report Section (Moved here)
            # Download Report Section
        if st.session_state.generated_data:
            report_data = pd.DataFrame(st.session_state.generated_data)
            st.subheader("Download Query Report")
            render_download_buttons(report_data)

        if generated_data:
            report_data = pd.DataFrame(generated_data)
            download_format = st.selectbox(
                "Choose download format for the query report:",
                ("JSON", "CSV", "Text", "PDF")
            )

            if download_format == "CSV":
                st.download_button(
                    label="Download CSV",
                    data=report_data.to_csv(index=False),
                    file_name="sql_query_report.csv",
                    mime="text/csv"
                )
            elif download_format == "JSON":
                st.download_button(
                    label="Download JSON",
                    data=report_data.to_json(orient="records"),
                    file_name="sql_query_report.json",
                    mime="application/json"
                )
            elif download_format == "Text":
                text_content = ""
                for _, row in report_data.iterrows():
                    text_content += f"Prompt:\n{row['prompt']}\n\n"
                    text_content += f"Query:\n{row['query']}\n\n"
                    text_content += f"Explanation:\n{row['explanation']}\n\n"
                    text_content += f"Expected Output:\n{row['output']}\n\n"
                    text_content += "-" * 50 + "\n"
                st.download_button(
                    label="Download Text",
                    data=text_content,
                    file_name="sql_query_report.txt",
                    mime="text/plain"
                )
            elif download_format == "PDF":
                pdf_content = generate_pdf_report(report_data)
                st.download_button(
                    label="Download PDF",
                    data=pdf_content,
                    file_name="sql_query_report.pdf",
                    mime="application/pdf"
                )
        # else:
        #  st.warning("No queries were generated yet. Please generate some queries first.")

    elif tabs == "Upload SQL Files":
        if uploaded_files:
            # Clear previous generated_data to avoid duplicates
            st.session_state.generated_data = []
            st.info("Processing uploaded SQL files...")
            for uploaded_file in uploaded_files:
                st.markdown(f"## File: {uploaded_file.name}")
                file_content = uploaded_file.read().decode("utf-8")
                queries = process_sql_file(file_content)

                for idx, query in enumerate(queries, start=1):
                    st.markdown(f"### Query {idx}")
                    st.code(query, language="sql")

                    explanation = generate_explanation(query)
                    expected_output = generate_expected_output(query)

                    st.success("Explanation:")
                    st.markdown(explanation)
                    if explanation:
                        # st.audio(text_to_speech(explanation), format="audio/mp3")
                        audio_bytes, tmp_file_path = text_to_speech(explanation)
                        if audio_bytes:
                         st.audio(audio_bytes, format="audio/mp3")
                        os.unlink(tmp_file_path)  # Clean up
                    # st.audio(text_to_speech(explanation), format="audio/mp3")

                    st.success("Expected Output:")
                    st.markdown(expected_output)

                    # Execute query if database connection is provided
                    executed_results = None
                    if all(db_params.values()):
                        st.info("Executing query...")
                        result = execute_query(query, db_params)
                        if result["status"] == "success":
                            if "data" in result and not result["data"].empty:
                                st.success("Query executed successfully! Displaying results:")
                                st.dataframe(result["data"])
                                executed_results = result["data"]  # Store the DataFrame
                            else:
                                st.success(result["message"])
                                executed_results = result["message"]  # Store the message
                                st.session_state.query_history.append({"query": query, "result": result["message"]})
                        else:
                            st.error(f"Error executing query: {result['message']}")
                            executed_results = result["message"]  # Store
                            st.session_state.query_history.append({"query": query, "result": result["message"]})


                    st.session_state.generated_data.append({
                        "prompt": f"Query {idx} from file {uploaded_file.name}",
                        "query": query,
                        "explanation": explanation,
                        "output": expected_output,
                        "results": executed_results if executed_results is not None else "No results (no DB connection or query failed)"
                    })

            # st.write("Generated Data for Report:", st.session_state.generated_data)
        else:
            st.warning("Please upload a SQL file to get started.")

        # Download Report Section
        if st.session_state.generated_data:
            report_data = pd.DataFrame(st.session_state.generated_data)
            st.subheader("Download Query Report")
            render_download_buttons(report_data)
       

    elif tabs == "Visualize Data":
        st.header("Visualize Data")

        # Initialize session state variables
        if "viz_text_input" not in st.session_state:
            st.session_state.viz_text_input = ""
        if "viz_sql_query" not in st.session_state:
            st.session_state.viz_sql_query = ""
        if "viz_data" not in st.session_state:
            st.session_state.viz_data = None
        if "all_columns" not in st.session_state:
            st.session_state.all_columns = []
        if "numeric_columns" not in st.session_state:
            st.session_state.numeric_columns = []
        if "x_axis" not in st.session_state:
            st.session_state.x_axis = None
        if "y_axis" not in st.session_state:
            st.session_state.y_axis = None
        if "chart_type" not in st.session_state:
            st.session_state.chart_type = "Bar"

        # Text input for plain English query
        text_input = st.text_area(
            "Enter your data request in plain English:",
            placeholder="E.g., Show all employees in the IT department",
            value=st.session_state.viz_text_input,
            key="viz_text_input_area"
        )

        # Update session state with new input and reset only if input changes
        if text_input != st.session_state.viz_text_input:
            st.session_state.viz_text_input = text_input
            st.session_state.viz_sql_query = ""
            st.session_state.viz_data = None
            st.session_state.all_columns = []
            st.session_state.numeric_columns = []
            st.session_state.x_axis = None
            st.session_state.y_axis = None

        if st.button("Generate and Execute Query"):
            # Clear previous state for a fresh run
            st.session_state.viz_sql_query = ""
            st.session_state.viz_data = None
            st.session_state.all_columns = []
            st.session_state.numeric_columns = []
            st.session_state.x_axis = None
            st.session_state.y_axis = None

            if not text_input.strip():
                st.warning("Please enter a valid plain English request.")
            else:
                with st.spinner("Generating SQL Query..."):
                    try:
                        template = """
                            Create a SQL Query snippet using the below text:
                            {text_input}
                            I just want a SQL Query.
                        """
                        formatted_template = template.format(text_input=text_input)
                        response = model.generate_content(formatted_template)
                        sql_query = response.text.strip().lstrip("```sql").rstrip("```")
                        st.session_state.viz_sql_query = sql_query

                        if all(db_params.values()):
                            st.info("Executing query...")
                            result = execute_query(sql_query, db_params)
                            if result["status"] == "success":
                                if "data" in result and not result["data"].empty:
                                    st.session_state.viz_data = result["data"]
                                    st.session_state.all_columns = result["data"].columns.tolist()
                                    st.session_state.numeric_columns = result["data"].select_dtypes(include=["number"]).columns.tolist()
                                    # Set default X/Y if available
                                    if st.session_state.all_columns:
                                        st.session_state.x_axis = st.session_state.all_columns[0]
                                    if st.session_state.numeric_columns:
                                        st.session_state.y_axis = st.session_state.numeric_columns[0]
                                        # Add to query history
                                        st.session_state.query_history.append({
                                    "query": sql_query,
                                    "result": result["data"]
                                       })
                                else:
                                    st.session_state.viz_data = None
                                    # Add to query history
                                    st.session_state.query_history.append({
                                    "query": sql_query,
                                    "result": result["message"]
                                     })
                            else:
                                st.error(f"Error executing query: {result['message']}")
                                st.session_state.viz_data = None
                                # Add to query history
                                st.session_state.query_history.append({
                                "query": sql_query,
                                "result": result["message"]
                            })
                        else:
                            st.warning("No database connection provided. Visualization requires a connected database.")
                            st.session_state.viz_data = None

                    except Exception as e:
                        st.error(f"Error generating or executing query: {e}")
                        st.session_state.viz_data = None

        # Display results and visualization options
        if st.session_state.viz_sql_query:
            st.success("Generated SQL Query:")
            st.code(st.session_state.viz_sql_query, language="sql")

            if st.session_state.viz_data is not None:
                if not st.session_state.viz_data.empty:
                    st.success("Query Results:")
                    st.dataframe(st.session_state.viz_data)

                    # Chart type selection
                    chart_type = st.selectbox(
                        "Select chart type (for SELECT queries)",
                        ["Bar", "Line", "Pie", "Scatter", "Histogram", "Box", "Area"],
                        index=["Bar", "Line", "Pie", "Scatter", "Histogram", "Box", "Area"].index(st.session_state.chart_type),
                        key="chart_type_select"
                    )
                    st.session_state.chart_type = chart_type

                    # X/Y axis selection
                    if st.session_state.all_columns:
                        x_axis = st.selectbox(
                            "Select X-axis column",
                            st.session_state.all_columns,
                            index=st.session_state.all_columns.index(st.session_state.x_axis) if st.session_state.x_axis in st.session_state.all_columns else 0,
                            key="x_axis_select"
                        )
                        y_axis = st.selectbox(
                            "Select Y-axis column (optional)",
                            st.session_state.numeric_columns,
                            index=st.session_state.numeric_columns.index(st.session_state.y_axis) if st.session_state.y_axis in st.session_state.numeric_columns else 0,
                            key="y_axis_select"
                        )

                        # Update session state with selections
                        st.session_state.x_axis = x_axis
                        st.session_state.y_axis = y_axis

                        # Generate and display chart
                        if st.session_state.x_axis:
                            with st.spinner("Generating visualization..."):
                                try:
                                    data = st.session_state.viz_data
                                    if chart_type == "Bar":
                                        fig = px.bar(data, x=x_axis, y=y_axis, title="Bar Chart")
                                    elif chart_type == "Line":
                                        fig = px.line(data, x=x_axis, y=y_axis, title="Line Chart")
                                    elif chart_type == "Pie":
                                        fig = px.pie(data, names=x_axis, values=y_axis, title="Pie Chart")
                                    elif chart_type == "Scatter":
                                        fig = px.scatter(data, x=x_axis, y=y_axis, title="Scatter Plot")
                                    elif chart_type == "Histogram":
                                        fig = px.histogram(data, x=x_axis, title="Histogram")
                                    elif chart_type == "Box":
                                        fig = px.box(data, x=x_axis, y=y_axis, title="Box Plot")
                                    elif chart_type == "Area":
                                        fig = px.area(data, x=x_axis, y=y_axis, title="Area Chart")
                                    st.plotly_chart(fig)
                                    
                                except Exception as e:
                                    st.error(f"Error generating chart: {e}")
                        else:
                            st.warning("Please select an X-axis column to visualize.")
                    else:
                        st.warning("No columns available to visualize.")
                else:
                    st.warning("Query returned no data to visualize.")
            else:
                st.warning("No data available to visualize from the last execution.")
        else:
            st.info("Enter a request and click 'Generate and Execute Query' to start.")






    # Query History tab
    elif tabs == "Query History":
        st.header("Query History 📜")
        if st.session_state.query_history:
            for idx, entry in enumerate(st.session_state.query_history, 1):
                st.markdown(f"**Query {idx}**")
                st.code(entry["query"], language="sql")
                st.write("**Result**:")
                st.write(entry["result"])
                st.markdown("---")
        else:
            st.info("No queries in history yet. Run some queries in the AI Agent or other tabs!")
   





    
    elif tabs == "Database Schema":
     st.header("Database Schema Explorer 🗃️")

     if not all(db_params.values()):
        st.warning("Please connect to a database to view the schema.")
     else:
        schema = fetch_tables(db_params)
        if isinstance(schema, pd.DataFrame) and not schema.empty:
            # Original Graph Displayed First
            st.subheader("Schema Graph")
            G = nx.DiGraph()
            for table in schema["Table Name"]:
                table_data = fetch_table_data(db_params, table)
                G.add_node(table)
                for col in table_data.columns:
                    G.add_edge(table, col)
            try:
                st.graphviz_chart(nx.nx_pydot.to_pydot(G).to_string())
            except OSError as e:
                st.error(f"Graph visualization failed: {e}. Ensure Graphviz is installed and 'dot' is in your PATH.")

            # Interactive Table-Based Schema Explorer
            st.subheader("Schema Overview")
            tables = schema["Table Name"].tolist()
            for table in tables:
                with st.expander(f"Table: {table}", expanded=False):
                    try:

                        if db_params.get("type") == "sqlite":
                            conn = sqlite3.connect(db_params["database"])
                            cursor = conn.cursor()
                            cursor.execute(f"PRAGMA table_info({table})")
                            columns_info = pd.DataFrame(cursor.fetchall(), columns=["cid", "name", "type", "notnull", "default", "pk"])
                            columns_info = columns_info[["name", "type", "notnull", "default", "pk"]]
                            columns_info.columns = ["Field", "Type", "Null", "Default", "Key"]
                            columns_info["Null"] = columns_info["Null"].apply(lambda x: "NO" if x else "YES")
                            columns_info["Key"] = columns_info["Key"].apply(lambda x: "PRI" if x else "")
                        else:
                            conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
                            cursor = conn.cursor()
                            cursor.execute(f"SHOW COLUMNS FROM {table}")
                            columns_info = pd.DataFrame(cursor.fetchall(), columns=["Field", "Type", "Null", "Key", "Default", "Extra"])
                        
                        st.write("Columns:")
                        st.dataframe(columns_info)
                      

                        # Show sample data
                        sample_data = fetch_table_data(db_params, table).head(5)
                        st.write("Sample Data:")
                        st.dataframe(sample_data)

                        # Detect relationships

                        if db_params.get("type") == "sqlite":
                            cursor.execute(f"PRAGMA foreign_key_list({table})")
                            fk_info = cursor.fetchall()
                            relationships = [f"{fk[3]} → {fk[2]}" for fk in fk_info]  # from_column → to_table
                        else:
                            cursor.execute(f"SHOW CREATE TABLE {table}")
                            create_stmt = cursor.fetchone()[1].lower()
                            relationships = []
                            for line in create_stmt.split("\n"):
                                if "foreign key" in line:
                                    fk_col = line.split("`")[1]
                                    ref_table = line.split("references")[1].split("`")[1]
                                    relationships.append(f"{fk_col} → {ref_table}")
                        
                        if relationships:
                            st.write("Relationships:")
                            for rel in relationships:
                                st.write(f"- {rel}")
                        else:
                            st.write("No foreign key relationships detected.")
                        conn.close()
                    except Error as e:
                        st.error(f"Error fetching details for {table}: {e}")

            # Feature: Schema Export
            st.subheader("Export Schema")
            export_format = st.selectbox("Choose export format:", ["SQL Script", "JSON"])
            if st.button("Export"):
                if db_params.get("type") == "sqlite":
                    conn = sqlite3.connect(db_params["database"])
                else:
                    conn = mysql.connector.connect(**{k: v for k, v in db_params.items() if k != "type"})
                cursor = conn.cursor()





                
                if export_format == "SQL Script":
                    sql_script = ""
                    for table in tables:
                        cursor.execute(f"SHOW CREATE TABLE {table}")
                        sql_script += cursor.fetchone()[1] + ";\n\n"
                    st.download_button(
                        label="Download SQL Script",
                        data=sql_script,
                        file_name="schema.sql",
                        mime="text/sql"
                    )
                elif export_format == "JSON":
                    schema_json = {}
                    for table in tables:
                        cursor.execute(f"DESCRIBE {table}")
                        schema_json[table] = [{"column": row[0], "type": row[1]} for row in cursor.fetchall()]
                    st.download_button(
                        label="Download JSON",
                        data=json.dumps(schema_json, indent=2),
                        file_name="schema.json",
                        mime="application/json"
                    )
                conn.close()

            # Feature: Interactive Schema Editor (Enhanced with Remove Options)
            st.subheader("Schema Editor")
            action = st.selectbox("Choose an action:", ["Add Table", "Add Column", "Remove Column", "Remove Table"])
            
            if action == "Add Table":
                table_name = st.text_input("New Table Name:", key="new_table_name")
                column_def = st.text_area("Column Definitions (e.g., id INT PRIMARY KEY, name VARCHAR(255)):", key="new_table_cols")
                if st.button("Generate SQL", key="generate_add_table"):
                    if not table_name or not column_def:
                        st.warning("Please provide a table name and column definitions.")
                    elif not table_name.isalnum():
                        st.warning("Table name should contain only alphanumeric characters (no spaces or special characters).")
                    else:
                        sql = f"CREATE TABLE `{table_name}` ({column_def});"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_add_table"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Table created successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

            elif action == "Add Column":
                table_to_modify = st.selectbox("Select Table:", tables, key="select_table_to_modify_add")
                column_name = st.text_input("New Column Name:", key="new_column_name")
                column_type = st.text_input("Column Type (e.g., VARCHAR(255), INT):", key="new_column_type")
                if st.button("Generate SQL", key="generate_add_column"):
                    if not table_to_modify or not column_name or not column_type:
                        st.warning("Please provide a table, column name, and column type.")
                    else:
                        safe_column_name = f"`{column_name}`" if any(c in column_name for c in "+-*/() ") else column_name
                        sql = f"ALTER TABLE `{table_to_modify}` ADD COLUMN {safe_column_name} {column_type};"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_add_column"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Column added successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

            elif action == "Remove Column":
                table_to_modify = st.selectbox("Select Table:", tables, key="select_table_to_modify_remove_col")
                if table_to_modify:
                    conn = mysql.connector.connect(**db_params)
                    cursor = conn.cursor()
                    cursor.execute(f"SHOW COLUMNS FROM `{table_to_modify}`")
                    columns = [row[0] for row in cursor.fetchall()]
                    conn.close()
                    column_to_remove = st.selectbox("Select Column to Remove:", columns, key="column_to_remove")
                    if st.button("Generate SQL", key="generate_remove_column"):
                        if not column_to_remove:
                            st.warning("Please select a column to remove.")
                        else:
                            safe_column_name = f"`{column_to_remove}`" if any(c in column_to_remove for c in "+-*/() ") else column_to_remove
                            sql = f"ALTER TABLE `{table_to_modify}` DROP COLUMN {safe_column_name};"
                            st.code(sql, language="sql")
                            st.session_state["generated_sql"] = sql
                
                    if "generated_sql" in st.session_state and st.button("Execute", key="execute_remove_column"):
                        with st.spinner("Executing query..."):
                            result = execute_query(st.session_state["generated_sql"], db_params)
                            if result["status"] == "success":
                                st.success("Column removed successfully!")
                                del st.session_state["generated_sql"]
                                st.rerun()
                            else:
                                st.error(f"Error executing query: {result['message']}")

            elif action == "Remove Table":
                table_to_remove = st.selectbox("Select Table to Remove:", tables, key="select_table_to_remove")
                if st.button("Generate SQL", key="generate_remove_table"):
                    if not table_to_remove:
                        st.warning("Please select a table to remove.")
                    else:
                        sql = f"DROP TABLE `{table_to_remove}`;"
                        st.code(sql, language="sql")
                        st.session_state["generated_sql"] = sql
                
                if "generated_sql" in st.session_state and st.button("Execute", key="execute_remove_table"):
                    with st.spinner("Executing query..."):
                        result = execute_query(st.session_state["generated_sql"], db_params)
                        if result["status"] == "success":
                            st.success("Table removed successfully!")
                            del st.session_state["generated_sql"]
                            st.rerun()
                        else:
                            st.error(f"Error executing query: {result['message']}")

        else:
            st.error("Error fetching schema: " + schema.get("message", "Unknown error"))

  


    elif tabs == "Learn SQL":
     st.header("Learn SQL with AI, Videos, and Quizzes 🎥📝")
    
    # SQL Learning Roadmap
     st.subheader("SQL Learning Roadmap 🛤️")
     roadmap = [
        "1️⃣ **Introduction to Databases & SQL** – Learn about relational databases and SQL basics.",
        "2️⃣ **Basic Queries** – SELECT, WHERE, ORDER BY, LIMIT.",
        "3️⃣ **Filtering & Aggregation** – GROUP BY, HAVING, COUNT, AVG, SUM.",
        "4️⃣ **Joins & Relationships** – INNER JOIN, LEFT JOIN, RIGHT JOIN, FULL JOIN.",
        "5️⃣ **Subqueries & Nested Queries** – Writing efficient subqueries.",
        "6️⃣ **Advanced SQL Functions** – CASE, COALESCE, Common Table Expressions (CTEs).",
        "7️⃣ **Indexes & Performance Optimization** – Indexing, Query Optimization.",
        "8️⃣ **Stored Procedures & Triggers** – Automating SQL tasks.",
        "9️⃣ **SQL for Data Analysis** – Window functions, analytical queries.",
        "🔟 **Practice & Real-World Applications** – Work on projects & real datasets."
    ]
     for step in roadmap:
        st.markdown(step)

    # Tabs for Learning and Quiz
     learn_tab, quiz_tab = st.tabs(["Learn SQL", "Take a Quiz"])

    # Learn SQL Tab
     with learn_tab:
        st.markdown("<h4 style='color: #333333;'>Ask a question about SQL (e.g., 'Explain SQL JOIN')</h4>", unsafe_allow_html=True)
        query = st.text_input("Enter your SQL question here", key="learn_sql_input")
        want_voice_for_topic = st.checkbox("Generate voice explanation for SQL Topic Content", value=False)

        if st.button("Generate", key="learn_generate"):
            if query:
                with st.spinner("Fetching SQL Topic Content..."):
                    try:
                        # Generate SQL topic content
                        topic_template = """
                            Provide a detailed explanation of the SQL topic: {query}
                            Focus on the content and concepts related to this topic.
                        """
                        formatted_template = topic_template.format(query=query)
                        response = model.generate_content(formatted_template)
                        sql_topic_content = response.text.strip()

                        if sql_topic_content:
                            st.success("SQL Topic Content:")
                            st.markdown(sql_topic_content)
                            if want_voice_for_topic:
                                with st.spinner("Generating voice explanation..."):
                                    audio_bytes, tmp_file_path = text_to_speech(sql_topic_content)
                                    if audio_bytes:
                                        st.audio(audio_bytes, format="audio/mp3")
                                    if tmp_file_path and os.path.exists(tmp_file_path):
                                        os.unlink(tmp_file_path)

                        # Fetch YouTube videos
                        videos = search_youtube_videos(YOUTUBE_API_KEY, query)
                        if videos:
                            st.write("Here are some tutorial videos for you:")
                            for video in videos:
                                st.markdown(f"[{video['title']}]({video['url']})")
                        else:
                            st.write("No videos found. Try another query.")

                        # Generate additional explanation
                        explanation = generate_explanation(query)
                        if explanation:
                            st.success("Additional Explanation:")
                            st.markdown(explanation)
                            with st.spinner("Generating voice explanation..."):
                                audio_bytes, tmp_file_path = text_to_speech(explanation)
                                if audio_bytes:
                                    st.audio(audio_bytes, format="audio/mp3")
                                if tmp_file_path and os.path.exists(tmp_file_path):
                                    os.unlink(tmp_file_path)

                    except Exception as e:
                        st.error(f"Error processing your request: {e}")
            else:
                st.warning("Please enter an SQL topic or question.")

    # Quiz Tab
     with quiz_tab:
        st.subheader("SQL Quiz Time! 📝")
        quiz_topic = st.text_input("Enter an SQL topic for the quiz (e.g., 'SQL Joins')", key="quiz_topic_input")
        
        # Add option to select number of questions
        num_questions = st.slider(
            "How many questions would you like in your quiz?",
            min_value=1,
            max_value=10,
            value=5,  # Default value
            step=1,
            key="num_questions_slider"
        )

        if "quiz_questions" not in st.session_state:
            st.session_state.quiz_questions = []
        if "user_answers" not in st.session_state:
            st.session_state.user_answers = {}
        if "quiz_submitted" not in st.session_state:
            st.session_state.quiz_submitted = False

        if st.button("Generate Quiz", key="generate_quiz"):
            if quiz_topic:
                with st.spinner(f"Generating {num_questions} quiz questions..."):
                    try:
                        quiz_template = """
                            Create a quiz with {num_questions} multiple-choice questions about the SQL topic: {quiz_topic}.
                            For each question, provide:
                            - The question text
                            - Four answer options (labeled a, b, c, d)
                            - The correct answer (e.g., 'a', 'b', 'c', or 'd')
                            Format each question as follows:
                            QX: [Question text]
                            a) [Option a]
                            b) [Option b]
                            c) [Option c]
                            d) [Option d]
                            Correct Answer: [correct answer letter]
                        """
                        formatted_quiz_template = quiz_template.format(num_questions=num_questions, quiz_topic=quiz_topic)
                        response = model.generate_content(formatted_quiz_template)
                        quiz_content = response.text.strip()

                        # Parse quiz content into a structured format
                        questions = []
                        lines = quiz_content.split("\n")
                        current_question = {}
                        for line in lines:
                            line = line.strip()
                            if line.startswith("Q"):
                                if current_question:
                                    questions.append(current_question)
                                current_question = {"question": line[3:].strip(), "options": {}, "correct_answer": None}
                            elif line.startswith(("a)", "b)", "c)", "d)")):
                                letter = line[0]
                                current_question["options"][letter] = line[3:].strip()
                            elif line.startswith("Correct Answer:"):
                                current_question["correct_answer"] = line.split(":")[1].strip()
                        if current_question:
                            questions.append(current_question)

                        # Ensure we have the requested number of questions (trim or warn if fewer)
                        if len(questions) < num_questions:
                            st.warning(f"Only {len(questions)} questions were generated instead of {num_questions} due to API limitations.")
                        st.session_state.quiz_questions = questions[:num_questions]  # Trim to requested number if more are generated
                        st.session_state.user_answers = {}
                        st.session_state.quiz_submitted = False
                        st.success(f"Quiz with {len(st.session_state.quiz_questions)} questions generated for '{quiz_topic}'!")
                    except Exception as e:
                        st.error(f"Error generating quiz: {e}")
            else:
                st.warning("Please enter an SQL topic for the quiz.")

        # Display Quiz Questions
        if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
            st.markdown("### Your Quiz")
            for idx, q in enumerate(st.session_state.quiz_questions, 1):
                st.write(f"**Q{idx}: {q['question']}**")
                selected_answer = st.radio(
                    f"Select your answer for Q{idx}",
                    options=list(q["options"].keys()),
                    format_func=lambda x: f"{x} {q['options'][x]}",
                    key=f"q_{idx}"
                )
                st.session_state.user_answers[idx] = selected_answer

            if st.button("Submit Answers", key="submit_quiz"):
                st.session_state.quiz_submitted = True
                st.rerun()

        # Evaluate and Display Results
        if st.session_state.quiz_submitted and st.session_state.quiz_questions:
            st.markdown("### Quiz Results")
            score = 0
            total = len(st.session_state.quiz_questions)
            for idx, q in enumerate(st.session_state.quiz_questions, 1):
                user_answer = st.session_state.user_answers.get(idx)
                correct_answer = q["correct_answer"]
                is_correct = user_answer == correct_answer
                if is_correct:
                    score += 1
                st.write(f"**Q{idx}: {q['question']}**")
                st.write(f"Your Answer: {user_answer} {q['options'][user_answer]}")
                st.write(f"Correct Answer: {correct_answer} {q['options'][correct_answer]}")
                st.write(f"Result: {'✅ Correct' if is_correct else '❌ Incorrect'}")
                st.markdown("---")

            percentage = (score / total) * 100
            st.markdown(f"**Your Score: {score}/{total} ({percentage:.2f}%)**")
            if percentage == 100:
                st.success("Perfect score! You're an SQL master! 🎉")
            elif percentage >= 70:
                st.success("Great job! You're getting the hang of it! 😊")
            else:
                st.info("Nice try! Keep practicing to improve your SQL skills! 📚")

            if st.button("Try Another Quiz", key="reset_quiz"):
                st.session_state.quiz_questions = []
                st.session_state.user_answers = {}
                st.session_state.quiz_submitted = False
                st.rerun()

        elif not st.session_state.quiz_questions:
            st.info("Enter a topic, choose the number of questions, and click 'Generate Quiz' to start!")



    elif tabs == "SQL Playground":
     st.header("SQL Playground 🛠️")
     user_query = st.text_area("Write your SQL query here:", height=200, placeholder="e.g., SELECT * FROM employees")
     
     if st.button("Run Query"):
        if not user_query.strip():
            st.warning("Please enter a valid SQL query.")
        elif not all(db_params.values()):
            st.warning("Please connect to a database to execute queries.")
        else:
            with st.spinner("Executing query..."):
                result = execute_query(user_query, db_params)
                if result["status"] == "success":
                    if "data" in result and not result["data"].empty:
                        st.success("Query executed successfully! Displaying results:")
                        st.dataframe(result["data"])
                        # Add to query history
                        st.session_state.query_history.append({
                            "query": user_query,
                            "result": result["data"]
                        })
                    else:
                        st.success(result["message"])
                        # Add to query history
                        st.session_state.query_history.append({
                            "query": user_query,
                            "result": result["message"]
                        })
                else:
                    st.error(result["message"])
                    # Add to query history even if it fails
                    st.session_state.query_history.append({
                        "query": user_query,
                        "result": result["message"]
                    })
        schema = fetch_tables(db_params) if db_params.get("database") else None
        if isinstance(schema, pd.DataFrame) and not schema.empty:
            tables = schema["Table Name"].tolist()
            suggestions = [
                f"Try asking: 'Show all records from {tables[0]}'",
                f"Try asking: 'Count rows in {tables[0]}'",
                f"Try asking: 'Show {tables[0]} where a column > some value'"
            ]
    elif tabs == "AI Agent":
    # Header
     st.header("Chat with Your SQL Buddy 🤖")

    # Initialize session state unconditionally
     st.session_state.setdefault("chat_sessions", {
        "Chat 1": [{"role": "agent", "content": "Hey, SQL adventurer! I’m here to help you conquer those queries. What’s on your mind? 😄", "timestamp": datetime.now().strftime("%H:%M:%S")}]
    })
     st.session_state.setdefault("active_chat", "Chat 1")
     st.session_state.setdefault("editing_message", None)
     st.session_state.setdefault("last_response", "")
     st.session_state.setdefault("audio_data", None)
     st.session_state.setdefault("audio_file_path", None)
     st.session_state.setdefault("user_input", "")
     st.session_state.setdefault("send_triggered", False)
     st.session_state.setdefault("uploaded_file_content", None)
     st.session_state.setdefault("uploaded_image_content", None)
     st.session_state.setdefault("conversation_mode", False)
     st.session_state.setdefault("use_web", False)
     st.session_state.setdefault("use_deep_search", False)
     st.session_state.setdefault("show_suggestions", False)

    # Chat History
     chat_container = st.container()
     with chat_container:
        current_history = st.session_state.chat_sessions[st.session_state.active_chat]
        for idx, msg in enumerate(current_history):
            with st.chat_message(msg["role"], avatar="🤖" if msg["role"] == "agent" else "👤"):
                if msg["role"] == "user" and st.session_state.editing_message == idx:
                    edited_content = st.text_input("Tweak your message here:", value=msg["content"], key=f"edit_input_{idx}")
                    col1, col2 = st.columns(2)
                    with col1:
                        if st.button("Save", key=f"save_{idx}"):
                            current_history[idx]["content"] = edited_content
                            current_history[idx]["timestamp"] = datetime.now().strftime("%H:%M:%S")
                            if idx + 1 < len(current_history):
                                del current_history[idx + 1:]  # Remove subsequent messages
                            response = agent_response(edited_content, db_params, model, current_history[:idx], st.session_state.uploaded_file_content, st.session_state.uploaded_image_content, st.session_state.use_web, st.session_state.use_deep_search)
                            response = f"Got it! Here’s my updated take: {response} – How’s that? 😊"
                            current_history.append({"role": "agent", "content": response, "timestamp": datetime.now().strftime("%H:%M:%S")})
                            st.session_state.last_response = response
                            st.session_state.editing_message = None
                            st.rerun()
                    with col2:
                        if st.button("Cancel", key=f"cancel_{idx}"):
                            st.session_state.editing_message = None
                            st.rerun()
                else:
                    st.markdown(f"{msg['content']}  *({msg['timestamp']})*")
                    if "image" in msg and msg["image"]:
                        st.image(msg["image"], caption="Your Cool Image", use_container_width=True)
                    if "file_content" in msg and msg["file_content"]:
                        st.markdown(f"**Your File Says:**\n```\n{msg['file_content']}\n```")
                    if msg["role"] == "agent":
                        st.button("Copy", key=f"copy_{idx}", on_click=lambda x=msg["content"]: pyperclip.copy(x))
                    if msg["role"] == "user" and st.session_state.editing_message is None:
                        if st.button("Edit", key=f"edit_{idx}"):
                            st.session_state.editing_message = idx
                            st.rerun()

    # Spinner Placeholder
     spinner_placeholder = st.empty()

    # Input Area
     st.markdown("---")
     col_input, col_send = st.columns([10, 1])
     with col_input:
        user_input = st.text_input("Chat with me about SQL – what’s up?", key="grok_chat_input", label_visibility="visible", value=st.session_state.user_input, on_change=lambda: st.session_state.update(user_input=st.session_state.grok_chat_input))
     with col_send:
        if st.button("▶", key="send_button"):
            st.session_state.send_triggered = True

    # Input Tools
     col_file, col_space1, col_image, col_space2, col_web, col_deep, col_voice, col_think, col_convo = st.columns([1, 1, 1, 1, 1, 1, 1, 1, 1])
     with col_file:
        uploaded_file = st.file_uploader("Drop a SQL file here!", type=["sql"], key="file_uploader", label_visibility="visible")
        if uploaded_file:
            st.session_state.uploaded_file_content = uploaded_file.read().decode("utf-8")
            st.write("File uploaded! I’ll weave it into our chat.")
     with col_space1:
        pass
     with col_image:
        uploaded_image = st.file_uploader("Got a pic? Share it!", type=["png", "jpg", "jpeg"], key="image_uploader", label_visibility="visible")
        if uploaded_image:
            st.session_state.uploaded_image_content = uploaded_image.getvalue()
            st.write("Image ready! Let’s see how it fits in.")
     with col_space2:
        pass
     with col_web:
        st.session_state.use_web = st.checkbox("Web Magic", key="web_checkbox", value=st.session_state.use_web, label_visibility="visible")
     with col_deep:
        st.session_state.use_deep_search = st.checkbox("Deep Dive", key="deep_checkbox", value=st.session_state.use_deep_search, label_visibility="visible")
     with col_voice:
        if st.button("Voice 🎤", key="voice_button"):
            with st.spinner("Ear on – talk to me!"):
                recognizer = sr.Recognizer()
                with sr.Microphone() as source:
                    audio = recognizer.listen(source, timeout=5)
                try:
                    user_input = recognizer.recognize_google(audio)
                    st.write(f"You said: '{user_input}' – Awesome, let’s chat about it!")
                    current_history.append({"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%H:%M:%S")})
                    response = agent_response(user_input, db_params, model, current_history[:-1], st.session_state.uploaded_file_content, st.session_state.uploaded_image_content, st.session_state.use_web, st.session_state.use_deep_search)
                    response = f"Love hearing your voice! Here’s my reply: {response} – What’s next, buddy? 😄"
                    current_history.append({"role": "agent", "content": response, "timestamp": datetime.now().strftime("%H:%M:%S")})
                    st.session_state.last_response = response
                    st.session_state.user_input = ""
                    st.session_state.uploaded_file_content = None
                    st.session_state.uploaded_image_content = None
                    st.rerun()
                except sr.UnknownValueError:
                    st.error("Oops, I missed that! Could you speak up a bit?")
                except Exception as e:
                    st.error(f"Yikes, something hiccupped: {e}. Let’s try again!")
     with col_think:
        if st.button("Brainstorm 💡", key="think_button"):
            st.session_state.show_suggestions = True
     with col_convo:
        st.session_state.conversation_mode = st.checkbox("Chat Mode", key="convo_checkbox", value=st.session_state.conversation_mode, label_visibility="visible")

    # Process Input
     if st.session_state.send_triggered and user_input:
        with spinner_placeholder:
            with st.spinner("Pondering... 🤓" if not st.session_state.use_deep_search else "Diving deep... 🔍"):
                message = {"role": "user", "content": user_input, "timestamp": datetime.now().strftime("%H:%M:%S")}
                if st.session_state.uploaded_image_content:
                    message["image"] = st.session_state.uploaded_image_content
                if st.session_state.uploaded_file_content:
                    message["file_content"] = st.session_state.uploaded_file_content
                current_history.append(message)
                response = agent_response(user_input, db_params, model, current_history[:-1], st.session_state.uploaded_file_content, st.session_state.uploaded_image_content, st.session_state.use_web, st.session_state.use_deep_search)
                if st.session_state.conversation_mode:
                    response = f"Hmm, let’s chat about that! {response} – What’s your next thought? 😊"
                else:
                    response = f"Here’s my take on it: {response} – Cool, right? What’s up next? 😄"
                
                # Typing animation
                placeholder = st.empty()
                typed_response = ""
                for char in response:
                    typed_response += char
                    placeholder.markdown(typed_response)
                    time.sleep(0.01)
                placeholder.markdown(response)

                current_history.append({"role": "agent", "content": response, "timestamp": datetime.now().strftime("%H:%M:%S")})
                st.session_state.last_response = response
                st.session_state.user_input = ""
                st.session_state.send_triggered = False
                st.session_state.uploaded_file_content = None
                st.session_state.uploaded_image_content = None
                st.rerun()

    # Session Controls
     st.markdown("---")
     col_chat, col_new, col_clear, col_export, col_speak = st.columns(5)
     with col_chat:
        chat_options = list(st.session_state.chat_sessions.keys())
        st.selectbox("Pick a Chat", chat_options, index=chat_options.index(st.session_state.active_chat), key="chat_switch", on_change=lambda: st.session_state.update(active_chat=st.session_state.chat_switch))
     with col_new:
        if st.button("New Chat", key="new_chat"):
            new_chat_id = f"Chat {len(st.session_state.chat_sessions) + 1}"
            st.session_state.chat_sessions[new_chat_id] = [{"role": "agent", "content": "Hey, SQL adventurer! I’m here to help you conquer those queries. What’s on your mind? 😄", "timestamp": datetime.now().strftime("%H:%M:%S")}]
            st.session_state.active_chat = new_chat_id
            st.session_state.last_response = ""
            st.session_state.user_input = ""
            st.session_state.use_web = False
            st.session_state.use_deep_search = False
            st.session_state.conversation_mode = False
            st.session_state.show_suggestions = False
            st.rerun()
     with col_clear:
        if st.button("Reset Chat", key="clear_chat"):
            st.session_state.chat_sessions[st.session_state.active_chat] = [{"role": "agent", "content": "Chat reset! Ready for a fresh start – what’s your next move? 😊", "timestamp": datetime.now().strftime("%H:%M:%S")}]
            st.session_state.last_response = ""
            st.session_state.user_input = ""
            st.session_state.use_web = False
            st.session_state.use_deep_search = False
            st.session_state.conversation_mode = False
            st.session_state.show_suggestions = False
            st.rerun()
     with col_export:
        if st.button("Export Chat", key="export_chat"):
            chat_history = st.session_state.chat_sessions[st.session_state.active_chat]
            text_content = "\n".join([f"{msg['role'].upper()}: {msg['content']} ({msg['timestamp']})" for msg in chat_history])
            st.download_button("Save Chat", data=text_content, file_name=f"{st.session_state.active_chat}.txt", mime="text/plain", key="download_button")
     with col_speak:
        if st.button("Speak", key="speak_button"):
            if st.session_state.last_response:
                with st.spinner("Making some noise for you... 🎙️"):
                    audio_bytes, tmp_file_path = text_to_speech(st.session_state.last_response)
                    st.session_state.audio_data = audio_bytes
                    st.session_state.audio_file_path = tmp_file_path
                if st.session_state.audio_data:
                    st.audio(st.session_state.audio_data, format="audio/mp3")
                    if st.session_state.audio_file_path and os.path.exists(st.session_state.audio_file_path):
                        os.unlink(st.session_state.audio_file_path)
                        st.session_state.audio_file_path = None
            else:
                st.warning("I’ve got nothing to say yet! Toss me a question first! 😉")

     # Enhanced Suggestions
     st.markdown("---")
     if st.session_state.show_suggestions:
      suggestions = generate_suggestions(db_params)
      if isinstance(suggestions, list) and suggestions:
        st.markdown("**Let’s Spark Some Ideas!**")
        enhanced_suggestions = []
        schema = fetch_tables(db_params) if db_params.get("database") else None
        if schema is not None and isinstance(schema, pd.DataFrame) and not schema.empty:
            tables = schema["Table Name"].tolist()
            for table in tables:
                table_data = fetch_table_data(db_params, table)
                if table_data is not None and not table_data.empty:
                    cols = table_data.columns.tolist()
                    enhanced_suggestions.extend([
                        f"How about peeking at `{table}`? Try: `SELECT * FROM {table}`",
                        f"Count the fun in `{table}` with: `SELECT COUNT(*) FROM {table}`",
                        f"Filter `{table}` like a pro: `SELECT {cols[0]} FROM {table} WHERE {cols[1]} > 10`" if len(cols) > 1 else f"Explore `{table}` with `SELECT * FROM {table}`",
                        f"Get fancy with `{table}`: `SELECT {cols[0]}, SUM({cols[1]}) FROM {table} GROUP BY {cols[0]}`" if len(cols) > 1 else f"Dive into `{table}`!"
                    ])
        else:
            enhanced_suggestions = [
                "Let’s build something! Try: `CREATE TABLE friends (id INT, name VARCHAR(50))`",
                "Ask me: `Show me a JOIN example` – I’ll whip one up!",
                "Feeling curious? Say: `What’s a subquery?`"
            ]
        sampled_suggestions = random.sample(enhanced_suggestions, min(3, len(enhanced_suggestions))) if enhanced_suggestions else []
        for idx, suggestion in enumerate(sampled_suggestions):
            if st.button(f"{suggestion}", key=f"suggestion_{idx}_{suggestion[:20]}"):
                st.session_state.user_input = suggestion.split("Try: ")[1] if "Try: " in suggestion else suggestion
                st.session_state.send_triggered = True
                st.rerun()
     else:
        st.write("No database connected yet? No worries – ask me anything SQL-related! 😊")
  


if __name__ == "__main__":
    main()









