import streamlit as st
import requests

# FastAPI server URL
API_URL_RAG = "http://localhost:8000"
API_URL_QUERY = "http://localhost:8001"

# Streamlit app for uploading context and interacting with the LLM
st.title("Class 11 Physics - Contextual Chat with LLM")

# Sidebar for PDF Upload
st.sidebar.header("Upload NCERT PDF as Context")
uploaded_file = st.sidebar.file_uploader("Choose a PDF file", type="pdf")

# Upload PDF to the FastAPI server
if uploaded_file is not None:
    files = {'file': uploaded_file}
    response = requests.post(f"{API_URL_RAG}/upload_pdf", files=files)

    if response.status_code == 200:
        st.sidebar.success("PDF content has been uploaded successfully!")
    else:
        st.sidebar.error("Failed to upload PDF content.")

# Chat Window
st.header("Ask a Question")

# Input box for user query
user_input = st.text_input("Enter your question:")

# Button to submit the query
if st.button("Ask"):
    if user_input:
        # Send the query to the FastAPI agent endpoint
        query_data = {"query": user_input}
        response = requests.post(f"{API_URL_QUERY}/agent", json=query_data)
        print(response.json())
        if response.status_code == 200:
            # Display the response from the agent
            result = response.json()
            if "answer" in result:
                st.success(f"Answer: {result['answer']}")
            else:
                st.error("Failed to retrieve the answer.")           
        else:
            st.error("Failed to retrieve the answer.")
    else:
        st.warning("Please enter a question.")

# Option to clear the context
if st.sidebar.button("Delete Context"):
    response = requests.delete(f"{API_URL}/delete_context")
    if response.status_code == 200:
        st.sidebar.success("Context has been deleted.")
    else:
        st.sidebar.error("Failed to delete context.")
