import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from groq import Groq

app = FastAPI()

# Initialize Groq client
client = Groq(api_key="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")

# Sample weather API function
def get_weather_info(city):
    print(city)
    api_key = "a70f3bc06d1e1fa9ccdaf3d52412068a"
    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}"
    response = requests.get(url)
    print(response.json())
    if response.status_code == 200:
        weather_data = response.json()
        return f"Current temperature in {city} is {weather_data['main']['temp']} K."
    else:
        raise HTTPException(status_code=500, detail="Failed to retrieve weather information")


# API request body model
class QueryRequest(BaseModel):
    query: str

# Custom prompt to instruct LLM to categorize the query
def custom_prompt_to_classify(query):
    prompt = (
        f"You are a helpful assistant trained in Class 11 Physics (NCERT) and general knowledge, including weather queries. "
        f"Classify the following user query into one of the following categories: "
        f"1. Class 11 Physics (related to NCERT topics), "
        f"2. Weather (general weather-related queries), or "
        f"3. General (any other questions that are unrelated to Class 11 Physics or weather). "
        f"Based on the classification, respond with the corresponding category by only returning either of the following words [physics, weather, general] accordingly and no other word. If label is weather, also return the city the person is talking about like this: [weather, delhi] for example. The user query is: {query}"
    )
    return prompt

@app.post("/agent")
async def agent(request: QueryRequest):
    user_input = request.query
    
    # Step 1: Classify the query with the Groq LLM
    custom_prompt = custom_prompt_to_classify(user_input)
    classification_response = client.chat.completions.create(
        messages=[{"role": "user", "content": custom_prompt}],
        model="llama3-8b-8192",  # Adjust this to the model you're using
    )
    
    # Extract LLM's classification
    classification = classification_response.choices[0].message.content.strip().lower()
    print("classification :: ", classification)
    # Step 2: Decide action based on classification
    if "physics" in classification.lower():
        # Invoke the RAG system for context search
        rag_response = requests.post("http://localhost:8000/query", json={"query": user_input})
        if rag_response.status_code == 200:
            context = rag_response.json().get("results", [])
            # Append the top 5 context snippets to the original query
            context_string = " ".join(context[:5])
            print("context_string :: ", context_string)
            enriched_query = f"{user_input} {context_string}"
            
            # Query the LLM again with the enriched context
            final_llm_response = client.chat.completions.create(
                messages=[{"role": "user", "content": enriched_query}],
                model="llama3-8b-8192",
            )
            # print("final_llm_response :: ", final_llm_response)
            
            # Return the final enriched response
            final_answer = final_llm_response.choices[0].message.content
            return {"query": user_input, "answer": final_answer, "context_used": context}
        else:
            raise HTTPException(status_code=500, detail="Failed to retrieve data from the RAG system")
    
    elif "weather" in classification.lower():
        # Invoke the weather API
        weather_info = get_weather_info(classification.split(",")[-1].strip())
        return {"query": user_input, "answer": weather_info}
    
    else:
        # General queries: Return the LLM's answer directly without any context or external system invocation
        general_llm_response = client.chat.completions.create(
            messages=[{"role": "user", "content": user_input}],
            model="llama3-8b-8192",
        )
        final_answer = general_llm_response.choices[0].message.content
        return {"query": user_input, "answer": final_answer}

# Health check endpoint
@app.get("/health")
async def health():
    return {"status": "Server is up and running"}
