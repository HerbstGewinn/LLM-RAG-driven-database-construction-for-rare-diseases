import requests
from langchain_ollama.chat_models import ChatOllama
from langchain_ollama import OllamaEmbeddings
import os
from dotenv import load_dotenv

def establish_server_connection():
    # Authentication details

    load_dotenv()
    user = os.getenv("ollama_user")
    password = os.getenv("ollama_pw")
    protocol = "https"
    hostname = "chat.cosy.bio"
    host = f"{protocol}://{hostname}"
    auth_url = f"{host}/api/v1/auths/signin"
    api_url = f"{host}/ollama"
    account = {
        'email': user,
        'password': password
    }
    auth_response = requests.post(auth_url, json=account)

    jwt = auth_response.json()["token"]
    headers = {"Authorization": "Bearer " + jwt}
    return api_url, headers

def get_llm(api_url, headers, model="llama3.2:latest", format="json"):
    return ChatOllama(model=model, temperature=0, 
                          base_url=api_url, 
                          client_kwargs={"headers": headers}, 
                          format=format,
                          num_ctx=25000)

def get_embedding_model(api_url, headers, model="nomic-embed-text"):
    return OllamaEmbeddings(base_url=api_url, 
                                       model=model, 
                                       client_kwargs={"headers": headers})