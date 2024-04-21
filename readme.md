<!-- To install the packages -->

1. pip install virtualenv
2. py -m venv venv
3. source venv/bin/activate
4. pip install streamlit pypdf2 langchain python-dotenv faiss-cpu openai huggingface_hub
5. pip freeze > requirements.txt
6. pip install -r requirements.txt
7. streamlit run app.py

<!-- Other links to explore -->

1. instructor-embedding: very slow, need high gpu power but free embeddings
2.
