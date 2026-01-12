# Summary

This is an attempt to build an AI Agent that will read Jira ticket descriptions for new tickets and summarize the text in a neat and concise way based on Jira data created in the past

# Tech Stack
Python
FastAPI (with uvicorn)
sentence-transformers (with MiniLM)
pandas
ChromaDB
LexRank
tiktoken

# How to Run

1. Install and setup python3
2. Cloned this git repo
3. Navigate to the root of the repo and run `source .jts/bin/activate` to activate/switch to this virtual environment
4. Run `pip install -r requirements.txt` to setup the required dependencies. You may have to upgrade pip if you get an `AssertionError`
5. Run `unzip source-data/src-data.zip -d source-data` to unzip the source data
6. Run `uvicorn main:app` to ingest sample data and spin up the FastAPI server

Now, it's ready to accept your inputs via the endpoint http://localhost:8000/summarize?input=<xyz>

