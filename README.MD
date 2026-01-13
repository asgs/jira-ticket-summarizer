# Summary
This is an attempt to build an AI Agent that will read Jira ticket descriptions for new tickets and summarize the contents in a neat and concise way based on Jira data ingested in the past

# Tech Stack
- Python3
- FastAPI (with uvicorn)
- sentence-transformers (with MiniLM)
- pandas
- pyarrow
- ChromaDB
- LexRank
- tiktoken

# Data Setup
This agent needs external data sources to build the knowledge base required to augment the responses from the LM. Currently, two publicly accessible data sources were tried from Kaggle

- [Apache JIRA Issues](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues?select=issues.csv)
- [Jira Dataset](https://www.kaggle.com/datasets/cesaranasco/jira-dataset)

whichever dataset is used, the expectation here is it's a CSV file with column names configured in the main.py file

# How to Run?
1. Install and setup python3
2. Cloned this git repo
3. Navigate to the root of the repo and run `source .jts/bin/activate` to activate/switch to this virtual environment
4. Run `pip install -r requirements.txt` to setup the required dependencies. You may have to upgrade pip if you get an `AssertionError`
5. Run `unzip source-data/src-data.zip -d source-data` to unzip the source data
6. Run `uvicorn main:app` to ingest sample data and spin up the FastAPI server

Now, it's ready to accept your inputs via the endpoint http://localhost:8000/summarize?input=<xyz>

# TODOs
1. Provide options to switch to a cloud-model accepting necessary configs like URL, Auth Token/API Key, etc
2. Build a basic GUI for user to submit their queries and view the responses returned
3. Provide an option to user to retrieve top-k results. The default is 1
4. Provide an endpoint to ingest one or more documents or reingest one or more datasources on demand
5. Provide an endpoint to save/load pre-trained models on demand
6. Move the local LM name, dataset location, CSV column names to a configuration file
