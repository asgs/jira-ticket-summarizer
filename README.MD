# Summary
This is an attempt to build an AI ~Agent~ **Chatbot** that will read Jira ticket descriptions for new tickets and summarize the contents in a neat and concise way based on Jira data ingested in the past

# Tech Stack
- Python3
- FastAPI (with uvicorn)
- sentence-transformers (with MiniLM for embedding and Qwen2.5 LM for Instruct purposes)
- pandas
- pyarrow
- ChromaDB
- tiktoken
- ReactJS for frontend
- Babel for JSX

# Data Setup
This agent needs external data sources to build the knowledge base required to augment the responses from the LM. Currently, two publicly accessible data sources were tried from Kaggle

- [Apache JIRA Issues](https://www.kaggle.com/datasets/tedlozzo/apaches-jira-issues?select=issues.csv)
- [Jira Dataset](https://www.kaggle.com/datasets/cesaranasco/jira-dataset)

whichever dataset is used, the expectation here is it's a CSV file with column names configured in the main.py file

# How to Run?
1. Install python3
2. Clone this git repo
3. Navigate to the root of the repo and run `source .jts/bin/activate` to activate/switch to this virtual environment
4. Run `pip install -r requirements.txt` to setup the required dependencies. You may have to upgrade pip if you get an `AssertionError`
5. Download the data sources mentioned in the previous section and unzip to the source-data directory
6. Run `uvicorn main:app` to ingest sample data and spin up the FastAPI server

Once the ingestion completes, the app is ready to accept your inputs via the endpoint http://localhost:8000/summarize?input=xyz

# Sample Test

## Ingest a new document

Run `curl -v -XPOST "http://localhost:8000/ingest?input=We%20have%20a%20transport%20optimization%20requirement%20from%20many%20of%20our%20logistics%20clients.%20the%20goal%20is%20to%20evaluate%20all%20the%20lanes%20that%20the%20carriers%20could%20possibly%20use%20and%20provide%20the%20most%20cost-effective%20ones%20in%20terms%20of%20the%20criteria%20below.%20%231%20-%20lowest%20lane%20pricing%2C%20%232-%20%20highest%20carrier%20delivery%20success%20rate%2C%20and%20%233%20-%20shortest%20distance.%20this%20story%20will%20be%20assigned%20to%20development%20team%20to%20implement%20these%20criteria"|jq`

## Summarize a query

Run `curl -v -XPOST "http://localhost:8000/summarize?input=This%20story%20is%20to%20track%20the%20development%20efforts%20required%20to%20implement%20the%20transport%20optimization%20requirement.%20Break%20down%20the%20tasks%20involved%20with%20a%20high%2Dlevel%20estimate%20to%20build%20and%20test%20each%20and%20every%20criterion%20to%20achieve%20the%20optimization"|jq`

# TODOs
1. Provide options to switch to a cloud-model accepting necessary configs like URL, Auth Token/API Key, etc
2. [Done] - ~~Build a basic GUI for user to submit their queries and view the responses returned~~
3. [Done] - ~~Provide an option to user to retrieve top-k results. The default is 3 now~~
4. [**WIP**] - Provide an endpoint to ingest one or more documents or reingest one or more datasources on demand
5. Provide an endpoint to save/load pre-trained models on demand
6. [Done] - ~~Move the local LM name, dataset location, CSV column names to a configuration file~~
7. Model Quantization
8. Model Fine/Prompt-tuning
9. Configure ChromaDB to use hnsw indexing
