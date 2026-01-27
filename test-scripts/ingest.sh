BASE_URL=http://localhost:8000

# Ingest a new doc
curl -v -XPOST "$BASE_URL/ingest" -d '{"user_input":"We have a transport optimization requirement from many of our logistics clients. the goal is to evaluate all the lanes that the carriers could possibly use and provide the most cost-effective ones in terms of the criteria below. #1 - lowest lane pricing, #2 - highest carrier delivery success rate, and #3 - shortest distance. this story will be assigned to development team to implement these criteria"}' -H "Content-Type: application/json" | jq

