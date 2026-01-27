BASE_URL=http://localhost:8000

# Summarize a query
curl -v -XPOST "$BASE_URL/summarize" -d '{"user_input":"This story is to track the development efforts required to implement the transport optimization requirement. Break down the tasks involved with a high-level estimate to build and test each and every criterion to achieve the optimization"}' -H "Content-Type: application/json" | jq

