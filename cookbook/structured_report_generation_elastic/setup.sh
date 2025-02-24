#!/bin/bash
set -e

# Function to print messages
log() {
  echo "[`date +"%Y-%m-%d %H:%M:%S"`] $1"
}

# Check if container exists and remove it if it does
log "Checking for existing es_test container..."
if docker ps -a | grep -q es_test; then
  log "Found existing es_test container. Removing it..."
  docker rm -f es_test
fi

# 1. Start Elasticsearch server in Docker (single-node for testing) with security disabled
log "Starting Elasticsearch Docker container..."
docker run -d --name es_test -p 9200:9200 \
  -e "discovery.type=single-node" \
  -e "xpack.security.enabled=false" \
  docker.elastic.co/elasticsearch/elasticsearch:8.9.0

# Wait until Elasticsearch is responsive
log "Waiting for Elasticsearch to be available on http://localhost:9200 ..."
until curl -s http://localhost:9200 >/dev/null; do
  sleep 1
done
log "Elasticsearch is up and running."

# 2. Create an index named "weather" with mappings and populate with sample data

log "Creating the 'weather' index with mappings..."
curl -s -X PUT "http://localhost:9200/weather?pretty" -H 'Content-Type: application/json' -d'
{
  "mappings": {
    "properties": {
      "city": { "type": "keyword" },
      "country": { "type": "keyword" },
      "temperature": { "type": "float" },
      "condition": { "type": "text" },
      "timestamp": { "type": "date" }
    }
  }
}
'
log "'weather' index created."

log "Indexing dummy weather data..."

# New York, USA
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "New York",
  "country": "USA",
  "temperature": 22.5,
  "condition": "Sunny",
  "timestamp": "2025-02-02T12:00:00Z"
}
'

# London, UK
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "London",
  "country": "UK",
  "temperature": 16.0,
  "condition": "Cloudy",
  "timestamp": "2025-02-02T12:05:00Z"
}
'

# Paris, France
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Paris",
  "country": "France",
  "temperature": 18.3,
  "condition": "Rainy",
  "timestamp": "2025-02-02T12:10:00Z"
}
'

# Tokyo, Japan
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Tokyo",
  "country": "Japan",
  "temperature": 24.0,
  "condition": "Clear",
  "timestamp": "2025-02-02T12:15:00Z"
}
'

# Berlin, Germany
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Berlin",
  "country": "Germany",
  "temperature": 14.7,
  "condition": "Overcast",
  "timestamp": "2025-02-02T12:20:00Z"
}
'

# Sydney, Australia
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Sydney",
  "country": "Australia",
  "temperature": 26.4,
  "condition": "Sunny",
  "timestamp": "2025-02-02T12:25:00Z"
}
'

# Moscow, Russia
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Moscow",
  "country": "Russia",
  "temperature": -5.0,
  "condition": "Snowy",
  "timestamp": "2025-02-02T12:30:00Z"
}
'

# Beijing, China
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Beijing",
  "country": "China",
  "temperature": 10.2,
  "condition": "Smoggy",
  "timestamp": "2025-02-02T12:35:00Z"
}
'

# Rio de Janeiro, Brazil
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Rio de Janeiro",
  "country": "Brazil",
  "temperature": 28.1,
  "condition": "Humid",
  "timestamp": "2025-02-02T12:40:00Z"
}
'

# Cape Town, South Africa
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Cape Town",
  "country": "South Africa",
  "temperature": 20.3,
  "condition": "Windy",
  "timestamp": "2025-02-02T12:45:00Z"
}
'

# Mumbai, India
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "Mumbai",
  "country": "India",
  "temperature": 30.5,
  "condition": "Humid",
  "timestamp": "2025-02-02T12:50:00Z"
}
'

# San Francisco, USA
curl -s -X POST "http://localhost:9200/weather/_doc?pretty" -H 'Content-Type: application/json' -d'
{
  "city": "San Francisco",
  "country": "USA",
  "temperature": 17.8,
  "condition": "Foggy",
  "timestamp": "2025-02-02T12:55:00Z"
}
'

# Refresh the index to make sure documents are searchable immediately.
log "Refreshing the index..."
curl -s -X POST "http://localhost:9200/weather/_refresh?pretty"

# 3. Test: List all indices to verify the "weather" index is up
log "Testing: Listing all indices..."
curl -X GET "http://localhost:9200/_cat/indices?v&pretty"

log "Elasticsearch setup complete. The 'weather' index is populated with expanded dummy weather data."

# Uncomment the following line to stop and remove the container after testing
# docker rm -f es_test
