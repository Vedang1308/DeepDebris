#!/bin/bash
echo "1. Health Check:"
curl -s -o /dev/null -w "%{http_code}" http://127.0.0.1:8000/
echo -e "\n\n2. TLE (ISS):"
curl -s "http://127.0.0.1:8000/tle/25544" | head -c 100
echo -e "...\n\n3. Live Weather:"
curl -s "http://127.0.0.1:8000/weather/live"
echo -e "\n\n4. Risk List:"
curl -s "http://127.0.0.1:8000/risks"
echo -e "\n\n5. Analyze Risk (Mock):"
curl -s -X POST "http://127.0.0.1:8000/analyze_risk" -H "Content-Type: application/json" -d '{"sat_id": "25544", "debris_id": "DEB_123", "tca": "2025-12-27T00:00:00"}'
echo -e "\n\n6. OrbitGPT:"
curl -s -X POST "http://127.0.0.1:8000/chat" -H "Content-Type: application/json" -d '{"query": "Is the ISS safe?"}'
