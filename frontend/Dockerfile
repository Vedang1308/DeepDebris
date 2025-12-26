FROM python:3.9-slim

WORKDIR /app

# Install system dependencies for PyDeck/cffi if needed (often git is needed)
RUN apt-get update && apt-get install -y git build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Add plotly manually if not in requirements (it was added via command line)
RUN pip install --no-cache-dir -r requirements.txt plotly

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0"]
