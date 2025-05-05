FROM python:3.7-slim

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy the rest of the application
COPY . .

# Expose port 5000
EXPOSE 5000

# Start the server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "server:app"] 