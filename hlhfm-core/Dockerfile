FROM python:3.9-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY hlhfm.py .
COPY main.py .

# Expose port
EXPOSE 8000

# Run the application
CMD ["python", "main.py"]
