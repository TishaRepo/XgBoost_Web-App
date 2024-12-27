# Start with the base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files into the container
COPY . .

# Expose the port (not strictly necessary for Cloud Run but good practice)
EXPOSE 8080

# Specify the command to run your application
CMD ["python", "app.py"]
