# Stage 1: Use an official Python runtime as a parent image
# Using a specific version ensures reproducibility.
FROM python:3.9-slim

# Stage 2: Set the working directory inside the container
WORKDIR /app

# Stage 3: Copy dependency definition files
# We copy these first to leverage Docker's layer caching.
# If these files don't change, Docker won't reinstall dependencies on every build.
COPY requirements.txt .

# Stage 4: Install dependencies
# We use --no-cache-dir to keep the image size smaller.
RUN pip install --no-cache-dir -r requirements.txt

# Stage 5: Copy the application source code and model into the container
# Copy the 'src' directory into the container's '/app/src'
COPY src/ ./src
# Copy the trained model into the container's '/app'
COPY model.pkl .

# Stage 6: Expose the port the app runs on
# This tells Docker that the container listens on port 8000.
EXPOSE 8000

# Stage 7: Define the command to run the application
# This is the command that will be executed when the container starts.
# It's the same command we used to run the API locally.
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]