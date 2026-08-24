FROM python:3.12-slim

# opencv (pulled in transitively by mediapipe) is built against X11/GL libs,
# and mediapipe's own native core is linked against GLES/EGL for optional GPU
# delegates -- both need these present to even dlopen, even though this app
# only runs CPU inference.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libgomp1 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "Backend/server.py"]
