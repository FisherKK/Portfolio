FROM python:3.6.7
WORKDIR /app
VOLUME ["/app"]
COPY . /app
RUN pip install --upgrade pip && \
    pip install -r requirements.txt
EXPOSE 8887
CMD ["gunicorn", "--bind", "0.0.0.0:8887", "--workers", "12", "wsgi:app"]

