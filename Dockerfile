FROM python:latest
WORKDIR /colorsearch
RUN pip3 install requests numpy pillow scikit-learn fastapi uvicorn
COPY --link . ./
CMD ["uvicorn", "main:app", "--host=0.0.0.0", "--port=8000"]
