# frontent/Dockerfile

FROM python:3.6

COPY requirements.txt ./requirements.txt
RUN chmod 755 ./requirements.txt

WORKDIR /dashboard

RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY . /dashboard

EXPOSE 8501

ENTRYPOINT ["streamlit","run"]
#CMD ["app.py"]
#CMD ["dashboard/dashboard_app.py"]
#CMD ["streamlit", "run", "./dashboard/dashboard_app.py"]
#CMD ["uvicorn", "dashboard.main:app", "--host", "0.0.0.0", "--port", "15400"]
ENTRYPOINT ["streamlit", "run"]

CMD ["app.py"]