FROM python:3.10
WORKDIR /app
RUN mkdir plot
# Install dependencies
RUN apt-get update && apt-get install -y mariadb-client build-essential
RUN pip install --upgrade pip
# Copy scripts and SQL file
COPY FINALSQL.sql .
COPY FINALSQL.sh .
COPY FINALMAIN.py .
# Set executable permissions
RUN chmod +x FINALSQL.sh
RUN chmod +x FINALSQL.py
CMD ["./FINALSQL.sh"]
