FROM ubuntu:latest
FROM mcr.microsoft.com/mssql/server:2017-latest

RUN apt-get update && apt-get install -y software-properties-common && add-apt-repository universe && apt-get update && apt-get install -y mysql-client

ENV MYSQL_ROOT_PASSWORD=root
ENV DB_NAME baseball
    

# Copy SQL script to initialize database
COPY ./joshi.sql /sql/joshi.sql
COPY ./baseball.sql ./sql/baseball.sql


RUN apt-get update && apt-get install -y bash
COPY ./please.sh /script/please.sh
RUN chmod +x /script/please.sh && \
    echo 'export DATABASE=baseball' >> ~/.bashrc && \
    echo 'export USER=root' >> ~/.bashrc && \
    echo 'export PASSWORD=root' >> ~/.bashrc

ENTRYPOINT ["/bin/bash"]
CMD ["./script/please.sh"]

EXPOSE 3306
