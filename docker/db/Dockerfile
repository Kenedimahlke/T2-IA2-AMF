FROM postgres:16

ENV POSTGRES_DB=automind \
    POSTGRES_USER=automind \
    POSTGRES_PASSWORD=automind

COPY automind-data-extraction/data/processed/vehicles_final.csv /seed/vehicles_final.csv
COPY docker/db/init.sql /docker-entrypoint-initdb.d/init.sql
