
#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER benjamin PASSWORD 'UUDKrIXzGlz1Uneqnm7l3JZ0E';
    ALTER USER benjamin CREATEDB;
    CREATE DATABASE picasa;
    GRANT ALL PRIVILEGES ON DATABASE picasa TO benjamin;
EOSQL
