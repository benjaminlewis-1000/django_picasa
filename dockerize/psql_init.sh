
#!/bin/bash
set -e

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    CREATE USER benjamin PASSWORD 'meVc8p1myOC5CAfcTYPAhFJuT';
    ALTER USER benjamin CREATEDB;
    CREATE DATABASE picasa;
    GRANT ALL PRIVILEGES ON DATABASE picasa TO benjamin;
    ALTER DATABASE picasa OWNER TO benjamin;
    GRANT ALL ON DATABASE picasa TO benjamin;
EOSQL
