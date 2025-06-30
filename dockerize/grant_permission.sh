
# These commands must be run to grant permisison in the database.
# docker exec -it db_picasa sh  ; # then...

su postgres
psql picasa -c "GRANT ALL on ALL TABLES IN SCHEMA public to benjamin;"
psql picasa -c "GRANT ALL ON ALL SEQUENCES IN SCHEMA public to benjamin;"
psql picasa -c "GRANT ALL ON ALL FUNCTIONS IN SCHEMA public to benjamin;"


#    GRANT ALL PRIVILEGES ON DATABASE picasa TO benjamin;
#    ALTER DATABASE picasa OWNER TO benjamin;
#    GRANT ALL ON DATABASE picasa TO benjamin;

