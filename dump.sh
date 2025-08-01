docker exec -it rag-hpc-rc-postgres-1 pg_dump -U user -d vector_db > ./db/dump.sql
