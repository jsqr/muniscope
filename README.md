# muniscope

Automated analysis of municipal codes for legal epidemiology.

## Getting started

### Setting up a local postgres database

#### 1. Install postgres.

* MacOS: I find [Postgres.app](https://www.postgresql.org/download/macosx/) is an easy
way to get set up for local development instances on the Mac.

* Linux, etc.: [This site](https://www.postgresql.org/download/) gives instructions for
different distros, and other operating systems. You will also need to install the
`vector` extension.

#### 2. Start psql and create a user

```sh
psql -U postgres
```

then

```sql
CREATE USER muni WITH PASSWORD 'muni';
```

#### 2. Create a database and set up permissions

```sql
CREATE DATABASE muni;
```

then

```sh
psql -U postgres -d muni
```

then

```sql
CREATE EXTENSION IF NOT EXISTS vector;
GRANT CREATE ON SCHEMA public TO muni;
```

#### 3. Run the initialization script

```sh
. scripts/reset.sh localhost muni muni
```
