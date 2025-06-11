# Neo4j instructions

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Overview
Ingest data from Risk Atlas Nexus into an Neo4j database

## 1 Get started
### 1.1 Set up Neo4j

1. Pull docker image
```
docker pull neo4j:latest
```

We want to put the output of the risk atlas nexus cypher queries the examples/ai-risk-ontology.cypher file in the local folder ./examples.

Start a Neo4j container and mount the ./examples folder inside the container:

2. Run image
``` 
docker run --rm --name <containerID/name> --volume /<your_own_path>/risk-atlas-nexus-demos/neo4j-db/examples:/examples
--publish=7474:7474 \
--publish=7687:7687 \
--env NEO4J_AUTH=neo4j/<your_own_password> \ 
neo4j:2025.05.0
```

## 2. Populate the db
### 1.2 exec into the container
use the Cypher Shell tool 

```
docker exec --interactive --tty 391641719c00 cypher-shell -u neo4j -p riskatlasnexus
```
then use the `:source` command to run the example script in the cypher-shell

```
:source /examples/ai-risk-ontology.cypher
```

## 3. Now what?

Now you can query the data or use as you like. 

To count all nodes: `MATCH (n) RETURN count(n)`
To count all relationships: `MATCH ()-[r]->() RETURN count(r)`

### 3.1 simple visualisation of graph
You can connect with the http://localhost:7474/browser/.  In the browser you can run 
``` 
CALL db.schema.visualization()
```

## Hints
Check your container statuses
```
docker ps -a 
```

