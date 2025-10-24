# Convenience script to export a json graph version or IBM AI risk atlas from the yaml
# in src/risk_atlas_nexus/data/knowledge_graph

from risk_atlas_nexus import RiskAtlasNexus
from risk_atlas_nexus.toolkit.logging import configure_logger
from util.json_graph_dumper import JSONGraphDumper

logger = configure_logger(__name__)

OUTPUT_FILE = "graph_export/jsonld/ai-risk-ontology.json"
SCHEMA_FILE = "src/risk_atlas_nexus/ai_risk_ontology/schema/ai-risk-ontology.yaml"

ran = RiskAtlasNexus()

# export IBM AI risk atlas to latex
container = ran._ontology

with open(OUTPUT_FILE, "+tw", encoding="utf-8") as output_file:
    print(JSONGraphDumper(schema_path=SCHEMA_FILE).dumps(container), file=output_file)
    output_file.close()
    logger.info("Graph Json output complete")
