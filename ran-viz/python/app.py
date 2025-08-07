# Convenience script to export a json graph version or IBM AI risk atlas from the yaml
# in src/risk_atlas_nexus/data/knowledge_graph

from risk_atlas_nexus import RiskAtlasNexus
from util.json_graph_dumper import JSONGraphDumper

OUTPUT_FILE = "../front-end/demo/public/ai-risk-ontology.json"
SCHEMA_FILE = "https://raw.githubusercontent.com/IBM/risk-atlas-nexus/refs/heads/main/src/risk_atlas_nexus/ai_risk_ontology/schema/ai-risk-ontology.yaml"

ran = RiskAtlasNexus()

# export IBM AI risk atlas from graph
container = ran._ontology

def generate_graph_data():

    with open(OUTPUT_FILE, "+tw", encoding="utf-8") as output_file:
        print(JSONGraphDumper(schema_path=SCHEMA_FILE).dumps(container), file=output_file)
        output_file.close()


if __name__ == "__main__":
    generate_graph_data()