# Agentic Governance

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Overview

Agentic Governance is an AI framework that can effectively detect and manage risks associated with LLMs for a given use-case. The framework leverages agents to identify risks tailored to a specific use case, generate drift and risk monitors, and establish real-time monitoring functions for LLMs. By integrating these capabilities, our approach aims to provide a comprehensive risk management framework that addresses the unique requirements of each LLM application.

## Architecture

<img width="1262" alt="image" src="https://github.ibm.com/ai-gov-model-risk/agentic-governance/assets/380720/90a729e8-e95e-4383-a36b-ffc84a81287e">

## Installation

This project targets python version ">=3.11, <3.12". You can download specific versions of python here: https://www.python.org/downloads/

1. Set up `conda` or any python virtual environment and install Agentic Governance
   ```
   git clone git@github.com:IBM/risk-atlas-nexus-demos.git
   cd risk-atlas-nexus-demos/agentic-governance
   conda create -n agentic_governance python=3.11
   conda activate agentic_governance
   pip install -e .
   ```
2. Please follow the instructions at https://github.com/IBM/risk-atlas-nexus#installation to install `Risk Atlas Nexus` library

3. Update the config variables and inference engine params in the server config file. Start your LLM server viz. ollama, vllm. Update LLM server credentials in `cli/server_config.yaml`.

   - `nano cli/server_config.yaml`

4. Start the Agentic Governance server

   - `python cli/server.py --config-file cli/server_config.yaml --host localhost --port 8000`

5. Start the Agentic Governance client
   - `python cli/client.py --host localhost --port 8000`

## License

Agentic Governance is under Apache 2.0 license.

## IBM ❤️ Open Source AI

Agentic Governance has been brought to you by IBM. Please contact [Risk Atlas Nexus](mailto:risk-atlas-nexus@ibm.com) Team for any query.
