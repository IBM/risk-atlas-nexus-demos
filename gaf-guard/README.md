# GAF-Guard

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Overview

GAF-Guard is an AI framework that can effectively detect and manage risks associated with LLMs for a given use-case. The framework leverages agents to identify risks tailored to a specific use case, generate drift and risk monitors, and establish real-time monitoring functions for LLMs. By integrating these capabilities, our approach aims to provide a comprehensive risk management framework that addresses the unique requirements of each LLM application.

## Architecture

![90a729e8-e95e-4383-a36b-ffc84a81287e](https://github.com/user-attachments/assets/f0546c3d-cf95-49c8-8112-21308bf6f7e6)

## Documentation

See the [**GAF Guard Wiki**](https://github.com/IBM/risk-atlas-nexus-demos/wiki/GAF-Guard) for full documentation, installation guide, operational details and other information.

## Installation and Running the CLI App

This project targets python version ">=3.11, <3.12". You can download specific versions of python here: https://www.python.org/downloads/

1. Set up `conda` or any python virtual environment and install GAF-Guard
   ```
   git clone git@github.com:IBM/risk-atlas-nexus-demos.git
   cd risk-atlas-nexus-demos/gaf-guard
   conda create -n gaf-guard python=3.11
   conda activate gaf-guard
   pip install -e .
   ```

2. Update the config variables and inference engine params in the server config file. Start your LLM server viz. ollama, vllm. Update LLM server credentials in `app/cli/server_config.yaml`.

   - `nano app/cli/server_config.yaml`

3. Start the GAF-Guard server

   - `python app/cli/server.py --config-file app/cli/server_config.yaml --host localhost --port 8000`

4. Start the GAF-Guard client
   - `python app/cli/client.py --host localhost --port 8000`

## License

GAF-Guard is under Apache 2.0 license.

## IBM ❤️ Open Source AI

GAF-Guard has been brought to you by IBM. Please contact [Risk Atlas Nexus](mailto:risk-atlas-nexus@ibm.com) Team for any query.
