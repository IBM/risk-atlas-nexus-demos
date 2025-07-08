# GAF-Guard

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)](https://www.apache.org/licenses/LICENSE-2.0) [![](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/) <img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>

## Overview

GAF-Guard is an AI framework that can effectively detect and manage risks associated with LLMs for a given use-case. The framework leverages agents to identify risks tailored to a specific use case, generate drift and risk monitors, and establish real-time monitoring functions for LLMs. By integrating these capabilities, our approach aims to provide a comprehensive risk management framework that addresses the unique requirements of each LLM application.

A quick overview of the motivation and demonstration of the framework is here:
https://www.youtube.com/watch?v=M4JSkdFg6I0

## Architecture

![90a729e8-e95e-4383-a36b-ffc84a81287e](https://github.com/user-attachments/assets/f0546c3d-cf95-49c8-8112-21308bf6f7e6)

## Agent Communication Protocol (ACP)

GAF Guard utilizes the [**ACP**](https://github.com/i-am-bee/acp) protocol to facilitate communication between the GAF Guard Client and Server. Any ACP-compliant client can connect to GAF Guard Server to submit tasks and retrieve outputs. By adopting the ACP protocol, GAF Guard enables seamless integration with otherwise siloed agents, promoting the creation of interoperable agentic systems that support easier collaboration and broader ecosystem connectivity.

For more information on ACP, visit the official [site](https://agentcommunicationprotocol.dev/introduction/welcome) or check out this [blog post](https://www.ibm.com/think/topics/agent-communication-protocol).

## Risk Atlas Nexus

GAF Guard leverages resources and APIs from **Risk Atlas Nexus** to support key functions such as Risk Taxonomy, Risk Identification, Risk Questionnaire Predictions, Risk Assessment, and other AI Governance tasks. Risk Atlas Nexus serves as a central platform to unify and streamline diverse tools and resources related to the governance of foundation models. 

Check out the official repo of [Risk Atlas Nexus](https://github.com/IBM/risk-atlas-nexus).

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

5. Run benchmark
   - `python app/cli/run_benchmark.py --host localhost --port 8000 --trial-dir trials`

## Referencing the project

If you use GAF-Guard in your projects, please consider citing the following:

```bib
@article{gafguard2025,
      title={GAF-Guard: An Agentic Framework for Risk Management and Governance in Large Language Models},
      author={Seshu Tirupathi, Dhaval Salwala, Elizabeth M. Daly and Inge Vejsbjerg},
      year={2025},
      eprint={2507.02986},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2507.02986}
}
```

## License

GAF-Guard is under Apache 2.0 license.

## IBM ❤️ Open Source AI

GAF-Guard has been brought to you by IBM. Please contact [Risk Atlas Nexus](mailto:risk-atlas-nexus@ibm.com) Team for any query.
