# Databricks Free Edition: An End-to-End Applied Data Science Workflow
## Academic Presentation Summary for Berkeley Immersion Students

### Overview
This session presents an end-to-end demonstration using the recently launched Databricks Free Edition, a no-cost, non-commercial use version of the Databricks platform for students, educators, and hobbyists. The entire workflow showcases how to seamlessly build and operationalize both traditional Machine Learning (ML) and agentic solutions, leveraging open-source integrations. The demo is hands-on, with live code and direct GitHub repo integration in-session.

### Key Learning Objectives
- Experience Databricks’ modern open platform from data ingestion through productionization
- Understand open standards, open source, and cloud-native best practices in a reproducible, academic context
- Explore both declarative data engineering and agentic orchestration patterns

### End-to-End Workflow Components
1. Free Account Setup
Step-by-step walk-through of signing up for Databricks Free Edition
Direct GitHub code repository connection and synchronization within the Databricks workspace
2. Data Ingestion & ETL Pipeline Construction
Selection and exploration of a publicly available dataset
Visual ETL transformation with Lakeflow Designer, demonstrating pipeline construction
Compatibility with Spark Declarative Pipelines and open-source workflow standards
3. Open Data Governance with Unity Catalog
Data wrangling and cleansing workflows
Creating and managing a Unity Catalog (open source) for structured, secured, and discoverable data assets
4. Exploratory Data Analysis (EDA)
Interactive EDA using Python (pandas, matplotlib, seaborn) within Databricks notebooks
Additional EDA techniques using Genie integration for rapid data insight generation
5. Machine Learning & Agentic Workflows
Feature engineering and baseline ML model development using open source libraries
Orchestration and management with MLflow 3.0, including experiment tracking and model evaluation (open source)
Introduction to agentic workflows—demonstrating orchestration with agents alongside traditional ML pipelines
6. Model Deployment & Open Standards Access
Promotion of the trained ML model to an endpoint for consumption
Registration and access to the model via MCP (Model Connectivity Protocol) adhering to open standards
7. Interactive Application Integration
Development of an interactive Databricks App using JavaScript (Streamlit or React)
Real-time demonstration of agentic solution interaction, powered by the deployed ML endpoint

### Academic Relevance
This demo encapsulates the state-of-the-art in reproducible, open science workflows, from raw data through actionable endpoints. Graduate students gain practical experience with both engineering and applied data science lifecycles, all accessible at no cost on the Databricks Free Edition. The approach fosters both technical competence and confidence to build and share robust, production-grade analytics and agentic solutions.

### Summary Table
| Stage | Tools/Tech Used | Open Source/Standard? | Key Academic Takeaway |
|-------|-----------------|-----------------------|-----------------------|
| Setup | Databricks Free, GitHub | Yes | Platform access & reproducibility |
| ETL/Data | Lakeflow Designer, Spark | Yes | Visual, scalable pipeline design |
| Governance | Unity Catalog | Yes | Data discoverability/security |
| EDA | Python, Genie | Yes | Modern open EDA methods |
| ML/Agent | MLflow 3, Agents | Yes | Model and orchestration lifecycle |
| Deployment | Model Serving, MCP, Playground | Yes | Open model serving with function calling |
| App Integration | Streamlit/React | Yes | App-building with live ML agents |

The session is designed to inspire, inform, and empower early-career researchers with direct exposure to end-to-end, open-source-powered analytics and AI workflows.

## Instructions for getting started -> Navigation to first notebook.

All the code is located in the lakehouse-iot-platform directory. This end-to-end workflow requires specific data assets to be available in the cloud.

To run the code and start the workflow, first execute the `00-IOT-wind-turbine-introduction-DI-platform` notebook. This notebook explains the project and it contains a single line of code that:

1. Creates the catalog and schema where all metadata will be stored.
2. Loads the required data into the designated AWS S3 bucket.

Once the notebook finishes running, you can verify the data load by navigating to Catalog → My Organization in the left panel and selecting `main.dbdemos_iotturbine.Volumes`. Seeing the data here confirms it has been loaded into S3 and is ready for ingestion by Databricks.


Steps of the project:

1. ETL Process: The ETL step for your project involves ingesting, cleaning, and transforming data, ultimately producing eight Delta Live Tables. All related code is located in the `01-Data-ingestion/01.1-DLT-Wind-Turbine-SQL` notebook. However, you cannot run this notebook directly, as it contains STREAMING TABLES defined in the Lakeflow Declarative Pipeline format.

   To start the ETL process:
    - Go to the Jobs & Pipelines section in Databricks.
    - Point the job or pipeline to this notebook and control execution from the orchestration UI, not from the notebook itself.

   Key details:

    - The ETL process extracts data from source systems, cleans and transforms it, and loads it into production tables.
    - Using Delta Live Tables (DLT) and the declarative pipeline approach allows you to manage both batch and streaming data, ensuring reliability and scalability in data processing.
    - The workflow is managed outside the notebook to ensure proper orchestration, dependency handling, and monitoring, as is best practice with declarative pipelines and production ETL in the Databricks Lakehouse environment.

2. EDA Step:
Once the data is prepared and ready for analysis, the next phase is exploratory data analysis (EDA). The code for this stage can be found in the `02-Data-Science-ML/02.1-EDA` notebook. Here, data scientists will explore trends, visualize data distributions, and generate insights that inform the modeling process.

3. Model Creation Step:
In this phase, multiple models are developed and each step of experimentation is logged in MLflow. Unity Catalog offers seamless integration with MLflow, simplifying experiment tracking and model management. Within the `02-Data-Science-ML/02.2-predictive_model_creation` notebook, you will:

    - Create and run different model experiments.
    - Record experiment results in MLflow.
    - Register the final, chosen model as a production model in the MLflow model registry for easier deployment and governance.

4. Model Deployment Step: 
In this phase, the registered model is deployed as an endpoint to enable inference. You will use the `02-Data-Science-ML/02.3-model_deployment` notebook for this step. The deployment workflow typically includes:

    - Deploying the Model: The notebook guides you through deploying the chosen model from the MLflow model registry to a serving endpoint, making it accessible for real-time or batch predictions.

    - Batch Inference: After deployment, the same notebook demonstrates how to perform batch inference on a table—specifically, using one of the tables available in your catalog. This allows you to generate predictions at scale and store results back into your Lakehouse environment.

5. Generative AI Steps:
In this phase, you will leverage notebooks in the `03-Generative-AI` directory to add intelligent, AI-powered capabilities to your platform.

    `03.0-ai-tools` notebook: Develop a simple function designed to be used as a tool by an AI agent. This establishes the basic building blocks for agent-driven workflows.

    `03.1-ai-tools-iot-turbine-prescriptive-maintenance` notebook: This notebook utilizes Databricks' generative AI capabilities for advanced analytics and automation tasks. The key actions include:

    - Turbine Predictor Tool: Create a specialized tool that leverages AI to predict turbine failures, supporting prescriptive maintenance strategies.

    - Parsing and Saving Unstructured Data: Extract and store relevant text from unstructured sources such as PDF documents, making it available for downstream AI-powered search and analytics.

    - Vector Search Endpoint Creation: Set up a vector search endpoint and assign a dedicated vector search index. This enables high-performance, semantic search across your dataset.

    - AI Tool for Vector Search: Develop a tool that interfaces with the vector search index, enabling agents to retrieve contextually relevant information based on semantic similarity rather than traditional keyword search.

    The remaining notebooks in the 03-Generative-AI directory serve as practical guides for creating and deploying agents using Databricks Apps. They provide step-by-step instructions and examples, demonstrating how to leverage Databricks’ platform tools to build, configure, and operationalize AI agents within your environment. These resources are designed to help you extend your workflow, enabling advanced automation and custom agent functionalities tailored to specific industrial IoT scenarios.


Beyond the core workflow, the project includes additional resources found in several directories. The `04-Data-goverenence` directory offers best practices for access control, data lineage, and compliance, helping to ensure responsible data management. In `05-BI-Data-warehousing`, you'll find guidance on using Databricks SQL for analytics, dashboard creation, and business intelligence tasks. The `06-workflow-orchestration` directory provides information on how to schedule, automate, and monitor your data and ML pipelines effectively. Altogether, these materials serve as guides to help improve data security, analytics, and operational efficiency within your platform.

