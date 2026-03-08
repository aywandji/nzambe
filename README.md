# Nzambe RAG

Nzambe is a Retrieval-Augmented Generation (RAG) application designed to provide accurate, context-aware answers to queries based on the primary holy books: the Bible, Quran, and Torah.

## Architecture

The following diagram illustrates the flow from a user query to the final response within the AWS-hosted infrastructure.

```mermaid
graph TB
    subgraph Client_Side [User Interface]
        User((User))
    end

    subgraph AWS_Cloud [AWS Cloud Environment]
        ALB[AWS Load Balancer]

        subgraph Compute_Layer [Compute]
            ECS[FastAPI Server <br/> ECS Cluster]
            Lambda[AWS Lambda <br/> Processing Job]
        end

        subgraph Data_Storage [Storage]
            S3_Docs[(S3 Bucket <br/> Documents)]
            S3_Index[(S3 Vector Index)]
        end
    end

    subgraph Observability [Monitoring & Traces]
        Langfuse[[Langfuse]]
    end

    subgraph External_Services [External APIs]
        LLM[LLM API <br/> e.g. Bedrock/OpenAI]
    end

    %% Ingestion Pipeline (The New Part)
    S3_Docs -->|S3 Event Trigger| Lambda
    Lambda -->|Embeddings API| LLM
    LLM -.->|Vectors| Lambda
    Lambda -->|Save Vectors| S3_Index

    %% Interaction Flow (Read Path)
    User -->|Request| ALB
    ALB --> ECS

    %% RAG Logic
    ECS <-->|Retrieve Context| S3_Index
    ECS <-->|Prompt/Completion| LLM

    %% Observability Flow (Async Traces)
    ECS -.->|Telemetry & Traces| Langfuse
    Lambda -.->|Ingestion Traces| Langfuse

    %% Response
    ECS -->|Final Answer| ALB
    ALB --> User

    %% Styling
    classDef aws fill:#FF9900,stroke:#232F3E,color:white;
    classDef storage fill:#3F8624,stroke:#232F3E,color:white;
    classDef external fill:#7AA116,stroke:#232F3E,color:white;
    classDef obs fill:#000000,stroke:#6b7280,color:white;

    class ALB,ECS,Lambda aws;
    class S3_Docs,S3_Index storage;
    class LLM external;
    class Langfuse obs;
```

## Design Tradeoffs

Building a RAG application involving large sacred texts involves several critical tradeoffs:

- **Local vs. Cloud LLMs**: We use Ollama for local development to minimize costs during development, while the production environment is structured to interface with robust LLM APIs to handle the complexity of theological interpretation.

- **Vector Storage**: Instead of a dedicated vector database service (like Pinecone), we store indices in Amazon S3. This significantly reduces infrastructure costs for a low-traffic application while maintaining sufficient retrieval speed for our current document scale.

- **Accuracy vs. Latency**: The current implementation prioritizes simple retrieval. While adding reranking and BM25 hybrid search (planned) increases latency, it is necessary to handle the linguistic nuances found in ancient texts.

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/aywandji/nzambe.git
cd nzambe
```

### 2. Local Setup

**Download Data**: For the Bible in English, you can use:

```bash
curl https://www.gutenberg.org/cache/epub/10/pg10.txt --output .debug/data/bible_en.txt
```

**Configuration**: The baseline configuration is defined at `config/base.yaml`. Update it according to your preferences. The expected values are defined in pydantic settings `nzambe.config.NzambeSettings`. If you want to have a special config for your staging/prod envs, you must update the corresponding files. Also if you want a special config for your local development, you must create and update a `local.yaml` file.

**Environment Variables**: Create a `.env` file to add the environment variables that will be used during local development. You can follow the `example.env` structure. This file will be read while launching the RAG server.

**Third-Party Services**:

- **Langfuse**: Self-host Langfuse for observability. Add your keys to a `.env` file.
- **Ollama**: Install Ollama for local model serving.
- **OpenAI**: You can also use OpenAI models. You just have to update your config files (`local.yaml` and/or `base.yaml`) accordingly.

### 3. Deployment (AWS)

This project uses Terraform for Infrastructure as Code and GitHub Actions for CI/CD.

**Manual Deployment**: Navigate to the `/terraform` directory and follow the README file within that directory.

**CI/CD**: Pushing to the main branch triggers the GitHub Action workflow which:

1. Builds the Docker image.
2. Pushes the image to Amazon ECR.
3. Updates the ECS Service to deploy the new task definition.

## Usage

### Setup Environment and Install
Before querying, ensure your environment is set up and the package is installed. We recommend using [uv](https://github.com/astral-sh/uv) for fast, reliable dependency management.
```bash
# Create a virtual environment and install the package in editable mode
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .
```
### Launch the Local Server
For the local deployment, launch the FastAPI server with hot-reloading:

```bash
nzambe server --reload
```

### Querying the API

You can query the deployed app or local instance using the `nzambe` CLI:

**Via CLI**:

To query the local server:

```bash
nzambe query -q "Give me in order what was created at the beginning"
```

To query the remote server:

```bash
nzambe query -q "What happen to Jonah in one sentence?" --base-url <ALB_DNS_NAME>
```

**Note**: `<ALB_DNS_NAME>` can be retrieved from your Terraform outputs.

## Next Steps

- [ ] **Data Expansion**: Incorporate the Quran and Torah into the vector index.
- [ ] **Retrieval Optimization**: Implement Reranking and BM25 hybrid search. Also include a SummaryIndex to improve robustness on queries spanning multiple books.
- [ ] **Node Filtering**: Add metadata filtering to allow users to query specific books.
- [ ] **Observability**: Deploy Langfuse in the cluster to track every interaction and ease RAG app evaluation.
