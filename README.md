# AI Agent for SQL with LLM Fine-Tuning and Self-Supervised Database Embeddings

Insurance-fraud investigation assistant for querying relational insurance records with natural language, safe SQL, and claim-similarity retrieval.

This repository extends an SQL fine-tuning prototype into a containerized FastAPI platform with two separate model paths:

1. A fine-tuned open-source Falcon 11B causal LLM using PEFT QLoRA for SQL generation, tool orchestration, and grounded summaries.
2. A custom PyTorch Skip-Gram embedding model trained from scratch on structured relational insurance claim features for semantic claim similarity.

The system supports investigation workflows. It does not automatically determine fraud, and similarity is never treated as proof of fraud.

## Overview

Insurance investigators need to ask exact relational questions and also discover structurally similar claim patterns. This project combines natural-language SQL generation with vector retrieval over learned claim representations.

The application helps investigators:

- Query insurance records using natural language.
- Generate and execute safe read-only SQL.
- Retrieve exact records using PostgreSQL joins and filters.
- Find claims structurally similar to a selected claim.
- Combine similarity retrieval with exact SQL filters.
- Inspect shared repair shops, bank accounts, addresses, phone numbers, policies, vehicles, and payment recipients.
- Compare retrieved claims with historical fraud outcomes stored as database facts.
- Generate grounded investigation summaries from retrieved evidence.

Historical fraud labels are metadata and may be used for SQL filtering and offline evaluation. They are not used to train the self-supervised embedding model.

## Architecture Diagram

```mermaid
flowchart TD
    user["Investigator"] --> api["FastAPI API"]
    api --> agent["OpenAI Agents SDK"]
    agent --> falcon["Fine-tuned Falcon 11B QLoRA"]
    falcon --> sql_tool["SQL Tool"]
    falcon --> semantic_tool["Semantic Search Tool"]
    sql_tool --> validator["Read-only SQL Validator"]
    validator --> postgres["PostgreSQL Relational Store"]
    semantic_tool --> pgvector["pgvector Similarity Search"]
    pgvector --> candidate_ids["Top-K Candidate Claim IDs"]
    candidate_ids --> postgres
    postgres --> evidence["Complete Relational Evidence"]
    evidence --> falcon
    falcon --> summary["Grounded Investigation Summary"]

    joined["Joined Claim Representation"] --> tokens["Column-aware Tokens"]
    tokens --> skipgram["Custom PyTorch Skip-Gram Model"]
    skipgram --> token_matrix["Token Embedding Matrix"]
    token_matrix --> pooling["Mean Pooling"]
    pooling --> claim_vectors["Claim Vectors"]
    claim_vectors --> pgvector
```

## Two-Model Architecture

### Fine-Tuned Falcon SQL LLM

The Falcon model is the single reasoning and generation model in the architecture. It is responsible for interpreting investigator requests, generating read-only SQL, selecting agent tools, producing follow-up SQL from semantic candidates, and summarizing grounded evidence.

The SQL model is fine-tuned with supervised instruction tuning using PEFT QLoRA, LoRA adapters, 4-bit quantization, gradient accumulation, and optional Weights & Biases tracking.

### Custom Database Embedding Model

The embedding model is a custom PyTorch Skip-Gram with Negative Sampling model trained from scratch on structured, column-aware claim tokens. It does not use OpenAI embeddings, Sentence Transformers, BERT, or external embedding APIs.

```text
Structured joined claim
-> column-aware tokenization
-> Skip-Gram training
-> token embedding matrix
-> mean pooling
-> claim vector
-> pgvector
-> top-K candidate claim IDs
-> SQL
-> complete relational evidence
-> Falcon
-> grounded investigation summary
```

The joined claim representation combines deterministic relational features from claims, customers, policies, vehicles, repair shops, incidents, payment aggregates, and prior-claim engineered features.

Example engineered features include `previous_claim_count`, `previous_total_claim_amount`, `previous_fraud_count`, `customer_claim_frequency`, `policy_age_days`, `incident_to_claim_delay_days`, `shared_bank_account_count`, `shared_phone_count`, and `shared_address_count`.

Example tokens:

```text
incident_type=vehicle_theft
incident_city=london
claim_amount_bin=20000_25000
policy_age_bin=0_30_days
vehicle_type=suv
repair_shop_id=RS_018
police_report_available=true
previous_claim_count_bin=3_5
shared_bank_account_count_bin=2_3
```

Every value is prefixed by its feature name. Each claim row is treated as an unordered bag of tokens, where every token may act as context for every other token in that row.

## Technology Stack

- Python
- FastAPI
- PostgreSQL
- pgvector
- SQLAlchemy
- OpenAI Agents SDK
- Hugging Face Transformers
- PEFT
- QLoRA
- BitsAndBytes
- PyTorch
- Weights & Biases
- Docker
- JSON APIs

## System Workflows

### Exact SQL Query

```text
User prompt
-> FastAPI
-> OpenAI Agents SDK agent
-> fine-tuned Falcon generates SQL
-> SQL validator checks read-only safety
-> PostgreSQL executes approved SQL
-> relational rows are returned
-> Falcon generates grounded response
```

The embedding model and pgvector are not used for exact relational queries.

### Semantic Similarity Search

```text
User prompt
-> FastAPI
-> agent identifies similarity request
-> find_similar_claims retrieves source claim vector
-> pgvector returns candidate claim IDs ranked by cosine similarity
-> SQL retrieves complete relational records for those IDs
-> explain_similarity compares shared tokens, entities, and bins
-> Falcon generates grounded investigation summary
```

pgvector returns ranked candidate IDs. It does not return the full investigation context.

### Combined SQL And Semantic Query

```text
User prompt
-> FastAPI
-> agent identifies combined workflow
-> pgvector retrieves top-K semantically similar claim IDs
-> SQL applies exact filters to candidate IDs
-> PostgreSQL retrieves complete evidence records
-> similarity explanations are generated
-> Falcon writes a grounded summary
```

Similarity ranking and exact business filtering remain separate responsibilities.

## Project Structure

```text
src/
  api/
    routes/
    dependencies/
    middleware/
  agents/
  tools/
  database/
    migrations/
  sql_model/
  embedding_model/
  services/
  schemas/
  evaluation/
  core/

scripts/
data/
  sample/
  generated/
models/
artifacts/
tests/
  unit/
  integration/
  evaluation/
```

## Relational Schema

PostgreSQL stores normalized investigation evidence across these tables:

- `customers` - customer demographics, contact details, and account creation date.
- `policies` - policy type, status, coverage, premiums, deductibles, and effective dates.
- `vehicles` - insured vehicles, estimated values, and registration regions.
- `repair_shops` - repair shops, owners, locations, and bank-account references.
- `incidents` - incident type, date, city, address, weather, police report, and witness count.
- `claims` - claim amounts, dates, statuses, descriptions, historical outcomes, and `claim_embedding VECTOR(128)`.
- `claim_participants` - drivers, witnesses, passengers, third parties, and other linked people or organizations.
- `payments` - claim payments, recipients, bank-account references, amounts, dates, and statuses.

The schema includes foreign keys from claims to policies, vehicles, incidents, and repair shops, plus cascading detail records for participants and payments. Indexes support investigative lookups by date, amount, status, fraud outcome metadata, repair shop, bank account, contact information, and pgvector cosine similarity.

## Synthetic Data Workflow

The data generator creates deterministic linked insurance records with realistic investigation patterns:

- Normal claims and historically fraudulent claims.
- Repeated repair shops and payment recipients.
- Shared bank accounts, phone numbers, and addresses.
- Claims filed shortly after policy creation.
- High claim amounts and repeated high-value repairs.
- Coordinated groups of claims.
- Similar claims with mixed historical outcomes.

Generated data is written as one JSON file per database table under `data/generated/`:

```bash
python scripts/generate_insurance_data.py --seed 2410 --customers 200 --normal-claims 500
```

The loader imports table files into PostgreSQL in foreign-key order:

```bash
python scripts/load_insurance_data.py --input-dir data/generated --replace
```

A small fixture at `data/sample/insurance_sample.json` demonstrates the linked JSON shape. Claim embeddings remain empty in generated and sample data; vectors are created by the custom embedding pipeline and stored in PostgreSQL after indexing.

## Data Generation Policy

The repository does not contain the full synthetic dataset, PostgreSQL database files, model checkpoints, adapter weights, claim vectors, W&B logs, or generated evaluation outputs.

Committed data is limited to schema definitions, migrations, generator scripts, loading scripts, source code, configuration, documentation, tests, placeholders, and small sample fixtures.

Generated files are written locally to:

```text
data/generated/
models/
artifacts/
```

PostgreSQL data uses a Docker named volume and is not stored in the repository.

## Model Architecture

### Falcon QLoRA SQL Model

The SQL model path supports configurable 11B model loading, 4-bit quantization, LoRA adapter setup, supervised instruction tuning, adapter save/load, and SQL generation for the agent and API.

Training data follows this structure:

```json
{
  "instruction": "Show confirmed fraudulent claims above 15000 filed within 30 days of the policy start date.",
  "schema": "customers(...), policies(...), claims(...)",
  "output": "SELECT ...",
  "metadata": {
    "reasoning": "optional training-only metadata"
  }
}
```

The model is trained for instruction tuning, not few-shot prompting.

### Skip-Gram Claim Embedding Model

The embedding model path supports structured feature extraction, numeric binning, column-aware tokenization, Skip-Gram positive-pair creation, negative sampling, PyTorch training with Adam, token embedding export, claim-vector mean pooling, and claim-vector indexing into PostgreSQL `pgvector`.

Default training settings:

```text
embedding_dimension = 128
negative_samples = 5
epochs = 20
optimizer = Adam
minimum_token_frequency = 1
similarity_metric = cosine
claim_aggregation = mean_pooling
```

## API Overview

API endpoints:

- `POST /agent/query` - natural-language investigation request.
- `POST /sql/generate` - generate SQL without executing it.
- `POST /sql/execute` - validate and execute approved read-only SQL.
- `POST /semantic/claims/{claim_id}` - return similar claims.
- `POST /semantic/pattern` - search using structured claim fields.
- `POST /models/sql/train` - protected SQL-model training job definition.
- `POST /models/embedding/train` - protected embedding-training job definition.
- `POST /models/embedding/index` - claim-vector indexing job definition.
- `GET /models/status` - configured model paths and artifact status.
- `GET /health` - application and dependency status.

Training endpoints are protected and are not publicly exposed without authentication.

## SQL Safety

All generated or user-submitted SQL passes a read-only validator before execution. The validator blocks write statements, DDL, multiple SQL statements, bypass comments, system schema access, and unrestricted queries without sensible row limits. It uses SQL parsing in addition to keyword-level safeguards.

## Implementation Modules

The repository is organized around these major implementation areas:

- Database schema, migrations, and SQLAlchemy models.
- Synthetic data generation and database loading scripts.
- Falcon QLoRA SQL model training and inference pipeline.
- Custom Skip-Gram embedding model training and claim-vector indexing pipeline.
- Semantic search and similarity explanation services.
- Agent tools and FastAPI routes.
- Docker infrastructure, tests, and evaluation scripts.

## Project Evolution

This repository began as a SQL fine-tuning prototype using PEFT QLoRA for SQL generation. The original notebook and model references are preserved as the starting point for the broader insurance-fraud investigation platform.

Original prototype highlights:

- Base model: Falcon2-11B / `tiiuae/falcon-11B`
- 4-bit quantization for memory-efficient fine-tuning
- PEFT with LoRA adapters
- Supervised instruction tuning for SQL generation
- Hugging Face adapter publication
- Weights & Biases experiment tracking

Links from the original prototype:

- [Fine-tuned PEFT adapters on Hugging Face](https://huggingface.co/adityas2410/falcon11b-sql_instruct/tree/main)
- [Weights & Biases report](https://api.wandb.ai/links/adityas-ai2410-upwork/58a35uld)

Original inference screenshots:

<img width="768" alt="SQL fine-tuning inference screenshot" src="https://github.com/user-attachments/assets/8208d827-8496-496d-8623-212d7daf8f8e">
<img width="928" alt="SQL fine-tuning inference screenshot" src="https://github.com/user-attachments/assets/2fdd5425-61ef-4857-92b0-7590d99a9258">
