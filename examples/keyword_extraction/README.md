# Keyword Extraction Example

This example demonstrates how to use the RLRefine framework for keyword extraction from Chinese e-commerce reviews.

> **Note**: This example was designed and tested for Chinese text by the author (rxy). English effectiveness has not been fully validated.

## How to Run

### 1. Start the vLLM Service

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct --port 8000
```

### 2. Run the Example

```bash
cd RLRefine
python examples/keyword_extraction/run.py
```

## File Descriptions

- `schema.py`: Defines the JSON Schema for keyword extraction
- `config.py`: Task configuration
- `run.py`: Main execution script
- `sample_data.jsonl`: Sample data

## Custom Tasks

Refer to the [Custom Tasks](../../README.md#custom-tasks) section in the main README.
