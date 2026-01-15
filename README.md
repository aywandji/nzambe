# nzambe
Q&amp;A application for the main holy books (Bible, Quran and Torah)

## Holy books data
- the english bible:  curl https://www.gutenberg.org/cache/epub/10/pg10.txt --output .debug/data/bible_en.txt

## Spin up the project locally
### Set up the configs
Make sure the config yaml files are defined inside the config folder.
- base.yaml: base config for all the environments
- (Optional) prod.yaml: overrides for prod environment
- (Optional) staging.yaml: overrides for staging environment
- (Optional) local.yaml: overrides for local/dev environment

### Set up third party services
- Self host langfuse for observability: https://langfuse.com/self-hosting
  - create a user/password and a new project
  - add keys and base url
      - to an.env that will be loaded by the app
      - or as environment variables
- Download ollama for local model serving: https://docs.ollama.com/quickstart

### Run the project
- Launch the app server (in development mode) with:
  - nzambe server --reload to launch the server with hot reload
  - send a query
    - with the app cli: `nzambe query -q your_question`. Check the help with `nzambe query --help`
    - with curl:
      ```bash
      curl -X POST http://localhost:8000/query -H "Content-Type: application/json" -d '{"question": "give me in order what was created at the beginning"}'
      ```
    - or using the swagger UI doc at /docs


## Next steps
- query a lot of documents from the vector store and use similarity filtering to keep only relevant context nodes
- Use a llm reranker as a node post-processor to filter out less relevant context nodes and improve the accuracy of the answer
- combine bm25 retriever (tf-idf like) and semantic search to get more input context nodes for the llm reranker: https://developers.llamaindex.ai/python/examples/retrievers/bm25_retriever/
- use LongContextReorder post-processor to reorder the answer nodes to get the most relevant nodes at the edges of the context as llm tends to "forget" the data in the middle of very long contexts: https://developers.llamaindex.ai/python/framework/module_guides/querying/node_postprocessors/node_postprocessors/#longcontextreorder
- Add the 2 other holy books (Quran and Torah)
  - [ ]  basic workflow iteratively retrieving documents from each of the 3 databases.
  - [ ]  multi document query might be a way to optimize the 3 books query? https://developers.llamaindex.ai/python/framework/understanding/putting_it_all_together/q_and_a#multi-document-queries
  - [ ]  check this to further optimize things: https://developers.llamaindex.ai/python/framework/use_cases/q_and_a/#resources
