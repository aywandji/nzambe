# nzambe
Q&amp;A application for the main holy books (Bible, Quran and Torah)

## Holy books data
- The Bible in english:  curl https://www.gutenberg.org/cache/epub/10/pg10.txt --output .debug/data/bible_en.txt

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
- [ ] Add the 2 other holy books (Quran and Torah)
- [ ] Optimize the query process for speed and accuracy (reranking, bm25 retriever, nodes filtering, etc.)
