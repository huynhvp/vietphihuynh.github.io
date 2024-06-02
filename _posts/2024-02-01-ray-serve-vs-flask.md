---
layout: post
title: Serve a ML model on a single machine with Flask + Gunicorn vs. FastAPI + Uvicorn
date: 2024-02-01 00:09:00
description: 
tags: dev
categories: flask, gunicorn, fastapi, uvicorn
---

[Gunicorn](https://docs.gunicorn.org/en/stable/design.html) is a Python Web Server Gateway Interface (WSGI) HTTP server based on the pre-fork worker model. By default, Gunicorn workers are synchronous, which handle a single request at a time. However, Gunicorn also supports asynchronous workers, such as `gevent` or `eventlet`, which can manage multiple simutaneous requests. The web application framework, [Flask](https://github.com/pallets/flask), typically paired with Gunicorn, forms a synchronous or asynchronous web service depending on whether sync or async gunicorn workers are used.

[Uvicorn](https://www.uvicorn.org/), unlike Gunicorn, is inherently a Python Asynchronous Server Gateway Interface (ASGI) HTTP server. The web application framework, [FastAPI](https://github.com/tiangolo/fastapi) uses Uvicorn internally to serve an asynchronous web service.

Let's minimally deploy a standalone ML model on a single machine with Flask + Gunicorn and FastAPI + Uvicorn and measure the throughput (i.e. time to process 1000 concurrent requests) of both systems.

---
<b>Table of Contents</b>
* TOC []
{:toc}


#### 1. CPU-bound: Deploy a Named Entity Recognition (NER) model 

<b>Server</b>

```python
# server.py
import fastapi
import flask
import spacy

# load NER model
ner = spacy.load("en_core_web_sm")

# flask web server
flask_app = flask.Flask(__name__)

@flask_app.get("/")
def ner_flask():
    text = flask.request.json.get("text")
    entities = ner(text)
    return {ent.text: ent.label_ for ent in entities.ents}

# fastapi web server
fastapi_app = fastapi.FastAPI()

@fastapi_app.get("/")
async def ner_fastapi(request: fastapi.Request):
    text = (await request.json()).get("text")
    entities = ner(text)
    return {ent.text: ent.label_ for ent in entities.ents}

```

- Run the synchronus API with Flask and Gunicorn's sync workers.
```bash
gunicorn --bind 127.0.0.1:5000 --worker-connections 1000  -w 2  server:flask_app
```

- Run the asynchronus API with Flask and Gunicorn's async workers `gevent`.
```bash
gunicorn --bind 127.0.0.1:5000 --worker-class=gevent --worker-connections 1000  -w 2  server:flask_app
```

- Run the asynchronus API with FastAPI and Uvicorn
```bash
uvicorn --host 127.0.0.1 --port 5000 --limit-concurrency 1000 --workers 2  server:fastapi_app 
```

where:
- `-w` or `--workers`: number of workers for handling requests.
- `--worker-connections` or `--limit-concurrency`: maximum number of concurrent requests a worker can handle. 

<b>Client</b>: let's make 1000 simultaneous requests, send them to the server, and measure the processing time.

```python
# client.py
import asyncio
import time

from aiohttp import ClientSession

api_url = "http://127.0.0.1:5000/"

text = """Paris[a] is the capital and most populous city of France. With an official estimated population of 2,102,650 residents
as of 1 January 2023[2] in an area of more than 105 km2 (41 sq mi),[5] Paris is the fourth-most populated city in the European
Union and the 30th most densely populated city in the world in 2022.[6] Since the 17th century, Paris has been one of the world'
major centres of finance, diplomacy, commerce, culture, fashion, and gastronomy. For its leading role in the arts and sciences,
as well as its early and extensive system of street lighting, in the 19th century, it became known as the City of Light.[7]
The City of Paris is the centre of the Île-de-France region, or Paris Region, with an officia estimated population of 12,271,794
inhabitants on 1 January 2023, or about 19% of the population of France.
"""

async def fetch(session: ClientSession, i_request: int):
    # fetch NER result for a request
    async with session.get(api_url, json={"i_request": i_request, "text": text}) as response:
        result = await response.json()
    return result

async def main(i_trial: int):
    # send 1000 simultaneous requests to the server
    num_requests = 1000
    async with ClientSession() as session:
        tasks = []
        for i_request in range(num_requests):
            tasks.append(fetch(session, i_trial*num_requests + i_request))
        await asyncio.gather(*tasks)

num_trials = 5
times = []
for i_trial in range(num_trials):
    start = time.perf_counter()
    asyncio.run(main(i_trial))
    end = time.perf_counter()
    times.append(end - start)

print(f"Avg Time: {sum(times)/len(times):.2f}")

```

**<span style="color:green"><b>Processing time for a batch of 1000 simultanous requests</b></span>**

| Number of workers  | w=1  | w=2  | w=4
| API  | 
|-------|--------|---------|
| Flask + sync Gunicorn | 5.63 (s) | 3.34 (s) | 2.31 (s)
| Flask + async Gunicorn | 6.04 (s) | 3.29 (s) | 2.33 (s)
| FastAPI + async Uvicorn | 6.35 (s) | 3.5 (s) | 2.62 (s)

<p></p>

As the model spends all its time on the CPU to process requests, designing an async web application is not helpful.

#### 2. CPU- and IO- bound: Deploy a Named Entity Recognition (NER) model, preceded by IO operations.

Just for testing purposes, let's add a nonsensical `sleep(0.1)` to the model, to represent its IO-bound aspect, and mesure again the throughputs.

```python
# server.py
@flask_app.get("/")
def ner_flask():
    time.sleep(0.1) # do some IO operations
    text = flask.request.json.get("text")
    entities = ner(text)
    return {ent.text: ent.label_ for ent in entities.ents}

@fastapi_app.get("/")
async def ner_fastapi(request: fastapi.Request):
    await asyncio.sleep(0.1) # do some IO operations
    text = (await request.json()).get("text")
    entities = ner(text)
    return {ent.text: ent.label_ for ent in entities.ents}
```

**<span style="color:green"><b>Processing time for a batch of 1000 simultanous requests</b></span>**

| Priority apples | 1  | 2  | 4
|-------|--------|---------|
| Flask + sync Gunicorn | 129.69 (s) | 65.51 (s) | 33.25 (s)
| Flask + async Gunicorn | 7.62 (s) | 3.73 (s) | 3.19 (s)
| FastAPI + async Uvicorn | 7.80 (s) | 4.14 (s) | 2.74 (s)

<p></p>

Clearly, the async implementation significantly improve the efficiency of model serving.

Additionally, in both test cases, it appears that the difference between Flask with async Gunicorn workers and FastAPI is not conclusive.


