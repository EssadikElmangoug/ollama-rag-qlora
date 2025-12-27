 => ERROR [backend 5/7] RUN pip3 install --no-cache-dir --upgrade pip &&     pip3 install --no-cache-dir -r req  530.9s
 => [frontend internal] load .dockerignore                                                                         0.0s
 => => transferring context: 523B                                                                                  0.0s
 => [frontend internal] load build context                                                                         0.0s
 => => transferring context: 1.26kB                                                                                0.0s
 => [frontend base 1/1] FROM docker.io/library/node:20-alpine@sha256:658d0f63e501824d6c23e06d4bb95c71e7d704537c9d  0.0s
 => => resolve docker.io/library/node:20-alpine@sha256:658d0f63e501824d6c23e06d4bb95c71e7d704537c9d9272f488ac03a3  0.0s
 => CACHED [frontend deps 1/3] WORKDIR /app                                                                        0.0s
 => CACHED [frontend runner 2/5] RUN addgroup --system --gid 1001 nodejs &&     adduser --system --uid 1001 nextj  0.0s
 => CACHED [frontend deps 2/3] COPY package.json package-lock.json* ./                                             0.0s
 => CACHED [frontend deps 3/3] RUN npm config set fetch-retries 10 &&     npm config set fetch-retry-mintimeout 2  0.0s
 => CACHED [frontend builder 2/4] COPY --from=deps /app/node_modules ./node_modules                                0.0s
 => CACHED [frontend builder 3/4] COPY . .                                                                         0.0s
 => CACHED [frontend builder 4/4] RUN npm run build                                                                0.0s
 => CACHED [frontend runner 3/5] COPY --from=builder --chown=nextjs:nodejs /app/public ./public                    0.0s
 => CACHED [frontend runner 4/5] COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./                0.0s
 => CACHED [frontend runner 5/5] COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static        0.0s
 => [frontend] exporting to image                                                                                  0.1s
 => => exporting layers                                                                                            0.0s
 => => exporting manifest sha256:23d84d1355652ef15e57532128f26b2972040e77da10d870578a484e94e3647d                  0.0s
 => => exporting config sha256:8a9ed1cd93dee860e9ea52115fa67ceab71e5b665405f04f7bd9715b2699fdc9                    0.0s
 => => exporting attestation manifest sha256:a4328c8bb766165255fdde9b9cfa1b68ac0ad1e910170df278105a725c5bca18      0.0s
 => => exporting manifest list sha256:64a91da9d3c71e85b073137aca26eade5ddf109da82f83305dc9a9a504977705             0.0s
 => => naming to docker.io/library/ollama-rag-qlora-frontend:latest                                                0.0s
 => => unpacking to docker.io/library/ollama-rag-qlora-frontend:latest                                             0.0s
 => [frontend] resolving provenance for metadata file                                                              0.0s
------
 > [backend 5/7] RUN pip3 install --no-cache-dir --upgrade pip &&     pip3 install --no-cache-dir -r requirements.txt:
0.680 Requirement already satisfied: pip in /usr/lib/python3/dist-packages (22.0.2)
1.035 Collecting pip
1.235   Downloading pip-25.3-py3-none-any.whl (1.8 MB)
1.646      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.8/1.8 MB 4.4 MB/s eta 0:00:00
1.658 Installing collected packages: pip
1.658   Attempting uninstall: pip
1.659     Found existing installation: pip 22.0.2
1.660     Not uninstalling pip at /usr/lib/python3/dist-packages, outside environment /usr
1.660     Can't uninstall 'pip'. No files were found to uninstall.
2.074 Successfully installed pip-25.3
2.074 WARNING: Running pip as the 'root' user can result in broken permissions and conflicting behaviour with the system package manager. It is recommended to use a virtual environment instead: https://pip.pypa.io/warnings/venv
2.633 Collecting Flask==3.0.0 (from -r requirements.txt (line 1))
2.814   Downloading flask-3.0.0-py3-none-any.whl.metadata (3.6 kB)
2.920 Collecting flask-cors==4.0.0 (from -r requirements.txt (line 2))
2.965   Downloading Flask_Cors-4.0.0-py2.py3-none-any.whl.metadata (5.4 kB)
3.262 Collecting langchain>=0.1.0 (from -r requirements.txt (line 3))
3.303   Downloading langchain-1.2.0-py3-none-any.whl.metadata (4.9 kB)
3.420 Collecting langchain-community>=0.0.10 (from -r requirements.txt (line 4))
3.462   Downloading langchain_community-0.4.1-py3-none-any.whl.metadata (3.0 kB)
3.577 Collecting langchain-core>=0.1.0 (from -r requirements.txt (line 5))
3.632   Downloading langchain_core-1.2.5-py3-none-any.whl.metadata (3.7 kB)
3.721 Collecting langchain-text-splitters>=0.0.1 (from -r requirements.txt (line 6))
3.769   Downloading langchain_text_splitters-1.1.0-py3-none-any.whl.metadata (2.7 kB)
3.855 Collecting sentence-transformers>=2.2.2 (from -r requirements.txt (line 7))
3.905   Downloading sentence_transformers-5.2.0-py3-none-any.whl.metadata (16 kB)
4.012 Collecting faiss-cpu==1.13.1 (from -r requirements.txt (line 8))
4.055   Downloading faiss_cpu-1.13.1-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (7.6 kB)
4.150 Collecting pypdf==3.17.0 (from -r requirements.txt (line 9))
4.191   Downloading pypdf-3.17.0-py3-none-any.whl.metadata (7.5 kB)
4.283 Collecting python-docx==1.1.0 (from -r requirements.txt (line 10))
4.324   Downloading python_docx-1.1.0-py3-none-any.whl.metadata (2.0 kB)
4.409 Collecting openpyxl==3.1.2 (from -r requirements.txt (line 11))
4.453   Downloading openpyxl-3.1.2-py2.py3-none-any.whl.metadata (2.5 kB)
4.678 Collecting pandas==2.1.4 (from -r requirements.txt (line 12))
4.730   Downloading pandas-2.1.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (18 kB)
4.876 Collecting unstructured==0.10.30 (from -r requirements.txt (line 13))
4.920   Downloading unstructured-0.10.30-py3-none-any.whl.metadata (25 kB)
5.141 Collecting unsloth>=2024.8 (from unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
5.192   Downloading unsloth-2025.12.9-py3-none-any.whl.metadata (65 kB)
5.558 Collecting trl>=0.7.0 (from -r requirements.txt (line 15))
5.602   Downloading trl-0.26.2-py3-none-any.whl.metadata (11 kB)
5.694 Collecting peft>=0.6.0 (from -r requirements.txt (line 16))
5.741   Downloading peft-0.18.0-py3-none-any.whl.metadata (14 kB)
5.835 Collecting datasets>=2.14.0 (from -r requirements.txt (line 17))
5.886   Downloading datasets-4.4.2-py3-none-any.whl.metadata (19 kB)
6.004 Collecting transformers>=4.36.0 (from -r requirements.txt (line 18))
6.051   Downloading transformers-4.57.3-py3-none-any.whl.metadata (43 kB)
6.247 Collecting accelerate>=0.21.0 (from -r requirements.txt (line 19))
6.290   Downloading accelerate-1.12.0-py3-none-any.whl.metadata (19 kB)
6.385 Collecting bitsandbytes>=0.41.0 (from -r requirements.txt (line 20))
6.433   Downloading bitsandbytes-0.49.0-py3-none-manylinux_2_24_x86_64.whl.metadata (10 kB)
6.614 Collecting xformers>=0.0.22 (from -r requirements.txt (line 21))
6.656   Downloading xformers-0.0.33.post2-cp39-abi3-manylinux_2_28_x86_64.whl.metadata (1.2 kB)
6.742 Collecting requests>=2.31.0 (from -r requirements.txt (line 22))
6.783   Downloading requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
6.889 Collecting huggingface_hub>=0.19.0 (from -r requirements.txt (line 23))
6.935   Downloading huggingface_hub-1.2.3-py3-none-any.whl.metadata (13 kB)
7.062 Collecting torch>=2.0.0 (from -r requirements.txt (line 24))
7.112   Downloading torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (30 kB)
7.254 Collecting psutil>=5.9.0 (from -r requirements.txt (line 25))
7.296   Downloading psutil-7.2.0-cp36-abi3-manylinux2010_x86_64.manylinux_2_12_x86_64.manylinux_2_28_x86_64.whl.metadata (22 kB)
7.386 Collecting Werkzeug>=3.0.0 (from Flask==3.0.0->-r requirements.txt (line 1))
7.433   Downloading werkzeug-3.1.4-py3-none-any.whl.metadata (4.0 kB)
7.538 Collecting Jinja2>=3.1.2 (from Flask==3.0.0->-r requirements.txt (line 1))
7.581   Downloading jinja2-3.1.6-py3-none-any.whl.metadata (2.9 kB)
7.685 Collecting itsdangerous>=2.1.2 (from Flask==3.0.0->-r requirements.txt (line 1))
7.729   Downloading itsdangerous-2.2.0-py3-none-any.whl.metadata (1.9 kB)
7.858 Collecting click>=8.1.3 (from Flask==3.0.0->-r requirements.txt (line 1))
7.898   Downloading click-8.3.1-py3-none-any.whl.metadata (2.6 kB)
7.975 Collecting blinker>=1.6.2 (from Flask==3.0.0->-r requirements.txt (line 1))
8.014   Downloading blinker-1.9.0-py3-none-any.whl.metadata (1.6 kB)
8.362 Collecting numpy<3.0,>=1.25.0 (from faiss-cpu==1.13.1->-r requirements.txt (line 8))
8.414   Downloading numpy-2.2.6-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (62 kB)
8.534 Collecting packaging (from faiss-cpu==1.13.1->-r requirements.txt (line 8))
8.578   Downloading packaging-25.0-py3-none-any.whl.metadata (3.3 kB)
8.894 Collecting lxml>=3.1.0 (from python-docx==1.1.0->-r requirements.txt (line 10))
8.939   Downloading lxml-6.0.2-cp310-cp310-manylinux_2_26_x86_64.manylinux_2_28_x86_64.whl.metadata (3.6 kB)
9.036 Collecting typing-extensions (from python-docx==1.1.0->-r requirements.txt (line 10))
9.083   Downloading typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
9.160 Collecting et-xmlfile (from openpyxl==3.1.2->-r requirements.txt (line 11))
9.210   Downloading et_xmlfile-2.0.0-py3-none-any.whl.metadata (2.7 kB)
9.215 Collecting numpy<3.0,>=1.25.0 (from faiss-cpu==1.13.1->-r requirements.txt (line 8))
9.266   Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
9.368 Collecting python-dateutil>=2.8.2 (from pandas==2.1.4->-r requirements.txt (line 12))
9.410   Downloading python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
9.538 Collecting pytz>=2020.1 (from pandas==2.1.4->-r requirements.txt (line 12))
9.581   Downloading pytz-2025.2-py2.py3-none-any.whl.metadata (22 kB)
9.675 Collecting tzdata>=2022.1 (from pandas==2.1.4->-r requirements.txt (line 12))
9.718   Downloading tzdata-2025.3-py2.py3-none-any.whl.metadata (1.4 kB)
9.801 Collecting chardet (from unstructured==0.10.30->-r requirements.txt (line 13))
9.845   Downloading chardet-5.2.0-py3-none-any.whl.metadata (3.4 kB)
9.927 Collecting filetype (from unstructured==0.10.30->-r requirements.txt (line 13))
9.978   Downloading filetype-1.2.0-py2.py3-none-any.whl.metadata (6.5 kB)
10.07 Collecting python-magic (from unstructured==0.10.30->-r requirements.txt (line 13))
10.12   Downloading python_magic-0.4.27-py2.py3-none-any.whl.metadata (5.8 kB)
10.21 Collecting nltk (from unstructured==0.10.30->-r requirements.txt (line 13))
10.25   Downloading nltk-3.9.2-py3-none-any.whl.metadata (3.2 kB)
10.33 Collecting tabulate (from unstructured==0.10.30->-r requirements.txt (line 13))
10.38   Downloading tabulate-0.9.0-py3-none-any.whl.metadata (34 kB)
10.48 Collecting beautifulsoup4 (from unstructured==0.10.30->-r requirements.txt (line 13))
10.52   Downloading beautifulsoup4-4.14.3-py3-none-any.whl.metadata (3.8 kB)
10.61 Collecting emoji (from unstructured==0.10.30->-r requirements.txt (line 13))
10.65   Downloading emoji-2.15.0-py3-none-any.whl.metadata (5.7 kB)
10.74 Collecting dataclasses-json (from unstructured==0.10.30->-r requirements.txt (line 13))
10.78   Downloading dataclasses_json-0.6.7-py3-none-any.whl.metadata (25 kB)
10.91 Collecting python-iso639 (from unstructured==0.10.30->-r requirements.txt (line 13))
10.95   Downloading python_iso639-2025.11.16-py3-none-any.whl.metadata (15 kB)
11.03 Collecting langdetect (from unstructured==0.10.30->-r requirements.txt (line 13))
11.09   Downloading langdetect-1.0.9.tar.gz (981 kB)
11.28      ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 981.5/981.5 kB 6.6 MB/s  0:00:00
11.31   Installing build dependencies: started
12.85   Installing build dependencies: finished with status 'done'
12.85   Getting requirements to build wheel: started
13.01   Getting requirements to build wheel: finished with status 'done'
13.01   Preparing metadata (pyproject.toml): started
13.18   Preparing metadata (pyproject.toml): finished with status 'done'
13.79 Collecting rapidfuzz (from unstructured==0.10.30->-r requirements.txt (line 13))
13.84   Downloading rapidfuzz-3.14.3-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (12 kB)
13.92 Collecting backoff (from unstructured==0.10.30->-r requirements.txt (line 13))
13.97   Downloading backoff-2.2.1-py3-none-any.whl.metadata (14 kB)
14.11 Collecting langgraph<1.1.0,>=1.0.2 (from langchain>=0.1.0->-r requirements.txt (line 3))
14.15   Downloading langgraph-1.0.5-py3-none-any.whl.metadata (7.4 kB)
14.31 Collecting pydantic<3.0.0,>=2.7.4 (from langchain>=0.1.0->-r requirements.txt (line 3))
14.36   Downloading pydantic-2.12.5-py3-none-any.whl.metadata (90 kB)
14.46 Collecting jsonpatch<2.0.0,>=1.33.0 (from langchain-core>=0.1.0->-r requirements.txt (line 5))
14.50   Downloading jsonpatch-1.33-py2.py3-none-any.whl.metadata (3.0 kB)
14.69 Collecting langsmith<1.0.0,>=0.3.45 (from langchain-core>=0.1.0->-r requirements.txt (line 5))
14.73   Downloading langsmith-0.5.1-py3-none-any.whl.metadata (15 kB)
14.85 Collecting pyyaml<7.0.0,>=5.3.0 (from langchain-core>=0.1.0->-r requirements.txt (line 5))
14.89   Downloading pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.4 kB)
15.01 Collecting tenacity!=8.4.0,<10.0.0,>=8.1.0 (from langchain-core>=0.1.0->-r requirements.txt (line 5))
15.06   Downloading tenacity-9.1.2-py3-none-any.whl.metadata (1.2 kB)
15.18 Collecting uuid-utils<1.0,>=0.12.0 (from langchain-core>=0.1.0->-r requirements.txt (line 5))
15.23   Downloading uuid_utils-0.12.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.1 kB)
15.31 Collecting jsonpointer>=1.9 (from jsonpatch<2.0.0,>=1.33.0->langchain-core>=0.1.0->-r requirements.txt (line 5))
15.35   Downloading jsonpointer-3.0.0-py2.py3-none-any.whl.metadata (2.3 kB)
15.44 Collecting langgraph-checkpoint<4.0.0,>=2.1.0 (from langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
15.48   Downloading langgraph_checkpoint-3.0.1-py3-none-any.whl.metadata (4.7 kB)
15.57 Collecting langgraph-prebuilt<1.1.0,>=1.0.2 (from langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
15.61   Downloading langgraph_prebuilt-1.0.5-py3-none-any.whl.metadata (5.2 kB)
15.70 Collecting langgraph-sdk<0.4.0,>=0.3.0 (from langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
15.74   Downloading langgraph_sdk-0.3.1-py3-none-any.whl.metadata (1.6 kB)
15.88 Collecting xxhash>=3.5.0 (from langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
15.93   Downloading xxhash-3.6.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (13 kB)
16.06 Collecting ormsgpack>=1.12.0 (from langgraph-checkpoint<4.0.0,>=2.1.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
16.11   Downloading ormsgpack-1.12.1-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (3.2 kB)
16.20 Collecting httpx>=0.25.2 (from langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
16.25   Downloading httpx-0.28.1-py3-none-any.whl.metadata (7.1 kB)
16.57 Collecting orjson>=3.10.1 (from langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
16.62   Downloading orjson-3.11.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (41 kB)
16.71 Collecting requests-toolbelt>=1.0.0 (from langsmith<1.0.0,>=0.3.45->langchain-core>=0.1.0->-r requirements.txt (line 5))
16.76   Downloading requests_toolbelt-1.0.0-py2.py3-none-any.whl.metadata (14 kB)
16.89 Collecting zstandard>=0.23.0 (from langsmith<1.0.0,>=0.3.45->langchain-core>=0.1.0->-r requirements.txt (line 5))
16.93   Downloading zstandard-0.25.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (3.3 kB)
17.09 Collecting anyio (from httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
17.13   Downloading anyio-4.12.0-py3-none-any.whl.metadata (4.3 kB)
17.23 Collecting certifi (from httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
17.27   Downloading certifi-2025.11.12-py3-none-any.whl.metadata (2.5 kB)
17.36 Collecting httpcore==1.* (from httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
17.40   Downloading httpcore-1.0.9-py3-none-any.whl.metadata (21 kB)
17.49 Collecting idna (from httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
17.54   Downloading idna-3.11-py3-none-any.whl.metadata (8.4 kB)
17.66 Collecting h11>=0.16 (from httpcore==1.*->httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
17.71   Downloading h11-0.16.0-py3-none-any.whl.metadata (8.3 kB)
17.79 Collecting annotated-types>=0.6.0 (from pydantic<3.0.0,>=2.7.4->langchain>=0.1.0->-r requirements.txt (line 3))
17.83   Downloading annotated_types-0.7.0-py3-none-any.whl.metadata (15 kB)
18.99 Collecting pydantic-core==2.41.5 (from pydantic<3.0.0,>=2.7.4->langchain>=0.1.0->-r requirements.txt (line 3))
19.04   Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (7.3 kB)
19.12 Collecting typing-inspection>=0.4.2 (from pydantic<3.0.0,>=2.7.4->langchain>=0.1.0->-r requirements.txt (line 3))
19.16   Downloading typing_inspection-0.4.2-py3-none-any.whl.metadata (2.6 kB)
19.25 Collecting langchain-classic<2.0.0,>=1.0.0 (from langchain-community>=0.0.10->-r requirements.txt (line 4))
19.29   Downloading langchain_classic-1.0.1-py3-none-any.whl.metadata (4.2 kB)
19.62 Collecting SQLAlchemy<3.0.0,>=1.4.0 (from langchain-community>=0.0.10->-r requirements.txt (line 4))
19.67   Downloading sqlalchemy-2.0.45-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (9.5 kB)
20.18 Collecting aiohttp<4.0.0,>=3.8.3 (from langchain-community>=0.0.10->-r requirements.txt (line 4))
20.23   Downloading aiohttp-3.13.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (8.1 kB)
20.33 Collecting pydantic-settings<3.0.0,>=2.10.1 (from langchain-community>=0.0.10->-r requirements.txt (line 4))
20.37   Downloading pydantic_settings-2.12.0-py3-none-any.whl.metadata (3.4 kB)
20.56 Collecting httpx-sse<1.0.0,>=0.4.0 (from langchain-community>=0.0.10->-r requirements.txt (line 4))
20.61   Downloading httpx_sse-0.4.3-py3-none-any.whl.metadata (9.7 kB)
20.79 Collecting charset_normalizer<4,>=2 (from requests>=2.31.0->-r requirements.txt (line 22))
20.83   Downloading charset_normalizer-3.4.4-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (37 kB)
20.95 Collecting urllib3<3,>=1.21.1 (from requests>=2.31.0->-r requirements.txt (line 22))
20.99   Downloading urllib3-2.6.2-py3-none-any.whl.metadata (6.6 kB)
21.09 Collecting aiohappyeyeballs>=2.5.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
21.15   Downloading aiohappyeyeballs-2.6.1-py3-none-any.whl.metadata (5.9 kB)
21.23 Collecting aiosignal>=1.4.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
21.28   Downloading aiosignal-1.4.0-py3-none-any.whl.metadata (3.7 kB)
21.36 Collecting async-timeout<6.0,>=4.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
21.40   Downloading async_timeout-5.0.1-py3-none-any.whl.metadata (5.1 kB)
21.48 Collecting attrs>=17.3.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
21.52   Downloading attrs-25.4.0-py3-none-any.whl.metadata (10 kB)
21.65 Collecting frozenlist>=1.1.1 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
21.71   Downloading frozenlist-1.8.0-cp310-cp310-manylinux1_x86_64.manylinux_2_28_x86_64.manylinux_2_5_x86_64.whl.metadata (20 kB)
21.97 Collecting multidict<7.0,>=4.5 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
22.02   Downloading multidict-6.7.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.3 kB)
22.14 Collecting propcache>=0.2.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
22.19   Downloading propcache-0.4.1-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (13 kB)
22.48 Collecting yarl<2.0,>=1.17.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
22.52   Downloading yarl-1.22.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (75 kB)
22.64 Collecting marshmallow<4.0.0,>=3.18.0 (from dataclasses-json->unstructured==0.10.30->-r requirements.txt (line 13))
22.69   Downloading marshmallow-3.26.2-py3-none-any.whl.metadata (7.3 kB)
22.79 Collecting typing-inspect<1,>=0.4.0 (from dataclasses-json->unstructured==0.10.30->-r requirements.txt (line 13))
22.85   Downloading typing_inspect-0.9.0-py3-none-any.whl.metadata (1.5 kB)
22.85 Collecting async-timeout<6.0,>=4.0 (from aiohttp<4.0.0,>=3.8.3->langchain-community>=0.0.10->-r requirements.txt (line 4))
22.91   Downloading async_timeout-4.0.3-py3-none-any.whl.metadata (4.2 kB)
23.04 Collecting python-dotenv>=0.21.0 (from pydantic-settings<3.0.0,>=2.10.1->langchain-community>=0.0.10->-r requirements.txt (line 4))
23.09   Downloading python_dotenv-1.2.1-py3-none-any.whl.metadata (25 kB)
23.29 Collecting greenlet>=1 (from SQLAlchemy<3.0.0,>=1.4.0->langchain-community>=0.0.10->-r requirements.txt (line 4))
23.34   Downloading greenlet-3.3.0-cp310-cp310-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (4.1 kB)
23.42 Collecting mypy-extensions>=0.3.0 (from typing-inspect<1,>=0.4.0->dataclasses-json->unstructured==0.10.30->-r requirements.txt (line 13))
23.46   Downloading mypy_extensions-1.1.0-py3-none-any.whl.metadata (1.1 kB)
23.59 Collecting tqdm (from sentence-transformers>=2.2.2->-r requirements.txt (line 7))
23.64   Downloading tqdm-4.67.1-py3-none-any.whl.metadata (57 kB)
23.79 Collecting scikit-learn (from sentence-transformers>=2.2.2->-r requirements.txt (line 7))
23.96   Downloading scikit_learn-1.7.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (11 kB)
24.17 Collecting scipy (from sentence-transformers>=2.2.2->-r requirements.txt (line 7))
24.23   Downloading scipy-1.15.3-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (61 kB)
24.34 Collecting filelock (from transformers>=4.36.0->-r requirements.txt (line 18))
24.38   Downloading filelock-3.20.1-py3-none-any.whl.metadata (2.1 kB)
24.39 Collecting huggingface_hub>=0.19.0 (from -r requirements.txt (line 23))
24.43   Downloading huggingface_hub-0.36.0-py3-none-any.whl.metadata (14 kB)
24.83 Collecting regex!=2019.12.17 (from transformers>=4.36.0->-r requirements.txt (line 18))
24.87   Downloading regex-2025.11.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (40 kB)
25.06 Collecting tokenizers<=0.23.0,>=0.22.0 (from transformers>=4.36.0->-r requirements.txt (line 18))
25.11   Downloading tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (6.8 kB)
25.30 Collecting safetensors>=0.4.3 (from transformers>=4.36.0->-r requirements.txt (line 18))
25.34   Downloading safetensors-0.7.0-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.1 kB)
25.45 Collecting fsspec>=2023.5.0 (from huggingface_hub>=0.19.0->-r requirements.txt (line 23))
25.49   Downloading fsspec-2025.12.0-py3-none-any.whl.metadata (10 kB)
25.63 Collecting hf-xet<2.0.0,>=1.1.3 (from huggingface_hub>=0.19.0->-r requirements.txt (line 23))
25.68   Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (4.9 kB)
25.88 Collecting unsloth_zoo>=2025.12.7 (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
25.93   Downloading unsloth_zoo-2025.12.7-py3-none-any.whl.metadata (32 kB)
26.03 Collecting wheel>=0.42.0 (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
26.08   Downloading wheel-0.45.1-py3-none-any.whl.metadata (2.3 kB)
26.21 Collecting torchvision (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
26.26   Downloading torchvision-0.24.1-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (5.9 kB)
26.45 Collecting tyro (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
26.50   Downloading tyro-1.0.3-py3-none-any.whl.metadata (12 kB)
26.72 Collecting protobuf (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
26.76   Downloading protobuf-6.33.2-cp39-abi3-manylinux2014_x86_64.whl.metadata (593 bytes)
26.86 Collecting triton>=3.0.0 (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
26.90   Downloading triton-3.5.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (1.7 kB)
27.01 Collecting sentencepiece>=0.2.0 (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
27.05   Downloading sentencepiece-0.2.1-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (10 kB)
27.06 Collecting datasets>=2.14.0 (from -r requirements.txt (line 17))
27.11   Downloading datasets-4.3.0-py3-none-any.whl.metadata (18 kB)
27.23 Collecting hf_transfer (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
27.27   Downloading hf_transfer-0.1.9-cp38-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl.metadata (1.7 kB)
27.36 Collecting diffusers (from unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
27.55   Downloading diffusers-0.36.0-py3-none-any.whl.metadata (20 kB)
27.59 Collecting trl>=0.7.0 (from -r requirements.txt (line 15))
27.64   Downloading trl-0.24.0-py3-none-any.whl.metadata (11 kB)
27.85 Collecting pyarrow>=21.0.0 (from datasets>=2.14.0->-r requirements.txt (line 17))
27.90   Downloading pyarrow-22.0.0-cp310-cp310-manylinux_2_28_x86_64.whl.metadata (3.2 kB)
27.98 Collecting dill<0.4.1,>=0.3.0 (from datasets>=2.14.0->-r requirements.txt (line 17))
28.03   Downloading dill-0.4.0-py3-none-any.whl.metadata (10 kB)
28.12 Collecting multiprocess<0.70.17 (from datasets>=2.14.0->-r requirements.txt (line 17))
28.17   Downloading multiprocess-0.70.16-py310-none-any.whl.metadata (7.2 kB)
28.17 Collecting fsspec>=2023.5.0 (from huggingface_hub>=0.19.0->-r requirements.txt (line 23))
28.22   Downloading fsspec-2025.9.0-py3-none-any.whl.metadata (10 kB)
28.37 Collecting sympy>=1.13.3 (from torch>=2.0.0->-r requirements.txt (line 24))
28.42   Downloading sympy-1.14.0-py3-none-any.whl.metadata (12 kB)
28.51 Collecting networkx>=2.5.1 (from torch>=2.0.0->-r requirements.txt (line 24))
28.55   Downloading networkx-3.4.2-py3-none-any.whl.metadata (6.3 kB)
28.64 Collecting nvidia-cuda-nvrtc-cu12==12.8.93 (from torch>=2.0.0->-r requirements.txt (line 24))
28.68   Downloading nvidia_cuda_nvrtc_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
28.76 Collecting nvidia-cuda-runtime-cu12==12.8.90 (from torch>=2.0.0->-r requirements.txt (line 24))
28.82   Downloading nvidia_cuda_runtime_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
28.90 Collecting nvidia-cuda-cupti-cu12==12.8.90 (from torch>=2.0.0->-r requirements.txt (line 24))
28.94   Downloading nvidia_cuda_cupti_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
29.04 Collecting nvidia-cudnn-cu12==9.10.2.21 (from torch>=2.0.0->-r requirements.txt (line 24))
29.08   Downloading nvidia_cudnn_cu12-9.10.2.21-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
29.17 Collecting nvidia-cublas-cu12==12.8.4.1 (from torch>=2.0.0->-r requirements.txt (line 24))
29.21   Downloading nvidia_cublas_cu12-12.8.4.1-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
29.29 Collecting nvidia-cufft-cu12==11.3.3.83 (from torch>=2.0.0->-r requirements.txt (line 24))
29.34   Downloading nvidia_cufft_cu12-11.3.3.83-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
29.44 Collecting nvidia-curand-cu12==10.3.9.90 (from torch>=2.0.0->-r requirements.txt (line 24))
29.48   Downloading nvidia_curand_cu12-10.3.9.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.7 kB)
29.56 Collecting nvidia-cusolver-cu12==11.7.3.90 (from torch>=2.0.0->-r requirements.txt (line 24))
29.61   Downloading nvidia_cusolver_cu12-11.7.3.90-py3-none-manylinux_2_27_x86_64.whl.metadata (1.8 kB)
29.69 Collecting nvidia-cusparse-cu12==12.5.8.93 (from torch>=2.0.0->-r requirements.txt (line 24))
29.75   Downloading nvidia_cusparse_cu12-12.5.8.93-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
29.83 Collecting nvidia-cusparselt-cu12==0.7.1 (from torch>=2.0.0->-r requirements.txt (line 24))
29.88   Downloading nvidia_cusparselt_cu12-0.7.1-py3-none-manylinux2014_x86_64.whl.metadata (7.0 kB)
29.96 Collecting nvidia-nccl-cu12==2.27.5 (from torch>=2.0.0->-r requirements.txt (line 24))
30.00   Downloading nvidia_nccl_cu12-2.27.5-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.0 kB)
30.08 Collecting nvidia-nvshmem-cu12==3.3.20 (from torch>=2.0.0->-r requirements.txt (line 24))
30.12   Downloading nvidia_nvshmem_cu12-3.3.20-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (2.1 kB)
30.20 Collecting nvidia-nvtx-cu12==12.8.90 (from torch>=2.0.0->-r requirements.txt (line 24))
30.27   Downloading nvidia_nvtx_cu12-12.8.90-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.8 kB)
30.36 Collecting nvidia-nvjitlink-cu12==12.8.93 (from torch>=2.0.0->-r requirements.txt (line 24))
30.41   Downloading nvidia_nvjitlink_cu12-12.8.93-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl.metadata (1.7 kB)
30.48 Collecting nvidia-cufile-cu12==1.13.1.3 (from torch>=2.0.0->-r requirements.txt (line 24))
30.53   Downloading nvidia_cufile_cu12-1.13.1.3-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl.metadata (1.7 kB)
30.72 Collecting MarkupSafe>=2.0 (from Jinja2>=3.1.2->Flask==3.0.0->-r requirements.txt (line 1))
30.76   Downloading markupsafe-3.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (2.7 kB)
30.86 Collecting six>=1.5 (from python-dateutil>=2.8.2->pandas==2.1.4->-r requirements.txt (line 12))
30.90   Downloading six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
31.00 Collecting mpmath<1.4,>=1.1.0 (from sympy>=1.13.3->torch>=2.0.0->-r requirements.txt (line 24))
31.04   Downloading mpmath-1.3.0-py3-none-any.whl.metadata (8.6 kB)
31.24 Collecting torchao>=0.13.0 (from unsloth_zoo>=2025.12.7->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
31.28   Downloading torchao-0.15.0-cp310-abi3-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl.metadata (22 kB)
31.38 Collecting cut_cross_entropy (from unsloth_zoo>=2025.12.7->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
31.43   Downloading cut_cross_entropy-25.1.1-py3-none-any.whl.metadata (9.3 kB)
31.63 Collecting pillow (from unsloth_zoo>=2025.12.7->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
31.68   Downloading pillow-12.0.0-cp310-cp310-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.8 kB)
31.80 Collecting msgspec (from unsloth_zoo>=2025.12.7->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
31.84   Downloading msgspec-0.20.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl.metadata (5.5 kB)
31.94 Collecting exceptiongroup>=1.0.2 (from anyio->httpx>=0.25.2->langgraph-sdk<0.4.0,>=0.3.0->langgraph<1.1.0,>=1.0.2->langchain>=0.1.0->-r requirements.txt (line 3))
32.00   Downloading exceptiongroup-1.3.1-py3-none-any.whl.metadata (6.7 kB)
32.09 Collecting soupsieve>=1.6.1 (from beautifulsoup4->unstructured==0.10.30->-r requirements.txt (line 13))
32.13   Downloading soupsieve-2.8.1-py3-none-any.whl.metadata (4.6 kB)
32.24 Collecting importlib_metadata (from diffusers->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
32.28   Downloading importlib_metadata-8.7.1-py3-none-any.whl.metadata (4.7 kB)
32.38 Collecting zipp>=3.20 (from importlib_metadata->diffusers->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
32.43   Downloading zipp-3.23.0-py3-none-any.whl.metadata (3.6 kB)
32.53 Collecting joblib (from nltk->unstructured==0.10.30->-r requirements.txt (line 13))
32.57   Downloading joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
32.69 Collecting threadpoolctl>=3.1.0 (from scikit-learn->sentence-transformers>=2.2.2->-r requirements.txt (line 7))
32.73   Downloading threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
32.84 Collecting docstring-parser>=0.15 (from tyro->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
32.88   Downloading docstring_parser-0.17.0-py3-none-any.whl.metadata (3.5 kB)
32.97 Collecting typeguard>=4.0.0 (from tyro->unsloth>=2024.8->unsloth[colab-new]>=2024.8->-r requirements.txt (line 14))
33.02   Downloading typeguard-4.4.4-py3-none-any.whl.metadata (3.3 kB)
33.03 WARNING: unsloth 2025.12.9 does not provide the extra 'triton'
33.14 Downloading flask-3.0.0-py3-none-any.whl (99 kB)
33.20 Downloading Flask_Cors-4.0.0-py2.py3-none-any.whl (14 kB)
33.26 Downloading faiss_cpu-1.13.1-cp310-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (23.7 MB)
37.76    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 23.7/23.7 MB 5.3 MB/s  0:00:04
37.81 Downloading pypdf-3.17.0-py3-none-any.whl (277 kB)
37.94 Downloading python_docx-1.1.0-py3-none-any.whl (239 kB)
38.02 Downloading openpyxl-3.1.2-py2.py3-none-any.whl (249 kB)
38.11 Downloading pandas-2.1.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (12.3 MB)
40.55    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.3/12.3 MB 6.3 MB/s  0:00:02
40.60 Downloading unstructured-0.10.30-py3-none-any.whl (1.7 MB)
40.86    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 6.7 MB/s  0:00:00
40.91 Downloading numpy-1.26.4-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (18.2 MB)
44.56    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.2/18.2 MB 5.0 MB/s  0:00:03
44.61 Downloading langchain-1.2.0-py3-none-any.whl (102 kB)
44.67 Downloading langchain_core-1.2.5-py3-none-any.whl (484 kB)
44.81 Downloading jsonpatch-1.33-py2.py3-none-any.whl (12 kB)
44.85 Downloading langgraph-1.0.5-py3-none-any.whl (157 kB)
44.92 Downloading langgraph_checkpoint-3.0.1-py3-none-any.whl (46 kB)
44.96 Downloading langgraph_prebuilt-1.0.5-py3-none-any.whl (35 kB)
45.01 Downloading langgraph_sdk-0.3.1-py3-none-any.whl (66 kB)
45.07 Downloading langsmith-0.5.1-py3-none-any.whl (275 kB)
45.15 Downloading httpx-0.28.1-py3-none-any.whl (73 kB)
45.21 Downloading httpcore-1.0.9-py3-none-any.whl (78 kB)
45.26 Downloading packaging-25.0-py3-none-any.whl (66 kB)
45.32 Downloading pydantic-2.12.5-py3-none-any.whl (463 kB)
45.44 Downloading pydantic_core-2.41.5-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.1 MB)
45.80    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.1/2.1 MB 5.5 MB/s  0:00:00
45.85 Downloading pyyaml-6.0.3-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (770 kB)
45.97    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 770.3/770.3 kB 6.2 MB/s  0:00:00
46.01 Downloading tenacity-9.1.2-py3-none-any.whl (28 kB)
46.07 Downloading typing_extensions-4.15.0-py3-none-any.whl (44 kB)
46.13 Downloading uuid_utils-0.12.0-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (343 kB)
46.24 Downloading langchain_community-0.4.1-py3-none-any.whl (2.5 MB)
46.64    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.5/2.5 MB 6.2 MB/s  0:00:00
46.69 Downloading requests-2.32.5-py3-none-any.whl (64 kB)
46.74 Downloading aiohttp-3.13.2-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (1.7 MB)
46.99    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.7/1.7 MB 7.6 MB/s  0:00:00
47.05 Downloading charset_normalizer-3.4.4-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (153 kB)
47.16 Downloading dataclasses_json-0.6.7-py3-none-any.whl (28 kB)
47.20 Downloading httpx_sse-0.4.3-py3-none-any.whl (9.0 kB)
47.27 Downloading idna-3.11-py3-none-any.whl (71 kB)
47.32 Downloading langchain_classic-1.0.1-py3-none-any.whl (1.0 MB)
47.54    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.0/1.0 MB 4.6 MB/s  0:00:00
47.58 Downloading langchain_text_splitters-1.1.0-py3-none-any.whl (34 kB)
47.63 Downloading async_timeout-4.0.3-py3-none-any.whl (5.7 kB)
47.68 Downloading marshmallow-3.26.2-py3-none-any.whl (50 kB)
47.72 Downloading multidict-6.7.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (241 kB)
47.82 Downloading pydantic_settings-2.12.0-py3-none-any.whl (51 kB)
47.87 Downloading sqlalchemy-2.0.45-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (3.2 MB)
48.71    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.2/3.2 MB 3.6 MB/s  0:00:00
48.76 Downloading typing_inspect-0.9.0-py3-none-any.whl (8.8 kB)
48.87 Downloading urllib3-2.6.2-py3-none-any.whl (131 kB)
48.94 Downloading yarl-1.22.0-cp310-cp310-manylinux2014_x86_64.manylinux_2_17_x86_64.manylinux_2_28_x86_64.whl (346 kB)
49.05 Downloading sentence_transformers-5.2.0-py3-none-any.whl (493 kB)
49.16 Downloading transformers-4.57.3-py3-none-any.whl (12.0 MB)
51.96    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 12.0/12.0 MB 4.3 MB/s  0:00:02
52.00 Downloading huggingface_hub-0.36.0-py3-none-any.whl (566 kB)
52.09    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 566.1/566.1 kB 6.6 MB/s  0:00:00
52.13 Downloading hf_xet-1.2.0-cp37-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
52.70    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 5.8 MB/s  0:00:00
52.75 Downloading tokenizers-0.22.1-cp39-abi3-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (3.3 MB)
53.31    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.3/3.3 MB 5.9 MB/s  0:00:00
53.37 Downloading unsloth-2025.12.9-py3-none-any.whl (376 kB)
53.48 Downloading trl-0.24.0-py3-none-any.whl (423 kB)
53.60 Downloading datasets-4.3.0-py3-none-any.whl (506 kB)
53.72 Downloading dill-0.4.0-py3-none-any.whl (119 kB)
53.78 Downloading fsspec-2025.9.0-py3-none-any.whl (199 kB)
53.86 Downloading multiprocess-0.70.16-py310-none-any.whl (134 kB)
53.92 Downloading peft-0.18.0-py3-none-any.whl (556 kB)
54.00    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 556.4/556.4 kB 7.3 MB/s  0:00:00
54.05 Downloading accelerate-1.12.0-py3-none-any.whl (380 kB)
54.14 Downloading bitsandbytes-0.49.0-py3-none-manylinux_2_24_x86_64.whl (59.1 MB)
69.65    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 59.1/59.1 MB 3.8 MB/s  0:00:15
69.69 Downloading torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl (899.8 MB)
208.0    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━             630.5/899.8 MB 5.5 MB/s eta 0:00:50
208.0 WARNING: Connection timed out while downloading.
208.0 WARNING: Attempting to resume incomplete download (630.5 MB/899.8 MB, attempt 1)
223.0 WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d2421f2c20>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
233.5 WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d241d402e0>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
234.5 WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d240fe9210>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
236.5 WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d22eaa8850>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
240.5 WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d241ce4490>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
249.5 Resuming download torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl (630.5 MB/899.8 MB)
279.1    ━━━━━━━━━━━━━━━━━━━━━━━━━━━             631.5/899.8 MB 105.7 kB/s eta 0:42:20
279.1 WARNING: Connection timed out while downloading.
279.1 WARNING: Attempting to resume incomplete download (631.5 MB/899.8 MB, attempt 2)
284.1 WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d242239030>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
284.6 WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23ff066e0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
285.6 WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d240ade830>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
287.7 WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d2402451e0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
306.0 Resuming download torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl (631.5 MB/899.8 MB)
377.6    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    833.9/899.8 MB 4.3 MB/s eta 0:00:16
377.6 WARNING: Connection timed out while downloading.
377.6 WARNING: Attempting to resume incomplete download (833.9 MB/899.8 MB, attempt 3)
387.6 WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d240061870>: Failed to establish a new connection: [Errno -3] Temporary failure in name resolution')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
496.9 WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'ReadTimeoutError("HTTPSConnectionPool(host='files.pythonhosted.org', port=443): Read timed out. (read timeout=15)")': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
503.0 WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d93e380>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
505.0 WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d93dd80>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
509.0 WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23dc810f0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
509.0 WARNING: Attempting to resume incomplete download (833.9 MB/899.8 MB, attempt 4)
509.0 WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23f26d990>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
509.5 WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23f26e080>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
510.5 WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d912260>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
512.5 WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d910850>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
516.5 WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d911d50>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
521.5 WARNING: Attempting to resume incomplete download (833.9 MB/899.8 MB, attempt 5)
521.5 WARNING: Retrying (Retry(total=4, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23dc80d30>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
522.0 WARNING: Retrying (Retry(total=3, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d23d93e3b0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
523.1 WARNING: Retrying (Retry(total=2, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d2401bdd20>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
525.1 WARNING: Retrying (Retry(total=1, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d2401bdb40>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
529.1 WARNING: Retrying (Retry(total=0, connect=None, read=None, redirect=None, status=None)) after connection broken by 'NewConnectionError('<pip._vendor.urllib3.connection.HTTPSConnection object at 0x77d240599de0>: Failed to establish a new connection: [Errno -2] Name or service not known')': /packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
529.1 error: incomplete-download
529.1
529.1 × Download failed after 6 attempts because not enough bytes were received (833.9 MB/899.8 MB)
529.1 ╰─> URL: https://files.pythonhosted.org/packages/38/45/be5a74f221df8f4b609b78ff79dc789b0cc9017624544ac4dd1c03973150/torch-2.9.1-cp310-cp310-manylinux_2_28_x86_64.whl
529.1
529.1 note: This is an issue with network connectivity, not pip.
529.1 hint: Use --resume-retries to configure resume attempt limit.
------
Dockerfile:31

--------------------

  30 |     # Install Python dependencies

  31 | >>> RUN pip3 install --no-cache-dir --upgrade pip && \

  32 | >>>     pip3 install --no-cache-dir -r requirements.txt

  33 |

--------------------

target backend: failed to solve: process "/bin/sh -c pip3 install --no-cache-dir --upgrade pip &&     pip3 install --no-cache-dir -r requirements.txt" did not complete successfully: exit code: 1



View build details: docker-desktop://dashboard/build/default/default/wcff7o2onb4me9353g12o3hn3