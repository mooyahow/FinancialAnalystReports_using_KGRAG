## Project Overview
To effectively reflect the complex information structure and domain-specific characteristics of securities firm reports,
this project proposes a **KG-RAG (Knowledge Graph–augmented Retrieval-Augmented Generation)** approach for a report-based
question answering service.

## Goal
- Propose and evaluate a **Q&A service for securities firm research reports** using **KG-RAG**.

## Background
- Investment decision-making often requires specialized knowledge about markets and companies.
- Securities firms publish research reports, but these reports contain large volumes of information and are time-consuming
  to navigate.
- There is a need for a RAG system that can provide **expert-level answers to complex queries**, improving the efficiency
  of investment-related decision support.

## Why KG-RAG
KG-RAG is beneficial because it can:
- Automatically extract and structure **triples (subject–relation–object)** from both structured and unstructured data.
- Retrieve **top-k triples** based on semantic similarity between the query and triples, and further explore related
  knowledge via **neighbor expansion** in a graph database.

## Method
1. **Collect reports** published on official websites of multiple securities firms.
2. Convert **PDF → text**, then extract both **body text and tables**.
3. Use **Google Gemini 2.5 Pro** to extract **triples (subject–relation–object)**.
4. Store extracted triples in **Neo4j**.
   - For comparison, embed triples and store them in **ChromaDB**.
5. Compare **naive RAG** vs **KG-RAG** for question answering performance.

## Results & Findings
- For evaluations focused on **report narrative/body text**, **KG-RAG** performed better.
- For questions centered on **numerical and quantitative information**, **naive RAG** performed better.
- Overall, KG-RAG is effective for queries that require **connecting related pieces of information**, but it may suffer
  performance degradation when handling **highly structured/tabular numeric information**, indicating a limitation in
  structured data representation and retrieval.