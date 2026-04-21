# CaseFacts: A Benchmark for Legal Fact-Checking and Precedent Retrieval

**To appear in ACL 2026 Main Conference**

[![arXiv](https://img.shields.io/badge/arXiv-2601.17230-b31b1b.svg)](https://arxiv.org/abs/2601.17230) [![HuggingFace Dataset](https://img.shields.io/badge/🤗%20HuggingFace-Dataset-orange)](https://huggingface.co/datasets/IDIRLab/CaseFacts) [![ACL 2026](https://img.shields.io/badge/ACL-2026-blue)](https://2026.aclweb.org/) [![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

📄 **Paper:** [CaseFacts: A Benchmark for Legal Fact-Checking and Precedent Retrieval](https://arxiv.org/abs/2601.17230)

**Authors:** Akshith Reddy Putta, Jacob Devasier, Chengkai Li

## Abstract

Automated Fact-Checking has largely focused on verifying general knowledge against static corpora, overlooking high-stakes domains like law where truth is evolving and technically complex. We introduce **CaseFacts**, a benchmark for verifying colloquial legal claims against U.S. Supreme Court precedents. Unlike existing resources that map formal texts to formal texts, CaseFacts challenges systems to bridge the semantic gap between layperson assertions and technical jurisprudence while accounting for temporal validity. The dataset consists of 6,294 claims categorized as Supported, Refuted, or Overruled. We construct this benchmark using a multi-stage pipeline that leverages Large Language Models (LLMs) to synthesize claims from expert case summaries, employing a novel semantic similarity heuristic to efficiently identify and verify complex legal overrulings. Experiments with state-of-the-art LLMs reveal that the task remains challenging; notably, augmenting models with unrestricted web search degrades performance compared to closed-book baselines due to the retrieval of noisy, non-authoritative precedents. We release CaseFacts to spur research into legal fact verification systems.

## Dataset

The dataset is located in the `dataset` directory. This contains the full train and test sets, along with the set of Supreme Court cases used in the project.

The dataset can also be downloaded from [HuggingFace](https://huggingface.co/datasets/IDIRLab/CaseFacts).

## Dataset Statistics

| Split | Total | Supported | Refuted | Overruled |
|-------|-------|-----------|---------|-----------|
| Train | 5,794 | 2,605 | 2,732 | 457 |
| Test  | 500   | 280       | 177     | 43        |

## Requirements

Python version: 3.11

Install dependencies with:

```bash
pip install -r requirements.txt
```
