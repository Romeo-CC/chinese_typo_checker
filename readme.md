# Chinese Hanzi Typo Checker

## Introduction
In Chinese text scenarios, typos are inevitable due to the input method, as most Chinese characters are entered using pinyin. Additionally, Chinese characters are phono-semantic compounds (形声字), which further contributes to typo occurrences.

Therefore, techniques for detecting and correcting typos in Chinese text are highly valuable and in demand. This repository aims to build a Chinese typo checker by leveraging the capabilities of masked language models like BERT.


```mermaid
%%{init: {'theme':'default'}}%%
flowchart LR
    
    A@{ shape: lean-r, label: "Input Chinese Text" } --> B@{ shape: rect, label: "Tokenizer" }
    
    B -->|"input_ids<br>attention_mask"| C@{ shape: procs, label: "Stacked Encoder Layers" }
    subgraph Model
    C -->|"last hidden state"| D@{ shape: rect, label: "MaskedLM Head <br>for Typo Correction" }
    C -->|"last hidden state"| E@{ shape: rect, label: "Token CLS Head <br>for Typo Detection" }
    end
    D --> F@{ shape: stadium, label: "Typo Correction" }
    E --> G@{ shape: stadium, label: "Typo Detection" }   
```