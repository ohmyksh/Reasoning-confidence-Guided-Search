# Reasoning-Confidence Guided Search For Multi-Hop Question Answering
This is my research project for POSTECH CSED499A.
* Advised by Wook-Shin Han 
* Data Systems Lab @ POSTECH    

## Introduction
This research aimed to develop a system where a large language model (LLM) performs multi-hop question answering (QA).  

<img src="framework.jpg" width="70%" alt="Framework"></img> 
  
Large language models (LLMs) are trained on extensive real-world knowledge and show human-like performance on various benchmarks. However, complex real-world question answering remains a challenge. Wei et al. proposed Chain-of-Thought prompting to enhance step-by-step reasoning by using examples with problems and solutions. This method relies on all necessary information being provided as input or stored in the model’s parameters. For complex multi-hop question answering, not all required knowledge is available in the input, and parametric knowledge may not be up-to-date.  
To address this, Retrieval-Augmented Generation (RAG) uses non-parametric knowledge from external sources, which can be updated easily and improve accuracy. However, a single retrieval approach has limitations, as multi-hop question answering may require information not directly related to the query. Multiple retrievals are needed to search various documents for accurate responses. Therefore, this research aims to advance the RAG framework to perform multi-hop question answering using LLMs.

## Baseline and State-of-the-Art Methods
- **Baseline**: IRCoT adopts a simple and fixed approach to retrieving documents for multi-hop question answering. Each time the large language model generates the next reasoning sentence based on the prompt, it always performs a search, using the most recently generated sentence directly as the query. 
[IRCoT (ACL 2023)](https://arxiv.org/abs/2212.10509)
- **State-of-the-Art (SOTA)**: DRAGIN aims to dynamically determine when to retrieve and what to retrieve during the generation process. It criticizes the existing IRCoT method for being inefficient in terms of time and cost because it always performs a search without considering the necessity of it. Additionally, using the generated sentence as the search query might input irrelevant and unnecessary information into the model, affecting the accuracy of the generated response. Therefore, considering these two elements is crucial for performance and efficiency.
[DRAGIN (ACL 2024)](https://arxiv.org/abs/2403.10081)

<img src="compare.png" width="55%" alt="Comparison"></img>  

### Analysis of Issues in SOTA Models

1. **Determining When to Retrieve**:
The DRAGIN framework identifies hallucination in the results generated by large language models (LLMs) by calculating token generation probabilities, subsequent impact on generation results, and semantic importance for each token in the generated output. Retrieval is triggered when these scores exceed predefined thresholds, operating under the assumption that higher internal uncertainty within the LLM corresponds to hallucination. However, hallucination occurs when an LLM generates incorrect answers with false confidence, believing them to be true. This phenomenon is difficult to judge based solely on internal probabilities. Therefore, this method does not cover all cases of hallucination, indicating the need for alternative approaches to detect hallucination.

2. **Constructing Search Queries**:
When retrieval is deemed necessary, DRAGIN constructs queries by selecting the top n tokens with the highest relevance to the generation of the token in question from the previously generated tokens. While the autoregressive nature of LLMs relies on preceding tokens to generate the next ones, this approach overlooks the possibility of relevant tokens appearing later in the generated sequence. Consequently, it fails to consider the entire context, thus limiting the comprehensiveness of the queries.

Recognizing these limitations highlights the potential for enhancing both the determination of when retrieval is necessary and the construction of effective search queries that fully consider the semantic context.


## Proposed Approach
The multi-hop QA task can be structured as a search process where the model traverses inference states to reach an answer. If hallucination is detected in the generation result, the model backtracks to regenerate correct inferences.

Starting from the initial query, the model progresses through intermediate states, generating a final response by traversing through these states. The detailed process of moving from one state to the next can be seen in below Figure. If hallucination is not detected in the generated results at a particular state, the model moves on to the next state and continues generating until it meets the termination condition. However, if hallucination is detected, the model constructs a search query to retrieve the necessary documents. These documents are then incorporated into the model's input, and the generation process is repeated.

<img src="framework.jpg" width="70%" alt="Framework"></img>  

### Improvements in Query Construction
This research focused on improving the query construction process. The existing SOTA model extracted and concatenated important tokens from previously generated tokens to form a query. This approach failed to capture the semantic elements of the token for which information was sought. For example, consider a situation where incorrect information about the university where Einstein sought a job was generated. Listing the relevant tokens from those preceding the incorrectly specified university name would form a query as shown in the figure below.

<img src="example-sota-query.png" width="65%" alt="sota-query"></img> 

Considering only the preceding context and using a simple sequence of tokens approach fails to account for the semantic information that reflects the intent of the initial query and the context of the generated results. To address this, I devised and applied two query construction methods:

- **Attention to All Context**: 
Attention weights reflect the model's evaluation of the importance of each token in relation to both preceding and following context. This means the model assigns weights based on how crucial each word or phrase is within the context. By doing so, the model better maintains contextual coherence between the query and the generated results.

- **LLM Generated Query**: 
The LLM generates queries based on the given context. Given that LLMs are trained on vast amounts of data, they excel in understanding context and generating semantically relevant responses. Query expansion using LLMs has been proven effective in previous research, demonstrating their capability in reformulating or creating new queries. This method allows for the generation of more meaningful queries that consider both the intent of the initial query and the context.

<img src="zeroshot-query.png" width="70%" alt="zeroshot"></img> 

## Result & Error Case Analysis

### Result
I conducted experiments comparing query construction methods on two representative multi-hop question answering benchmarks using the Llama-8B-instruct model and BM25 as the retriever. As there was no significant performance change, I further analyzed the error cases.

<img src="result.png" width="60%" alt="result"></img> 

### Error Case
I identified two main error sources.

- **the issue of hallucination detection**:
There were cases where hallucinations in the generated answers were not corrected. The hallucination detection issue can be classified into factual hallucinations and faithfulness hallucinations.

- **the issue of metrics**:
There were problems where the generated answers were marked incorrect during evaluation, even though they had the same meaning as the actual correct answers. This includes cases where the generated answer’s meaning is the same, but differences in expression lead the evaluation metric to recognize it as incorrect.

The statistics of 50 results categorized according to the above issues are shown in the following graph. Therefore, it was confirmed that the current framework fails to accurately detect hallucinations and correct errors in the generated results, leading to incorrect final responses.

<img src="error-case.png" width="40%" alt="error-case"></img> 
<img src="error.png" width="30%" alt="error"></img> 


## Future Work

This research aims to advance in two main aspects based on the analysis of error cases. The core issue identified is the hallucination detection problem. Therefore, it is crucial to improve the detection of hallucinations to decide when to retrieve information and also address potential issues in the retrieval process even when hallucinations are detected.

- **Hallucination Detection**  
As analyzed in the error cases, the current methodology fails to correctly detect and rectify hallucinations in the results generated by large language models. Existing methods rely on internal calculations such as probabilities and attention scores to judge hallucinations. However, these methods have significant limitations in accurately detecting and correcting hallucinations. To improve hallucination detection, the system can compare the generated results not only with internal model calculations but also with real-world knowledge. Retrieval involves finding documents similar to the query from a subset of the entire database. If no documents similar to the query exist in the database that contains comprehensive real-world knowledge, it may indicate hallucination in the generation process. Thus, the absence of similar documents in the database can be used to detect hallucinations by measuring the low similarity of the results. We plan to develop methods within this framework to better detect hallucinations, referencing existing methodologies on hallucination detection .

- **Issues in the Retrieval Process**  
Firstly, it is essential to construct queries to search for the necessary information to accurately regenerate the hallucinated parts. Based on this research, comparative results on query construction methods have been obtained, indicating that while a sparse retriever (BM25) was used, applying a dense retriever could be beneficial. Secondly, even with a well-constructed query, issues with the retriever might prevent it from finding relevant documents from external databases. Further experiments are needed to analyze these issues.

By addressing these aspects, I aim to enhance the overall framework's capability in effectively detecting and correcting hallucinations, thus improving the accuracy and reliability of multi-hop QA tasks.
