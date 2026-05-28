4 Experiments
This section analyzes two experiments conducted using LLM-memory based benchmarks. The first evaluation employs the Deep Memory Retrieval (DMR) task developed in ~\cite{bib.bib3} which uses a 500-conversation subset of the Multi-Session Chat dataset introduced in "Beyond Goldfish Memory: Long-Term Open-Domain Conversation" ~\cite{bib.bib22}. The second evaluation utilizes the LongMemEval benchmark from "LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory" ~\cite{bib.bib7}. Specifically we use the LongMemEvals dataset which provides an extensive conversation context of on average 115000 tokens.
 For both experiments we integrate the conversation history into a Zep knowledge graph through Zep’s APIs. We then retrieve the 20 most relevant edges (facts) and entity nodes (entity summaries) using the techniques described in Section 3. The system reformats this data into a context string matching the functionality provided by Zep’s memory APIs.
 While these experiments demonstrate key retrieval capabilities of Graphiti they represent a subset of the system’s full search functionality. This focused scope enables clear comparison with existing benchmarks while reserving the exploration of additional knowledge graph capabilities for future work.


## Section References
[bib.bib3] [3] Charles Packer Sarah Wooders Kevin Lin Vivian Fang Shishir G. Patil Ion Stoica and Joseph E. Gonzalez. Memgpt: Towards llms as operating systems 2024.
[bib.bib22] [22] Jing Xu Arthur Szlam and Jason Weston. Beyond goldfish memory: Long-term open-domain conversation 2021.
[bib.bib7] [7] Di Wu Hongwei Wang Wenhao Yu Yuwei Zhang Kai-Wei Chang and Dong Yu. Longmemeval: Benchmarking chat assistants on long-term interactive memory 2024.