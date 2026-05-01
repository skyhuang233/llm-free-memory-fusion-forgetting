1 Introduction
Despite recent advancements in dialogue models based on LLMs for extended contexts~\cite{bib.bib5 bib.bib56} as well as the integration of retrieval augmented generation (RAG) techniques~\cite{bib.bib51 bib.bib47 bib.bib48} there is still a need for thorough evaluation of their efficacy in handling very long conversations. Indeed studies in long-term open-domain dialogues have concentrated on assessing model responses within limited contexts e.g. \sim1K tokens over five chat sessions~\cite{bib.bib58 bib.bib21 bib.bib62}. This long term evaluation is crucial for refining engaging chatbots capable of remembering key information from past interactions to generate empathetic consistent and useful responses.
 To this end we present the first study of very long-term open-domain multi-modal dialogues closely mirroring real-world online interactions collected via a human-machine pipeline where we first use LLM-based generative agents to generate conversations and then ask human annotators to fix any long-term inconsistencies in the conversations. Specifically drawing on the understanding that real-world conversations are a complex blend of collective memories~\cite{bib.bib4 bib.bib18} individual viewpoints~\cite{bib.bib19} external influences~\cite{bib.bib17} and the unique persona of the speakers~\cite{bib.bib46 bib.bib9 bib.bib68 bib.bib49} we create very long-term dialogues based on LLM agent with the following features: (1) a unique persona (§ 3.1[ref_id]S3.SS1); (2) a timeline of causally interlinked events in their lives (§ 3.2[ref_id]S3.SS2); and (3) reflect & response mechanism to respond based on dialogue history (like in ~\cite{bib.bib45}) and image sharing & image reaction behavior which sends or reacts to images (§ 3.3[ref_id]S3.SS3). Finally human annotators fix long-range inconsistencies in dialogues remove irrelevant images and verify the grounding of dialogs to events (§ 3.4[ref_id]S3.SS4). With this pipeline we create LoCoMo a dataset of 50 very long-term dialogues each consisting of 300 turns and 9K tokens on avg. over up to 35 sessions (see Figure 1 and Table 1).
 Conventional approaches for evaluating conversational agents in open-domain dialogues involves directly evaluating the agent response based on past dialogue history. It often employs lexical overlap~\cite{bib.bib44} and semantic overlap~\cite{bib.bib64} between ground truth and the agent response or consistency~\cite{bib.bib15} contradiction~\cite{bib.bib43 bib.bib54} and empathy~\cite{bib.bib60 bib.bib61} of the agent response. However these evaluation metrics are not well-suited for directly assessing the agent’s comprehension of long-term contexts.
 In this study we present a holistic evaluation framework to assess an agent’s proficiency in managing and responding within long-term contexts (see Figure 2). First agents need to “recall” past context correctly to integrate relevant information into future responses. We present a direct examination of their memory via a question answering (QA) task (§ 4.1[ref_id]S4.SS1). We classify questions into five distinct reasoning types to evaluate memory from multiple perspectives: single-hop multi-hop temporal commonsense or world knowledge and adversarial. Second agents also need to recognize long-range causal and temporal connections in the dialogues to generate empathetic and relevant responses. We propose a measurement of their causal and temporal understanding with an event graph summarization task (§ 4.2[ref_id]S4.SS2). In this task the event graphs linked to each LLM speaker serve as the correct answers and models are tasked with extracting this information from the conversation history. Third conversational agents need to utilize relevant context recalled from past conversations to generate responses that are consistent with the ongoing narrative. We assess this ability via the multi-modal dialog generation task (§ 4.3[ref_id]S4.SS3).
 We present extensive experimental results on the LoCoMo benchmark using instruction-based LLMs long-context LLMs and RAG techniques (§ 5[ref_id]S5). Our findings include:
 (1) Long-context LLMs and RAG demonstrate effectiveness in QA tasks improving ‘memory’ capabilities of LLMs (with improvements ranging from 22-66%) but still significantly lag behind human levels (by 56%) especially in temporal reasoning (by 73%);
 (2) long-context LLMs demonstrate significant difficulty with adversarial questions in the QA task showing a performance that is 83% lower than the base model. They are especially prone to misassigning dialogs or events to the wrong speaker. Moreover they show poor performance on event graph summarization lagging behind the base model by 14% indicating that they may grasp the factual elements within the entire conversation but do not accurately comprehend the context; and
 (3) RAG offers a balanced compromise combining the accuracy of short-context LLMs with the extensive comprehension of wide-context LLMs and does particularly well when dialogues are transformed into a database of assertions (observations) about each speaker’s life and persona.
[TABLE START]<table>
	<tr>
		<th>Dataset</th>
		<th>Avg. turns per conv.</th>
		<th>Avg. sessions per conv.</th>
		<th>Avg. tokens per conv.</th>
		<th>Time Interval</th>
		<th>Multimodal</th>
		<th>Collection</th>
	</tr>
	<tr>
		<td>MPChat ~\cite{bib.bib1}</td>
		<td>2.8</td>
		<td>1</td>
		<td>53.3</td>
		<td>-</td>
		<td>✓</td>
		<td>Reddit</td>
	</tr>
	<tr>
		<td>MMDialog ~\cite{bib.bib12}</td>
		<td>4.6</td>
		<td>1</td>
		<td>72.5</td>
		<td>-</td>
		<td>✓</td>
		<td>Social media</td>
	</tr>
	<tr>
		<td>Daily Dialog ~\cite{bib.bib32}</td>
		<td>7.9</td>
		<td>1</td>
		<td>114.7</td>
		<td>-</td>
		<td>✗</td>
		<td>Crowdsourcing</td>
	</tr>
	<tr>
		<td>SODA ~\cite{bib.bib23}</td>
		<td>7.6</td>
		<td>1</td>
		<td>122.4</td>
		<td>-</td>
		<td>✗</td>
		<td>LLM-generated</td>
	</tr>
	<tr>
		<td>MSC ~\cite{bib.bib58} (train; 1-4 sessions)</td>
		<td>53.3</td>
		<td>4</td>
		<td>1,225.9</td>
		<td>few days</td>
		<td>✗</td>
		<td>Crowdsourcing</td>
	</tr>
	<tr>
		<td>Conversation Chronicles ~\cite{bib.bib21}</td>
		<td>58.5</td>
		<td>5</td>
		<td>1,054.7</td>
		<td>few hours - years</td>
		<td>✗</td>
		<td>LLM-generated</td>
	</tr>
	<tr>
		<td>LoCoMo (ours)</td>
		<td>304.9</td>
		<td>19.3</td>
		<td>9,209.2</td>
		<td>few months</td>
		<td>✓</td>
		<td>LLM-gen. + crowdsourc.</td>
	</tr>
</table>
Table 1: Statistics of LoCoMo compared to existing dialog datasets. The average length of a conversation in LoCoMo is 9x that of MSC ~\cite{bib.bib58}, distributed over 6x more turns and 4x more sessions (on average).[TABLE END]

[IMAGE START] [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/f274f6fdc545cd12f7ea667d2a5ba5da.png] Figure 2: Overview of our evaluation framework. We propose three tasks: question answering, event summarization and multimodal dialog generation to evaluate models’ comprehension in very long-term dialogues.[IMAGE END]

[IMAGE START] [IMAGE URL: /mnt/bmcpfs-29000zjpjtl6xjmjiifyk/xkhu/ideation_workspace/papers/reference_figures/b01ca881a135cf8831678f1a64bf8362.png] Figure 1: An example in LoCoMo. Dialogs are steered by the speakers’ personas and corresponding events e.g., Joanna’s responses are consistent with her pet allergies. For Nate, the event got a new dog is followed by a playdate with neighbor’s dog, showcasing long-term memory. Multimodal dialog is enabled with image-sharing and image-response behaviors.[IMAGE END]



## Section References
[bib.bib5] Bertsch et al. (2024) Amanda Bertsch Uri Alon Graham Neubig and Matthew Gormley. 2024. Unlimiformer: Long-range transformers with unlimited length input. Advances in Neural Information Processing Systems 36.
[bib.bib56] Xiao et al. (2023) Guangxuan Xiao Yuandong Tian Beidi Chen Song Han and Mike Lewis. 2023. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453.
[bib.bib51] Shuster et al. (2021) Kurt Shuster Spencer Poff Moya Chen Douwe Kiela and Jason Weston. 2021. Retrieval augmentation reduces hallucination in conversation. In Findings of the Association for Computational Linguistics: EMNLP 2021 pages 3784–3803.
[bib.bib47] Ram et al. (2023) Ori Ram Yoav Levine Itay Dalmedigos Dor Muhlgay Amnon Shashua Kevin Leyton-Brown and Yoav Shoham. 2023. In-context retrieval-augmented language models [https://doi.org/10.1162/tacl_a_00605]. Transactions of the Association for Computational Linguistics 11:1316–1331.
[bib.bib48] Shi et al. (2023) Weijia Shi Sewon Min Michihiro Yasunaga Minjoon Seo Rich James Mike Lewis Luke Zettlemoyer and Wen-tau Yih. 2023. Replug: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652.
[bib.bib58] Xu et al. (2022) Jing Xu Arthur Szlam and Jason Weston. 2022. Beyond goldfish memory: Long-term open-domain conversation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 5180–5197.
[bib.bib21] Jang et al. (2023) Jihyoung Jang Minseong Boo and Hyounghun Kim. 2023. Conversation chronicles: Towards diverse temporal and relational dynamics in multi-session conversations [https://doi.org/10.18653/v1/2023.emnlp-main.838]. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing pages 13584–13606 Singapore. Association for Computational Linguistics.
[bib.bib62] Zhang et al. (2023) Qiang Zhang Jason Naradowsky and Yusuke Miyao. 2023. Mind the gap between conversations for improved long-term dialogue generation [https://doi.org/10.18653/v1/2023.findings-emnlp.720]. In Findings of the Association for Computational Linguistics: EMNLP 2023 pages 10735–10762 Singapore. Association for Computational Linguistics.
[bib.bib4] Assmann and Czaplicka (1995) Jan Assmann and John Czaplicka. 1995. Collective memory and cultural identity. New german critique (65):125–133.
[bib.bib18] Hirst and Manier (2008) William Hirst and David Manier. 2008. Towards a psychology of collective memory. Memory 16(3):183–200.
[bib.bib19] Hirst et al. (2018) William Hirst Jeremy K Yamashiro and Alin Coman. 2018. Collective memory from a psychological perspective. Trends in cognitive sciences 22(5):438–451.
[bib.bib17] Hirst and Echterhoff (2012) William Hirst and Gerald Echterhoff. 2012. Remembering in conversations: The social sharing and reshaping of memories. Annual review of psychology 63:55–79.
[bib.bib46] Pruitt and Grudin (2003) John Pruitt and Jonathan Grudin. 2003. Personas: practice and theory. In Proceedings of the 2003 conference on Designing for user experiences pages 1–15.
[bib.bib9] Cooper (1999) Alan Cooper. 1999. The inmates are running the asylum. Springer.
[bib.bib68] Zhou et al. (2020) Li Zhou Jianfeng Gao Di Li and Heung-Yeung Shum. 2020. The design and implementation of xiaoice an empathetic social chatbot. Computational Linguistics 46(1):53–93.
[bib.bib49] Shum et al. (2020) Michael Shum Stephan Zheng Wojciech Kryscinski Caiming Xiong and Richard Socher. 2020. Sketch-fill-a-R: A persona-grounded chit-chat generation framework [https://doi.org/10.18653/v1/2020.nlp4convai-1.14]. In Proceedings of the 2nd Workshop on Natural Language Processing for Conversational AI pages 118–131 Online. Association for Computational Linguistics.
[bib.bib45] Park et al. (2023) Joon Sung Park Joseph O’Brien Carrie Jun Cai Meredith Ringel Morris Percy Liang and Michael S. Bernstein. 2023. Generative agents: Interactive simulacra of human behavior [https://doi.org/10.1145/3586183.3606763]. In Proceedings of the 36th Annual ACM Symposium on User Interface Software and Technology UIST ’23 New York NY USA. Association for Computing Machinery.
[bib.bib44] Papineni et al. (2002) Kishore Papineni Salim Roukos Todd Ward and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation [https://doi.org/10.3115/1073083.1073135]. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics pages 311–318 Philadelphia Pennsylvania USA. Association for Computational Linguistics.
[bib.bib64] Zhang et al. (2019) Tianyi Zhang Varsha Kishore Felix Wu Kilian Q Weinberger and Yoav Artzi. 2019. Bertscore: Evaluating text generation with bert. In International Conference on Learning Representations.
[bib.bib15] Ghazarian et al. (2022) Sarik Ghazarian Nuan Wen Aram Galstyan and Nanyun Peng. 2022. Deam: Dialogue coherence evaluation using amr-based semantic manipulations. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 771–785.
[bib.bib43] Nie et al. (2021) Yixin Nie Mary Williamson Mohit Bansal Douwe Kiela and Jason Weston. 2021. I like fish especially dolphins: Addressing contradictions in dialogue modeling. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) pages 1699–1713.
[bib.bib54] Welleck et al. (2019) Sean Welleck Jason Weston Arthur Szlam and Kyunghyun Cho. 2019. Dialogue natural language inference [https://doi.org/10.18653/v1/P19-1363]. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics pages 3731–3741 Florence Italy. Association for Computational Linguistics.
[bib.bib60] Zhang et al. (2021a) Chen Zhang Yiming Chen Luis Fernando D’Haro Yan Zhang Thomas Friedrichs Grandee Lee and Haizhou Li. 2021a. Dynaeval: Unifying turn and dialogue level evaluation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers) pages 5676–5689.
[bib.bib61] Zhang et al. (2022) Chen Zhang Luis Fernando D’Haro Qiquan Zhang Thomas Friedrichs and Haizhou Li. 2022. Fined-eval: Fine-grained automatic dialogue-level evaluation. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing pages 3336–3355.
[bib.bib1] Ahn et al. (2023) Jaewoo Ahn Yeda Song Sangdoo Yun and Gunhee Kim. 2023. MPCHAT: Towards multimodal persona-grounded conversation [https://doi.org/10.18653/v1/2023.acl-long.189]. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 3354–3377 Toronto Canada. Association for Computational Linguistics.
[bib.bib12] Feng et al. (2023) Jiazhan Feng Qingfeng Sun Can Xu Pu Zhao Yaming Yang Chongyang Tao Dongyan Zhao and Qingwei Lin. 2023. MMDialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation [https://doi.org/10.18653/v1/2023.acl-long.405]. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 7348–7363 Toronto Canada. Association for Computational Linguistics.
[bib.bib32] Li et al. (2017) Yanran Li Hui Su Xiaoyu Shen Wenjie Li Ziqiang Cao and Shuzi Niu. 2017. Dailydialog: A manually labelled multi-turn dialogue dataset. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 1: Long Papers) pages 986–995.
[bib.bib23] Kim et al. (2023) Hyunwoo Kim Jack Hessel Liwei Jiang Peter West Ximing Lu Youngjae Yu Pei Zhou Ronan Bras Malihe Alikhani Gunhee Kim Maarten Sap and Yejin Choi. 2023. SODA: Million-scale dialogue distillation with social commonsense contextualization [https://doi.org/10.18653/v1/2023.emnlp-main.799]. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing pages 12930–12949 Singapore. Association for Computational Linguistics.