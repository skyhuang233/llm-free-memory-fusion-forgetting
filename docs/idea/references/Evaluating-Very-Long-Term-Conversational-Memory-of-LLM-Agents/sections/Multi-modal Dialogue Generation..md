Multi-modal Dialogue Generation.
Multi-modal Dialogue Generation.
 We generate 50 conversations using our automated pipeline (without human filtering; § 3[ref_id]S3) for training data and train three versions of MiniGPT-5~\cite{bib.bib65}: (1) Base trains on prior dialogue turns only; (2) + summary trains on prior dialogue turns and a global summary of the ongoing conversation; (3) + observation trains on prior dialogue turns and observations retrieved from conversation history. Each run is initialized with a MiniGPT-5 checkpoint finetuned on MMDialog ~\cite{bib.bib12}.


## Section References
[bib.bib65] Zheng et al. (2023) Kaizhi Zheng Xuehai He and Xin Eric Wang. 2023. Minigpt-5: Interleaved vision-and-language generation via generative vokens. arXiv preprint arXiv:2310.02239.
[bib.bib12] Feng et al. (2023) Jiazhan Feng Qingfeng Sun Can Xu Pu Zhao Yaming Yang Chongyang Tao Dongyan Zhao and Qingwei Lin. 2023. MMDialog: A large-scale multi-turn dialogue dataset towards multi-modal open-domain conversation [https://doi.org/10.18653/v1/2023.acl-long.405]. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) pages 7348–7363 Toronto Canada. Association for Computational Linguistics.