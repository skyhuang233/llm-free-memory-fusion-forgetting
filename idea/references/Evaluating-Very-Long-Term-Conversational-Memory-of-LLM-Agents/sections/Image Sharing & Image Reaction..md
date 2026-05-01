Image Sharing & Image Reaction.
Image Sharing & Image Reaction.
 The image sharing & image reaction functions are integrated to add a multi-modal dimension to the long-term dialogues.2Image captions are also saved to long-term memory. The image sharing function is called when the agent decides to send an image. This process includes: (1) Generate a caption c for the intended image using \{M}; (2) Convert the caption c into relevant keywords w using \{M}; (3) Use the keywords k to find an image through web search WEB(k)3https://pypi.org/project/icrawler/; (4) Share the chosen image. Conversely the image reaction function is triggered upon receiving an image from another agent and entails: (1) Generate caption c for the received image4We use BLIP-2 ~\cite{bib.bib31} as the captioning model.; (2) Generate a reaction for the received image in response using \{M} (See Appendix A.2.1[ref_id]A1.SS2.SSS1).


## Section References
[bib.bib31] Li et al. (2023b) Junnan Li Dongxu Li Silvio Savarese and Steven Hoi. 2023b. Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models. In International Conference on Machine Learning.