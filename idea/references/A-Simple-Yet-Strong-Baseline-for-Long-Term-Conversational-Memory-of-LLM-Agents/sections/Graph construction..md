Graph construction.
We construct a heterogeneous graph G=(VE) with three node types:
 •
 Session nodes v_{s} for s\in\{S}.
 •
 EDU nodes v_{e} for e\in\{E}.
 •
 Argument nodes v_{a} for a\in\{A}.
 Edges are defined as:
 \displaystyle E_{\text{sess-edu}} \displaystyle E_{\text{edu-arg}} \displaystyle E_{\text{syn}}
 Here \text{sim}(aa^{\prime}) is cosine similarity between h_{\text{arg}}(a) and h_{\text{arg}}(a^{\prime}) and we cap the number of synonym neighbors per a (e.g. at 100). The final node set is V=\{v_{s}\}\cup\{v_{e}\}\cup\{v_{a}\} and edge set E=E_{\text{sess-edu}}\cup E_{\text{edu-arg}}\cup E_{\text{syn}}.
 For later retrieval we cache embeddings for EDU texts h_{\text{edu}}(e)=h(\text{text}(e)). Graph construction is performed offline as new sessions arrive.