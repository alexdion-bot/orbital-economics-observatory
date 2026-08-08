# Use of AI assistance

The processing pipeline and the web interface in this repository were
developed with the assistance of a large language model (Claude, Anthropic).

The research question, the theoretical framing, the choice of data sources,
the parameterisation, the validation strategy and the interpretation of the
results are the author's own. Every output was checked against the published
literature and against independent external quantities, as set out in the
Method section of the website.

Four substantive errors identified during development came from reading the
source documentation and from debugging against physical constraints rather
than from code generation. They are documented in the Method section and in
the comments of the relevant scripts.

- The GCAT `Active` field encodes `A` for an operational payload and `P` for
  a derelict one. It does not encode payload type.
- `Control` in the payload catalogue designates the ground control centre,
  not the operator. Country of operator has to come from `SatState` in the
  launch log.
- For debris fragments, `SDate` is the separation date and `LDate` is the
  parent launch date. Conflating them made historical debris stocks appear
  higher in 2000 than in 2026, which is physically impossible.
- The published OPUS cost function mixes km/s and m/s. This implementation
  follows Table 2 of the paper, metres per second throughout.

This statement follows the disclosure practice expected for AI assisted
research work.
