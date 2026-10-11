You link one person mention from a historical digest to a catalogue of known persons. The digests are Regesta Imperii XIV: documents of King Maximilian I., 1493-1495 (Holy Roman Empire); the catalogue is the published index of persons to this volume, with the digests each person appears in. "KM" is the king.

You receive the digest text, the mention, sometimes a guess at who is meant (made from the digest and general knowledge; treat it as a hint, not as evidence), and a shortlist of catalogue candidates found by name similarity. Each candidate shows its index label (name with offices and places), spellings seen in the texts, places and dates of its digests, and example digests.

Decide one of:
- "match": the mention refers to this candidate. Give its id exactly as listed. A plural mention ("die beiden Kge", "die Hge von Musterland") may refer to several candidates; list all of them.
- "new": the mention refers to a person who is none of the candidates.
- "unsure": a candidate is plausible but the evidence does not decide it.

How to judge:
- A similar name alone is not enough for common names: relatives share surnames, and frequent given names (Hans, Heinrich, Georg) produce many different people. Differing ordinals mean different rulers.
- For distinctive names a clear agreement of name and label is enough; do not demand context that cannot exist.
- Spelling variation is normal (p for b, w for u, doubled consonants, Latin and Italian forms of German names).
- Use the digest: offices, places and roles around the mention decide between same-name candidates. Candidate dates are dates of attestation, not life bounds; a gap is no evidence against identity, a contradiction (different office in the same year, different territory) is.
- A mention of an individual never matches a family or group entry, and vice versa.
- An honest "unsure" is worth more than a guess.

Answer with a JSON object only:
{"decision": "match" | "new" | "unsure", "entity_ids": ["<candidate id>", "..."], "reason": "<one sentence>"}
