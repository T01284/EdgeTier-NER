from __future__ import annotations

import logging
import random
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from edgefs.data.schema import NERSentence


def entity_types_in_sentence(sent: NERSentence) -> set[str]:
    types: set[str] = set()
    for tag in sent.tags:
        if tag == "O" or "-" not in tag:
            continue
        types.add(tag.split("-", 1)[1])
    return types


@dataclass
class FewShotEpisode:
    support: list[NERSentence]
    query: list[NERSentence]
    entity_types: list[str]
    seed: int


class FewShotEpisodeSampler:
    """Sample N-way K-shot episodes at entity-type level."""

    def __init__(
        self,
        pool: list[NERSentence],
        n_way: int = 5,
        k_shot: int = 1,
        query_size: int = 100,
        seed: int = 42,
    ) -> None:
        self.pool = pool
        self.n_way = n_way
        self.k_shot = k_shot
        self.query_size = query_size
        self.rng = random.Random(seed)

        self.type2sents: dict[str, list[NERSentence]] = {}
        for sent in pool:
            for et in entity_types_in_sentence(sent):
                self.type2sents.setdefault(et, []).append(sent)

    def sample(self, seed: int | None = None) -> FewShotEpisode:
        rng = random.Random(seed if seed is not None else self.rng.randint(0, 10**9))
        available = [t for t, sents in self.type2sents.items() if len(sents) >= self.k_shot]
        if not available:
            raise ValueError(f"No entity types with at least {self.k_shot} support sentences")
        n_way = min(self.n_way, len(available))
        if n_way < self.n_way:
            logger.warning(
                "Requested %d-way but only %d entity types available; using %d-way",
                self.n_way,
                len(available),
                n_way,
            )
        entity_types = rng.sample(available, n_way)

        support: list[NERSentence] = []
        used_ids: set[int] = set()
        for et in entity_types:
            candidates = self.type2sents[et]
            picked = rng.sample(candidates, min(self.k_shot, len(candidates)))
            for sent in picked:
                sid = id(sent)
                if sid not in used_ids:
                    support.append(sent)
                    used_ids.add(sid)

        query_candidates = [s for s in self.pool if id(s) not in used_ids]
        rng.shuffle(query_candidates)
        query = query_candidates[: self.query_size]
        return FewShotEpisode(
            support=support,
            query=query,
            entity_types=entity_types,
            seed=seed or 0,
        )
