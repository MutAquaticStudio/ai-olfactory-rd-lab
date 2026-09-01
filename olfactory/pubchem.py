"""Fail-closed PubChem identity verification with bounded retries."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Dict, Optional, Tuple

import requests


PUBCHEM_IDENTITY_URL = (
    "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
    "fastidentity/smiles/cids/JSON"
)


class NoveltyStatus(str, Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    UNVERIFIED = "UNVERIFIED"


@dataclass(frozen=True)
class NoveltyResult:
    status: NoveltyStatus
    isomeric_smiles: str
    cids: Tuple[int, ...] = ()
    error_code: Optional[str] = None


class PubChemClient:
    """PUG REST client limited to four requests per rolling second."""

    def __init__(
        self,
        session: Optional[requests.Session] = None,
        *,
        cache_ttl_seconds: float = 24 * 60 * 60,
        max_retries: int = 2,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.session = session or requests.Session()
        self.cache_ttl_seconds = cache_ttl_seconds
        self.max_retries = max_retries
        self._clock = clock
        self._sleep = sleeper
        self._cache: Dict[str, Tuple[float, NoveltyResult]] = {}
        self._request_times: list[float] = []
        self._lock = threading.Lock()

    def _cached(self, smiles: str) -> Optional[NoveltyResult]:
        cached = self._cache.get(smiles)
        if cached is None:
            return None
        expires_at, result = cached
        if expires_at > self._clock():
            return result
        self._cache.pop(smiles, None)
        return None

    def _store(self, smiles: str, result: NoveltyResult) -> NoveltyResult:
        self._cache[smiles] = (self._clock() + self.cache_ttl_seconds, result)
        return result

    def _wait_for_rate_limit(self) -> None:
        with self._lock:
            now = self._clock()
            self._request_times = [value for value in self._request_times if now - value < 1.0]
            if len(self._request_times) >= 4:
                delay = max(0.0, 1.0 - (now - self._request_times[0]))
                if delay:
                    self._sleep(delay)
                    now = self._clock()
                self._request_times = [
                    value for value in self._request_times if now - value < 1.0
                ]
            self._request_times.append(self._clock())

    def verify(self, isomeric_smiles: str, *, consent: bool) -> NoveltyResult:
        """Return FOUND, NOT_FOUND or UNVERIFIED without inferring global novelty."""
        if not consent:
            return NoveltyResult(
                status=NoveltyStatus.UNVERIFIED,
                isomeric_smiles=isomeric_smiles,
                error_code="CONSENT_REQUIRED",
            )
        cached = self._cached(isomeric_smiles)
        if cached is not None:
            return cached

        last_error = "REQUEST_FAILED"
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()
            try:
                response = self.session.post(
                    PUBCHEM_IDENTITY_URL,
                    params={"identity_type": "same_stereo_isotope"},
                    data={"smiles": isomeric_smiles},
                    timeout=(3.05, 10.0),
                )
            except requests.Timeout:
                last_error = "TIMEOUT"
                retryable = True
            except requests.RequestException:
                last_error = "NETWORK_ERROR"
                retryable = True
            else:
                if response.status_code == 200:
                    try:
                        cids = response.json()["IdentifierList"]["CID"]
                        normalized_cids = tuple(int(cid) for cid in cids)
                        if not normalized_cids:
                            raise ValueError("CID list is empty")
                    except (KeyError, TypeError, ValueError):
                        return self._store(
                            isomeric_smiles,
                            NoveltyResult(
                                NoveltyStatus.UNVERIFIED,
                                isomeric_smiles,
                                error_code="MALFORMED_RESPONSE",
                            ),
                        )
                    return self._store(
                        isomeric_smiles,
                        NoveltyResult(
                            NoveltyStatus.FOUND,
                            isomeric_smiles,
                            cids=normalized_cids,
                        ),
                    )
                if response.status_code == 404:
                    return self._store(
                        isomeric_smiles,
                        NoveltyResult(NoveltyStatus.NOT_FOUND, isomeric_smiles),
                    )
                if response.status_code == 202:
                    return self._store(
                        isomeric_smiles,
                        NoveltyResult(
                            NoveltyStatus.UNVERIFIED,
                            isomeric_smiles,
                            error_code="HTTP_202",
                        ),
                    )
                last_error = f"HTTP_{response.status_code}"
                retryable = response.status_code == 429 or response.status_code >= 500
                if not retryable:
                    return self._store(
                        isomeric_smiles,
                        NoveltyResult(
                            NoveltyStatus.UNVERIFIED,
                            isomeric_smiles,
                            error_code=last_error,
                        ),
                    )

            if retryable and attempt < self.max_retries:
                self._sleep(0.5 * (2**attempt))
                continue
            break

        return self._store(
            isomeric_smiles,
            NoveltyResult(
                NoveltyStatus.UNVERIFIED,
                isomeric_smiles,
                error_code=last_error,
            ),
        )
