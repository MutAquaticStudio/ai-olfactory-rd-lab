"""Optional retrosynthesis evidence, kept separate from Ertl SAscore."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional, Protocol


@dataclass(frozen=True)
class SynthesisAssessment:
    status: str
    method: str = "AiZynthFinder"
    time_limit_seconds: int = 300
    route_found: Optional[bool] = None
    route_steps: Optional[int] = None
    search_time_seconds: Optional[float] = None
    precursor_coverage: Optional[float] = None
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["warnings"] = list(self.warnings)
        return payload


class RetrosynthesisService(Protocol):
    def search(self, isomeric_smiles: str) -> SynthesisAssessment:
        ...


class UnavailableRetrosynthesisService:
    def search(self, isomeric_smiles: str) -> SynthesisAssessment:
        return SynthesisAssessment(
            status="NOT_CONFIGURED",
            warnings=("Route search is not configured in this runtime.",),
        )


def _default_runner(
    isomeric_smiles: str,
    config_path: Path,
    time_limit_seconds: int,
) -> Mapping[str, object]:
    """Use the public AiZynthFinder Python API when the optional package exists."""
    from aizynthfinder.aizynthfinder import AiZynthFinder  # type: ignore[import-not-found]

    finder = AiZynthFinder(configfile=str(config_path))
    finder.config.search.time_limit = int(time_limit_seconds)
    finder.target_smiles = isomeric_smiles
    finder.prepare_tree()
    finder.tree_search(show_progress=False)
    finder.build_routes()
    statistics = finder.extract_statistics() or {}
    route_count = len(finder.routes)
    step_keys = (
        "number of reactions in first solved route",
        "number_of_reactions_in_first_solved_route",
        "route_steps",
    )
    route_steps = next(
        (statistics[key] for key in step_keys if key in statistics),
        None,
    )
    return {
        "route_found": route_count > 0,
        "route_steps": int(route_steps) if route_steps is not None else None,
        "precursor_coverage": statistics.get("stock_coverage"),
    }


class AiZynthFinderService:
    def __init__(
        self,
        config_path: Path,
        *,
        time_limit_seconds: int = 300,
        runner: Callable[[str, Path, int], Mapping[str, object]] = _default_runner,
    ) -> None:
        self.config_path = Path(config_path).expanduser().resolve()
        self.time_limit_seconds = int(time_limit_seconds)
        self.runner = runner
        if self.time_limit_seconds <= 0 or self.time_limit_seconds > 300:
            raise ValueError("Retrosynthesis time limit must be between 1 and 300 seconds")

    def search(self, isomeric_smiles: str) -> SynthesisAssessment:
        if not self.config_path.is_file():
            return SynthesisAssessment(
                status="CONFIGURATION_ERROR",
                time_limit_seconds=self.time_limit_seconds,
                warnings=("AiZynthFinder configuration file is unavailable.",),
            )
        started = time.monotonic()
        try:
            result = self.runner(
                isomeric_smiles,
                self.config_path,
                self.time_limit_seconds,
            )
        except ImportError:
            return SynthesisAssessment(
                status="DEPENDENCY_UNAVAILABLE",
                time_limit_seconds=self.time_limit_seconds,
                search_time_seconds=time.monotonic() - started,
                warnings=("AiZynthFinder is not installed in this runtime.",),
            )
        except Exception as error:
            return SynthesisAssessment(
                status="SEARCH_ERROR",
                time_limit_seconds=self.time_limit_seconds,
                search_time_seconds=time.monotonic() - started,
                warnings=(type(error).__name__,),
            )
        elapsed = time.monotonic() - started
        route_found = bool(result.get("route_found", False))
        return SynthesisAssessment(
            status="ROUTE_FOUND" if route_found else "NO_ROUTE_FOUND",
            time_limit_seconds=self.time_limit_seconds,
            route_found=route_found,
            route_steps=(
                int(result["route_steps"])
                if result.get("route_steps") is not None
                else None
            ),
            search_time_seconds=elapsed,
            precursor_coverage=(
                float(result["precursor_coverage"])
                if result.get("precursor_coverage") is not None
                else None
            ),
            warnings=(
                "A computed route is evidence for review, not a synthesis guarantee.",
            ),
        )


def build_retrosynthesis_service(
    config_path: Optional[str],
) -> RetrosynthesisService:
    if not config_path:
        return UnavailableRetrosynthesisService()
    return AiZynthFinderService(Path(config_path))


__all__ = [
    "SynthesisAssessment",
    "RetrosynthesisService",
    "UnavailableRetrosynthesisService",
    "AiZynthFinderService",
    "build_retrosynthesis_service",
]
