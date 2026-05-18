#!/usr/bin/env python
"""
Genetic Algorithm optimiser for SDXL inpainting hyperparameters.

Implements a classic GA loop with:
  - population initialisation (random or seeded from defaults)
  - tournament selection
  - uniform crossover
  - per-parameter Gaussian mutation
  - elitism (top N individuals preserved verbatim)

Supports three evaluation modes:
  1. JSON dataset mode (--fitness-json as a list of genome+metric records):
     Each record contains genome fields (mask_cx, mask_cy, mask_radius,
     strength, guidance_scale, seed) AND metric fields (seam_quality,
     non_roi_drift, histogram_similarity, adjacent_ssim,
     z_gradient_smoothness).  A query genome is scored by nearest-neighbor
     (kNN) search in normalised genome space against the dataset, providing a
     meaningful optimisation signal.

  2. JSON fallback modes (backward-compatible, no signal):
     a) Flat metrics dict → constant fitness for every genome (warning).
     b) List of metric-only dicts → averaged fitness (warning).

  3. Mock mode (--mock): synthetic metrics correlated to genome parameters
     (for smoke testing without real data).

Outputs (to --output-dir):
  - best_genome.json     – best genome found
  - ga_history.csv       – per-generation stats (best, mean, worst, diversity)
  - population_final.json – final population with fitness values
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path for internal imports
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.ga.genome_encoding import (
    NUM_PARAMS,
    PARAM_DEFS,
    PARAM_NAMES,
    default_genome,
    random_genome,
    genome_from_normalised_vector,
    normalised_vector_from_genome,
    mutate_genome,
    crossover_genomes,
    check_genome,
    genome_to_str,
)
from scripts.ga.run_inpaint_fitness import (
    METRIC_SPECS,
    compute_fitness,
    mock_metrics,
)

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Individual:
    """A single genome with its fitness score."""
    genome: Dict[str, float]
    fitness: float = -float("inf")
    component_scores: Dict[str, float] = field(default_factory=dict)
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": self.genome,
            "fitness": self.fitness,
            "component_scores": self.component_scores,
            "generation": self.generation,
        }


@dataclass
class GenerationStats:
    """Summary statistics for one generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    worst_fitness: float
    median_fitness: float
    std_fitness: float
    diversity: float  # mean pair-wise distance in normalised space
    best_genome: Dict[str, float]
    elapsed_s: float


# ---------------------------------------------------------------------------
# Evaluation backend
# ---------------------------------------------------------------------------

class JsonMetricsEvaluator:
    """Evaluates individuals from a JSON file with three supported formats.

    **Format 1 – Genome dataset (recommended)** \n
    A list of records, each containing all 6 genome fields AND metric fields::

        [
          {"mask_cx": 0.5, "mask_cy": 0.5, "mask_radius": 0.15,
           "strength": 0.55, "guidance_scale": 7.0, "seed": 42,
           "seam_quality": 1.2, "non_roi_drift": 0.8,
           "histogram_similarity": 0.92, "adjacent_ssim": 0.87,
           "z_gradient_smoothness": 0.31},
          ...
        ]

    A query genome is scored by *k*-nearest-neighbor (k=3) in normalised
    genome space; the matched record's metrics are fed to
    :func:`compute_fitness`.  The population will receive varying fitness
    scores, providing an optimisation signal.

    **Format 2 – Flat metrics dict (legacy)** \n
    A single dict of metric name → value.  Every genome receives the same
    fitness (no signal – a warning is printed once).

    **Format 3 – List of metric-only dicts (legacy)** \n
    A list of dicts without genome fields.  The mean fitness across entries
    is used for every genome (no signal – a warning is printed once).

    If *mock* is True, synthetic metrics are generated based on the genome
    (for smoke testing without real data).
    """

    # How many genome fields must be present to detect the "dataset" format
    _MIN_FIELDS_FOR_DATASET = 6  # all 6 params in PARAM_NAMES
    # Default k for kNN averaging
    _KNN_DEFAULT = 3

    def __init__(self, json_path: Optional[Path] = None, mock: bool = False, seed: int = 42):
        self.json_path = json_path
        self.mock = mock
        self._rng = np.random.default_rng(seed)

        # State variables
        self._precomputed_fitness: Optional[Tuple[float, Dict[str, float]]] = None
        self._dataset: Optional[List[Dict[str, Any]]] = None
        self._dataset_vectors: Optional[np.ndarray] = None
        self._dataset_mode: Optional[str] = None  # 'constant' | 'average' | 'genome_nn'
        self._warned_legacy: bool = False

        if not mock and json_path is not None:
            self._load_and_detect(json_path)

    # ------------------------------------------------------------------
    # JSON loading & format detection
    # ------------------------------------------------------------------

    @staticmethod
    def _record_has_all_genome_fields(record: Dict[str, Any]) -> bool:
        """Return True when *record* contains all 6 genome parameter keys."""
        return all(name in record for name in PARAM_NAMES)

    def _load_and_detect(self, json_path: Path) -> None:
        """Load JSON and classify the format."""
        with open(json_path, "r") as f:
            data = json.load(f)

        if isinstance(data, dict):
            # Format 2: flat metrics dict → constant fitness
            fitness, comps, _ = compute_fitness(data, verbose=False)
            self._precomputed_fitness = (fitness, comps)
            self._dataset_mode = "constant"
            return

        if not isinstance(data, list):
            raise ValueError(
                f"Unexpected JSON root type in {json_path}: {type(data).__name__}. "
                "Expected dict or list."
            )

        if not data:
            raise ValueError(f"Empty JSON list in {json_path}")

        # Detect dataset format from all records (safer than first-record only).
        # Mixed formats are rejected to avoid silently dropping genome-dependent
        # signal when the first entry is metric-only.
        has_genome_flags = [
            isinstance(entry, dict) and self._record_has_all_genome_fields(entry)
            for entry in data
        ]
        if any(has_genome_flags):
            if not all(has_genome_flags):
                raise ValueError(
                    f"Mixed list format in {json_path}: some records contain all genome fields "
                    "while others do not. Provide either all genome+metric records "
                    "or metric-only dicts."
                )
            self._dataset = data
            self._precompute_genome_vectors()
            self._dataset_mode = "genome_nn"
            return

        # Format 3: list of metric-only dicts → averaged fitness
        fitnesses: List[float] = []
        comp_list: List[Dict[str, float]] = []
        for entry in data:
            f, comps, _ = compute_fitness(entry, verbose=False)
            fitnesses.append(f)
            comp_list.append(comps)
        avg_fitness = float(np.mean(fitnesses)) if fitnesses else 0.0
        self._precomputed_fitness = (avg_fitness, comp_list[-1] if comp_list else {})
        self._dataset_mode = "average"

    def _precompute_genome_vectors(self) -> None:
        """Build normalised genome vector array from the loaded dataset."""
        assert self._dataset is not None
        vectors = []
        for rec in self._dataset:
            genome = {name: float(rec[name]) for name in PARAM_NAMES}
            vectors.append(normalised_vector_from_genome(genome))
        self._dataset_vectors = np.array(vectors, dtype=np.float64)

    # ------------------------------------------------------------------
    # Evaluation entry point
    # ------------------------------------------------------------------

    def __call__(self, genome: Dict[str, float], verbose: bool = False) -> Tuple[float, Dict[str, float]]:
        if self.mock:
            metrics = mock_metrics(genome, rng=self._rng)
            fitness, comps, _ = compute_fitness(metrics, verbose=verbose)
            return fitness, comps

        if self._dataset_mode == "genome_nn":
            return self._score_by_knn(genome, verbose=verbose)

        if self._dataset_mode in ("constant", "average"):
            self._maybe_warn_legacy()
            fitness, comps = self._precomputed_fitness  # type: ignore[misc]
            return float(fitness), dict(comps)

        raise ValueError("No evaluation source (provide --fitness-json or use --mock)")

    def _maybe_warn_legacy(self) -> None:
        """Emit a one-time warning for legacy (no-signal) JSON formats."""
        if self._warned_legacy:
            return
        self._warned_legacy = True
        mode = self._dataset_mode
        if mode == "constant":
            msg = (
                "WARNING: fitness JSON is a flat metrics dict — every genome "
                "receives the same fitness (no optimisation signal). "
                "Provide a list of genome+metric records for genome-dependent scoring."
            )
        elif mode == "average":
            msg = (
                "WARNING: fitness JSON is a list of metric-only dicts — every genome "
                "receives the same averaged fitness (no optimisation signal). "
                "Include all 6 genome fields (mask_cx, mask_cy, mask_radius, "
                "strength, guidance_scale, seed) in each record for genome-dependent scoring."
            )
        else:
            return
        print(msg, file=sys.stderr)

    # ------------------------------------------------------------------
    # kNN-based scoring
    # ------------------------------------------------------------------

    def _score_by_knn(
        self,
        genome: Dict[str, float],
        verbose: bool = False,
        k: int = _KNN_DEFAULT,
    ) -> Tuple[float, Dict[str, float]]:
        """Score *genome* by kNN average in normalised genome space."""
        assert self._dataset is not None
        assert self._dataset_vectors is not None

        qvec = normalised_vector_from_genome(genome)
        dists = np.linalg.norm(self._dataset_vectors - qvec, axis=1)  # (N,)

        k_actual = min(k, len(dists))
        nearest_idx = np.argsort(dists)[:k_actual]

        if verbose:
            print(
                f"  [kNN] query dists min={dists[nearest_idx[0]]:.4f} "
                f"max={dists[nearest_idx[-1]]:.4f}  k={k_actual}"
            )

        # Average all non-genome metric fields across nearest neighbours
        averaged: Dict[str, float] = {}
        for name, _, _, _ in METRIC_SPECS:
            vals = []
            for idx in nearest_idx:
                v = self._dataset[idx].get(name)
                if v is not None:
                    try:
                        vals.append(float(v))
                    except (TypeError, ValueError):
                        pass
            if vals:
                averaged[name] = float(np.mean(vals))

        fitness, comps, _ = compute_fitness(averaged, verbose=verbose)
        return fitness, comps


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------

class GAOptimiser:
    """Genetic algorithm for inpainting hyperparameter optimisation."""

    def __init__(
        self,
        evaluator: Callable[[Dict[str, float]], Tuple[float, Dict[str, float]]],
        population_size: int = 20,
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        seed: int = 42,
    ):
        if population_size < 1:
            raise ValueError(f"population_size must be >= 1, got {population_size}")
        if tournament_size < 1:
            raise ValueError(f"tournament_size must be >= 1, got {tournament_size}")
        if not (0.0 <= elite_fraction <= 1.0):
            raise ValueError(f"elite_fraction must be in [0,1], got {elite_fraction}")

        self.evaluator = evaluator
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strength
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.tournament_size = min(tournament_size, population_size)
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.population: List[Individual] = []
        self.history: List[GenerationStats] = []
        self.best_individual: Optional[Individual] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_population(self, seed_genomes: Optional[List[Dict[str, float]]] = None) -> None:
        """Initialise population.

        If *seed_genomes* is provided, use them first, then fill the
        remainder with random individuals.
        """
        self.population = []
        used = 0
        if seed_genomes:
            for g in seed_genomes:
                ind = Individual(genome=g.copy())
                self.population.append(ind)
                used += 1
                if len(self.population) >= self.population_size:
                    break

        while len(self.population) < self.population_size:
            g = random_genome(self.rng)
            ind = Individual(genome=g)
            self.population.append(ind)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_population(self, generation: int, verbose: bool = False) -> None:
        """Evaluate all un-evaluated individuals in the population."""
        for ind in self.population:
            if ind.fitness > -float("inf") and ind.generation == generation:
                continue
            ind.fitness, ind.component_scores = self.evaluator(ind.genome, verbose=verbose)
            ind.generation = generation

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def tournament_select(self) -> Individual:
        """Select one individual via tournament selection."""
        best = self.rng.choice(self.population)
        for _ in range(1, self.tournament_size):
            contender = self.rng.choice(self.population)
            if contender.fitness > best.fitness:
                best = contender
        return best

    # ------------------------------------------------------------------
    # Evolution step
    # ------------------------------------------------------------------

    def evolve(self, generation: int) -> None:
        """Create next generation via selection, crossover, and mutation."""
        # Sort by fitness for elitism
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        next_pop: List[Individual] = []

        # Elitism
        for i in range(self.elite_count):
            elite = Individual(
                genome=self.population[i].genome.copy(),
                fitness=self.population[i].fitness,
                component_scores=self.population[i].component_scores.copy(),
                generation=generation,
            )
            next_pop.append(elite)

        # Fill rest via crossover + mutation
        while len(next_pop) < self.population_size:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()

            child1_genome, child2_genome = crossover_genomes(
                parent_a.genome, parent_b.genome, self.rng
            )

            child1_genome = mutate_genome(
                child1_genome, self.mutation_rate, self.mutation_strength, self.rng
            )
            child2_genome = mutate_genome(
                child2_genome, self.mutation_rate, self.mutation_strength, self.rng
            )

            child1 = Individual(genome=child1_genome, generation=generation)
            child2 = Individual(genome=child2_genome, generation=generation)

            next_pop.append(child1)
            if len(next_pop) < self.population_size:
                next_pop.append(child2)

        self.population = next_pop[:self.population_size]

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def compute_diversity(self) -> float:
        """Mean Euclidean distance between all pairs in normalised space."""
        if len(self.population) < 2:
            return 0.0
        vectors = [normalised_vector_from_genome(ind.genome) for ind in self.population]
        distances = []
        for i in range(len(vectors)):
            for j in range(i + 1, len(vectors)):
                d = float(np.linalg.norm(vectors[i] - vectors[j]))
                distances.append(d)
        return float(np.mean(distances)) if distances else 0.0

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(
        self,
        generations: int,
        verbose: bool = False,
        seed_genomes: Optional[List[Dict[str, float]]] = None,
    ) -> GenerationStats:
        """Run the GA for *generations* steps.

        Returns the final GenerationStats.
        """
        if generations < 1:
            raise ValueError(f"generations must be >= 1, got {generations}")

        t_start = time.perf_counter()

        self.init_population(seed_genomes)

        for gen in range(generations):
            t_gen = time.perf_counter()

            self.evaluate_population(gen, verbose=verbose)

            # Stats
            fitnesses = [ind.fitness for ind in self.population]
            best_idx = int(np.argmax(fitnesses))
            best_fit = float(fitnesses[best_idx])
            mean_fit = float(np.mean(fitnesses))
            worst_fit = float(np.min(fitnesses))
            median_fit = float(np.median(fitnesses))
            std_fit = float(np.std(fitnesses))
            diversity = self.compute_diversity()
            best_gen = self.population[best_idx].genome.copy()

            elapsed = time.perf_counter() - t_gen

            stats = GenerationStats(
                generation=gen,
                best_fitness=best_fit,
                mean_fitness=mean_fit,
                worst_fitness=worst_fit,
                median_fitness=median_fit,
                std_fitness=std_fit,
                diversity=diversity,
                best_genome=best_gen,
                elapsed_s=elapsed,
            )
            self.history.append(stats)

            if self.best_individual is None or best_fit > self.best_individual.fitness:
                self.best_individual = Individual(
                    genome=best_gen.copy(),
                    fitness=best_fit,
                    component_scores=self.population[best_idx].component_scores.copy(),
                    generation=gen,
                )

            if verbose:
                print(
                    f"Gen {gen:3d}: best={best_fit:.4f}  mean={mean_fit:.4f}  "
                    f"worst={worst_fit:.4f}  diversity={diversity:.4f}  "
                    f"[{genome_to_str(best_gen)}]"
                )

            # Evolve (except for last generation)
            if gen < generations - 1:
                self.evolve(gen + 1)

        total_elapsed = time.perf_counter() - t_start
        if verbose:
            print(f"\nGA finished in {total_elapsed:.2f}s")
            if self.best_individual:
                print(f"Best genome: {genome_to_str(self.best_individual.genome)}")
                print(f"Best fitness: {self.best_individual.fitness:.4f}")

        return stats

    # ------------------------------------------------------------------
    # Output writers
    # ------------------------------------------------------------------

    def save_best_genome(self, path: Path) -> None:
        """Write best_genome.json."""
        if self.best_individual is None:
            raise RuntimeError("No best individual — run the GA first.")
        data = {
            "genome": self.best_individual.genome,
            "fitness": self.best_individual.fitness,
            "component_scores": self.best_individual.component_scores,
            "generation": self.best_individual.generation,
            "run_seed": self.seed,
            "population_size": self.population_size,
            "mutation_rate": self.mutation_rate,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def save_history(self, path: Path) -> None:
        """Write ga_history.csv."""
        if not self.history:
            raise RuntimeError("No history — run the GA first.")
        fieldnames = [
            "generation", "best_fitness", "mean_fitness", "worst_fitness",
            "median_fitness", "std_fitness", "diversity", "elapsed_s",
            "best_mask_cx", "best_mask_cy", "best_mask_radius",
            "best_strength", "best_guidance_scale", "best_seed",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in self.history:
                g = h.best_genome
                writer.writerow({
                    "generation": h.generation,
                    "best_fitness": f"{h.best_fitness:.6f}",
                    "mean_fitness": f"{h.mean_fitness:.6f}",
                    "worst_fitness": f"{h.worst_fitness:.6f}",
                    "median_fitness": f"{h.median_fitness:.6f}",
                    "std_fitness": f"{h.std_fitness:.6f}",
                    "diversity": f"{h.diversity:.6f}",
                    "elapsed_s": f"{h.elapsed_s:.4f}",
                    "best_mask_cx": g.get("mask_cx", ""),
                    "best_mask_cy": g.get("mask_cy", ""),
                    "best_mask_radius": g.get("mask_radius", ""),
                    "best_strength": g.get("strength", ""),
                    "best_guidance_scale": g.get("guidance_scale", ""),
                    "best_seed": int(g.get("seed", 0)),
                })

    def save_population(self, path: Path) -> None:
        """Write population_final.json."""
        data = {
            "seed": self.seed,
            "population_size": self.population_size,
            "generations_completed": len(self.history),
            "individuals": [ind.to_dict() for ind in self.population],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="GA optimiser for SDXL inpainting hyperparameters (Phase A)."
    )
    ap.add_argument(
        "--generations", type=int, default=10,
        help="Number of generations to run (default: 10)",
    )
    ap.add_argument(
        "--population-size", type=int, default=20,
        help="Population size per generation (default: 20)",
    )
    ap.add_argument(
        "--mutation-rate", type=float, default=0.2,
        help="Per-parameter mutation probability (default: 0.2)",
    )
    ap.add_argument(
        "--mutation-strength", type=float, default=0.1,
        help="Standard deviation of Gaussian mutation in normalised space (default: 0.1)",
    )
    ap.add_argument(
        "--elite-fraction", type=float, default=0.1,
        help="Fraction of top individuals preserved verbatim (default: 0.1)",
    )
    ap.add_argument(
        "--tournament-size", type=int, default=3,
        help="Tournament selection size (default: 3)",
    )
    ap.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for deterministic runs (default: 42)",
    )
    ap.add_argument(
        "--fitness-json", type=str, default=None,
        help=(
            "Path to JSON file with evaluation data.  "
            "Supported formats:\n"
            "  1) List of genome+metric records (recommended):\n"
            "     each record has all 6 genome fields (mask_cx, mask_cy, "
            "mask_radius, strength, guidance_scale, seed) plus metric fields.\n"
            "     → genome-dependent kNN scoring (optimisation signal).\n"
            "  2) Flat metrics dict (legacy): constant fitness w/ warning.\n"
            "  3) List of metric-only dicts (legacy): averaged fitness w/ warning."
        ),
    )
    ap.add_argument(
        "--mock", action="store_true",
        help="Use mock/synthetic metrics (for smoke testing without real data)",
    )
    ap.add_argument(
        "--output-dir", type=str, default="outputs/ga",
        help="Output directory for GA artifacts (default: outputs/ga)",
    )
    ap.add_argument(
        "--verbose", action="store_true",
        help="Print per-generation details",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without running GA",
    )
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration summary
    config = {
        "generations": args.generations,
        "population_size": args.population_size,
        "mutation_rate": args.mutation_rate,
        "mutation_strength": args.mutation_strength,
        "elite_fraction": args.elite_fraction,
        "tournament_size": args.tournament_size,
        "seed": args.seed,
        "fitness_json": args.fitness_json,
        "mock": args.mock,
        "output_dir": str(output_dir),
    }

    if args.dry_run:
        print("=== DRY RUN ===")
        print(json.dumps(config, indent=2))
        return

    # Resolve fitness-json path
    fitness_json_path: Optional[Path] = None
    if args.fitness_json:
        fitness_json_path = Path(args.fitness_json)
        if not fitness_json_path.exists():
            print(f"ERROR: fitness-json not found: {fitness_json_path}", file=sys.stderr)
            sys.exit(1)

    # Build evaluator
    if args.mock:
        evaluator = JsonMetricsEvaluator(mock=True, seed=args.seed)
    elif fitness_json_path is not None:
        evaluator = JsonMetricsEvaluator(json_path=fitness_json_path)
    else:
        print(
            "ERROR: must provide --fitness-json <path> or --mock for evaluation.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Save config
    config_path = output_dir / "ga_config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    # Run GA
    ga = GAOptimiser(
        evaluator=evaluator,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_fraction=args.elite_fraction,
        tournament_size=args.tournament_size,
        seed=args.seed,
    )

    ga.run(generations=args.generations, verbose=args.verbose)

    # Save outputs
    ga.save_best_genome(output_dir / "best_genome.json")
    ga.save_history(output_dir / "ga_history.csv")
    ga.save_population(output_dir / "population_final.json")

    print(f"\nOutputs saved to: {output_dir}/")
    print(f"  best_genome.json")
    print(f"  ga_history.csv")
    print(f"  population_final.json")


if __name__ == "__main__":
    main()
