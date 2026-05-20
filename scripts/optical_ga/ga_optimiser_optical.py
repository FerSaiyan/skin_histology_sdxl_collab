#!/usr/bin/env python
"""
Genetic Algorithm optimiser for skin biophysical optical parameters.

Searches for a set of skin parameters (genome) that produce a colour
(L\\*a\\*b\\* or ITA) as close as possible to a target.

Two forward modes:
  - **surrogate** (default): lightweight analytical approximation, no
    external dependencies.  Suitable for CI and smoke tests.
  - **realistic**: requires ``xopto`` (MCML) or ``mcx`` binary for Monte
    Carlo simulation.  Prints an error if unavailable.

Three fitness modes:
  - ``lab``: direct L\\*a\\*b\\* matching with normalised denominators.
  - ``ita``: ITA + L\\* matching with a\\* soft constraint [2, 20].
  - ``ita_no_a``: only L\\* + b\\* (ITA) with a\\* soft constraint.

Outputs (to ``--output-dir``):
  - ``best_genome.json`` — best genome found and its Lab/fitness.
  - ``ga_history.csv`` — per-generation stats.
  - ``population_final.json`` — final population with fitness values.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

# Ensure repo root is on sys.path for internal imports
_REPO_ROOT = str(Path(__file__).resolve().parents[2])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from scripts.optical_ga.genome_encoding_optical import (
    default_genome,
    random_genome,
    genome_from_normalised_vector,
    normalised_vector_from_genome,
    genome_to_str,
    mutate_genome,
    crossover_genomes,
    check_genome,
    param_names,
    get_param_defs,
    num_params,
)

from scripts.optical_ga.forward_model import run_forward
from scripts.optical_ga.optical_fitness import compute_fitness, FITNESS_MODES


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class Individual:
    """A single genome with its fitness score."""
    genome: Dict[str, float]
    fitness: float = -float("inf")
    component_scores: Dict[str, float] = field(default_factory=dict)
    lab_sim: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    generation: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome": self.genome,
            "fitness": self.fitness,
            "component_scores": self.component_scores,
            "lab_sim": list(self.lab_sim),
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
    diversity: float
    best_genome: Dict[str, float]
    best_lab: Tuple[float, float, float]
    elapsed_s: float


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class OpticalEvaluator:
    """Evaluates a genome by running the forward model and computing fitness."""

    def __init__(
        self,
        lab_target: Tuple[float, float, float],
        forward_mode: str = "surrogate",
        fitness_mode: str = "lab",
        a_range: Optional[Tuple[float, float]] = None,
        num_photons: int = 1_000_000,
        with_specular: bool = True,
        verbose: bool = False,
    ):
        self.lab_target = lab_target
        self.forward_mode = forward_mode
        self.fitness_mode = fitness_mode
        self.a_range = a_range
        self.num_photons = num_photons
        self.with_specular = with_specular
        self.verbose = verbose

    def __call__(
        self, genome: Dict[str, float]
    ) -> Tuple[float, Dict[str, float], Tuple[float, float, float]]:
        """Run forward model and compute fitness.

        Returns (fitness, component_scores, lab_sim).
        """
        try:
            lab_sim, reflectance, rgb, bulk = run_forward(
                genome,
                mode=self.forward_mode,
                num_photons=self.num_photons,
                with_specular=self.with_specular,
            )
        except Exception as e:
            if self.verbose:
                print(f"  [WARN] Forward model failed: {e}", file=sys.stderr)
            return 0.0, {"error": str(e)}, (0.0, 0.0, 0.0)

        fitness, comps = compute_fitness(
            lab_sim,
            self.lab_target,
            mode=self.fitness_mode,
            a_range=self.a_range,
        )

        if self.verbose:
            print(
                f"  L={lab_sim[0]:.2f} a={lab_sim[1]:.2f} b={lab_sim[2]:.2f} "
                f"→ fitness={fitness:.4f}"
            )

        return fitness, comps, lab_sim


# ---------------------------------------------------------------------------
# GA loop
# ---------------------------------------------------------------------------


class GAOptimiserOptical:
    """Genetic algorithm for optical parameter optimisation."""

    def __init__(
        self,
        evaluator: Callable[
            [Dict[str, float]],
            Tuple[float, Dict[str, float], Tuple[float, float, float]],
        ],
        population_size: int = 50,
        mutation_rate: float = 0.2,
        mutation_strength: float = 0.1,
        elite_fraction: float = 0.1,
        tournament_size: int = 3,
        seed: int = 42,
        use_dermal_chromophores: bool = False,
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
        self.use_dermal_chromophores = use_dermal_chromophores

        self.population: List[Individual] = []
        self.history: List[GenerationStats] = []
        self.best_individual: Optional[Individual] = None

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_population(
        self, seed_genomes: Optional[List[Dict[str, float]]] = None
    ) -> None:
        """Initialise population."""
        self.population = []
        if seed_genomes:
            for g in seed_genomes:
                ind = Individual(genome=g.copy())
                self.population.append(ind)
                if len(self.population) >= self.population_size:
                    break

        while len(self.population) < self.population_size:
            g = random_genome(self.rng, use_dermal_chromophores=self.use_dermal_chromophores)
            ind = Individual(genome=g)
            self.population.append(ind)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate_population(self, generation: int, verbose: bool = False) -> None:
        """Evaluate all individuals."""
        for ind in self.population:
            if ind.fitness > -float("inf") and ind.generation == generation:
                continue
            ind.fitness, comps, lab_sim = self.evaluator(ind.genome)
            ind.component_scores = comps
            ind.lab_sim = lab_sim
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
        self.population.sort(key=lambda ind: ind.fitness, reverse=True)

        next_pop: List[Individual] = []

        # Elitism
        for i in range(self.elite_count):
            elite = Individual(
                genome=self.population[i].genome.copy(),
                fitness=self.population[i].fitness,
                component_scores=self.population[i].component_scores.copy(),
                lab_sim=self.population[i].lab_sim,
                generation=generation,
            )
            next_pop.append(elite)

        # Fill rest via crossover + mutation
        while len(next_pop) < self.population_size:
            parent_a = self.tournament_select()
            parent_b = self.tournament_select()

            child1_genome, child2_genome = crossover_genomes(
                parent_a.genome,
                parent_b.genome,
                self.rng,
                use_dermal_chromophores=self.use_dermal_chromophores,
            )

            child1_genome = mutate_genome(
                child1_genome,
                self.mutation_rate,
                self.mutation_strength,
                self.rng,
                use_dermal_chromophores=self.use_dermal_chromophores,
            )
            child2_genome = mutate_genome(
                child2_genome,
                self.mutation_rate,
                self.mutation_strength,
                self.rng,
                use_dermal_chromophores=self.use_dermal_chromophores,
            )

            child1 = Individual(genome=child1_genome, generation=generation)
            child2 = Individual(genome=child2_genome, generation=generation)

            next_pop.append(child1)
            if len(next_pop) < self.population_size:
                next_pop.append(child2)

        self.population = next_pop[: self.population_size]

    # ------------------------------------------------------------------
    # Diversity
    # ------------------------------------------------------------------

    def compute_diversity(self) -> float:
        """Mean Euclidean distance between all pairs in normalised space."""
        if len(self.population) < 2:
            return 0.0
        vectors = [
            normalised_vector_from_genome(
                ind.genome, use_dermal_chromophores=self.use_dermal_chromophores
            )
            for ind in self.population
        ]
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

            fitnesses = [ind.fitness for ind in self.population]
            best_idx = int(np.argmax(fitnesses))
            best_fit = float(fitnesses[best_idx])
            mean_fit = float(np.mean(fitnesses))
            worst_fit = float(np.min(fitnesses))
            median_fit = float(np.median(fitnesses))
            std_fit = float(np.std(fitnesses))
            diversity = self.compute_diversity()
            best_gen = self.population[best_idx].genome.copy()
            best_lab = self.population[best_idx].lab_sim

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
                best_lab=best_lab,
                elapsed_s=elapsed,
            )
            self.history.append(stats)

            if self.best_individual is None or best_fit > self.best_individual.fitness:
                self.best_individual = Individual(
                    genome=best_gen.copy(),
                    fitness=best_fit,
                    component_scores=self.population[best_idx].component_scores.copy(),
                    lab_sim=best_lab,
                    generation=gen,
                )

            if verbose:
                print(
                    f"Gen {gen:3d}: best={best_fit:.4f}  mean={mean_fit:.4f}  "
                    f"diversity={diversity:.4f}  "
                    f"L={best_lab[0]:.2f} a={best_lab[1]:.2f} b={best_lab[2]:.2f}"
                )

            if gen < generations - 1:
                self.evolve(gen + 1)

        total_elapsed = time.perf_counter() - t_start
        if verbose:
            print(f"\nGA finished in {total_elapsed:.2f}s")
            if self.best_individual:
                print(f"Best fitness: {self.best_individual.fitness:.4f}")
                print(f"Best Lab: {self.best_individual.lab_sim}")

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
            "lab_sim": list(self.best_individual.lab_sim),
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
        pnames = param_names(use_dermal_chromophores=self.use_dermal_chromophores)
        fieldnames = [
            "generation",
            "best_fitness",
            "mean_fitness",
            "worst_fitness",
            "median_fitness",
            "std_fitness",
            "diversity",
            "elapsed_s",
            "best_L",
            "best_a",
            "best_b",
        ] + [f"best_{n}" for n in pnames]

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for h in self.history:
                g = h.best_genome
                row = {
                    "generation": h.generation,
                    "best_fitness": f"{h.best_fitness:.6f}",
                    "mean_fitness": f"{h.mean_fitness:.6f}",
                    "worst_fitness": f"{h.worst_fitness:.6f}",
                    "median_fitness": f"{h.median_fitness:.6f}",
                    "std_fitness": f"{h.std_fitness:.6f}",
                    "diversity": f"{h.diversity:.6f}",
                    "elapsed_s": f"{h.elapsed_s:.4f}",
                    "best_L": f"{h.best_lab[0]:.4f}",
                    "best_a": f"{h.best_lab[1]:.4f}",
                    "best_b": f"{h.best_lab[2]:.4f}",
                }
                for name in pnames:
                    row[f"best_{name}"] = f"{g.get(name, 0.0):.6e}"
                writer.writerow(row)

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
        description=(
            "GA optimiser for skin biophysical optical parameters.\n\n"
            "Searches for skin parameters that best match a target colour "
            "(L\\*a\\*b\\* or ITA).  Uses a surrogate forward model by "
            "default (lightweight, no external deps)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Target colour
    target_group = ap.add_argument_group("Target colour")
    target_group.add_argument(
        "--target-L", type=float, default=60.0,
        help="Target CIE L\\* (default: 60.0)",
    )
    target_group.add_argument(
        "--target-a", type=float, default=10.0,
        help="Target CIE a\\* (default: 10.0)",
    )
    target_group.add_argument(
        "--target-b", type=float, default=15.0,
        help="Target CIE b\\* (default: 15.0)",
    )

    # GA parameters
    ga_group = ap.add_argument_group("GA settings")
    ga_group.add_argument(
        "--generations", type=int, default=5,
        help="Number of generations (default: 5)",
    )
    ga_group.add_argument(
        "--population-size", type=int, default=20,
        help="Population size per generation (default: 20)",
    )
    ga_group.add_argument(
        "--mutation-rate", type=float, default=0.2,
        help="Per-parameter mutation probability (default: 0.2)",
    )
    ga_group.add_argument(
        "--mutation-strength", type=float, default=0.1,
        help="Mutation stddev in normalised space (default: 0.1)",
    )
    ga_group.add_argument(
        "--elite-fraction", type=float, default=0.1,
        help="Fraction of top individuals preserved (default: 0.1)",
    )
    ga_group.add_argument(
        "--tournament-size", type=int, default=3,
        help="Tournament selection size (default: 3)",
    )
    ga_group.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Mode selection
    mode_group = ap.add_argument_group("Mode selection")
    mode_group.add_argument(
        "--forward-mode", type=str, default="surrogate",
        choices=["surrogate", "realistic"],
        help="Forward model mode (default: surrogate). "
        "Use 'realistic' for MC backend (requires xopto/mcx).",
    )
    mode_group.add_argument(
        "--fitness-mode", type=str, default="lab",
        choices=FITNESS_MODES,
        help="Fitness mode (default: lab)",
    )
    mode_group.add_argument(
        "--use-dermal-chromophores", action="store_true",
        help="Include 4 optional dermal chromophores (bilirubin, etc.)",
    )

    # Output
    output_group = ap.add_argument_group("Output")
    output_group.add_argument(
        "--output-dir", type=str, default="outputs/optical_ga",
        help="Output directory (default: outputs/optical_ga)",
    )
    output_group.add_argument(
        "--verbose", action="store_true",
        help="Print per-generation details",
    )
    output_group.add_argument(
        "--dry-run", action="store_true",
        help="Print configuration and exit without running",
    )

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Configuration summary
    lab_target = (args.target_L, args.target_a, args.target_b)
    config = {
        "lab_target": list(lab_target),
        "generations": args.generations,
        "population_size": args.population_size,
        "mutation_rate": args.mutation_rate,
        "mutation_strength": args.mutation_strength,
        "elite_fraction": args.elite_fraction,
        "tournament_size": args.tournament_size,
        "seed": args.seed,
        "forward_mode": args.forward_mode,
        "fitness_mode": args.fitness_mode,
        "use_dermal_chromophores": args.use_dermal_chromophores,
        "output_dir": str(output_dir),
    }

    if args.dry_run:
        print("=== DRY RUN ===")
        print(json.dumps(config, indent=2))
        print(f"Genome has {num_params(args.use_dermal_chromophores)} parameters")
        print(f"Parameter names: {param_names(args.use_dermal_chromophores)}")
        return

    print(f"=== Optical GA Optimiser ===")
    print(f"  Target Lab : {lab_target}")
    print(f"  Forward    : {args.forward_mode}")
    print(f"  Fitness    : {args.fitness_mode}")
    print(f"  Generations: {args.generations}")
    print(f"  Population : {args.population_size}")
    print()

    # Build evaluator
    evaluator = OpticalEvaluator(
        lab_target=lab_target,
        forward_mode=args.forward_mode,
        fitness_mode=args.fitness_mode,
        num_photons=1_000_000,
        verbose=args.verbose,
    )

    # Run GA
    ga = GAOptimiserOptical(
        evaluator=evaluator,
        population_size=args.population_size,
        mutation_rate=args.mutation_rate,
        mutation_strength=args.mutation_strength,
        elite_fraction=args.elite_fraction,
        tournament_size=args.tournament_size,
        seed=args.seed,
        use_dermal_chromophores=args.use_dermal_chromophores,
    )

    ga.run(generations=args.generations, verbose=True)

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
