"""Lightweight tests for optical GA genome encoding, forward model, and fitness."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.optical_ga.genome_encoding_optical import (
    WAVELENGTHS,
    NUM_WAVELENGTHS,
    CORE_PARAM_DEFS,
    DERMAL_CHROMOPHORE_DEFS,
    default_genome,
    random_genome,
    genome_from_normalised_vector,
    normalised_vector_from_genome,
    num_params,
    param_names,
    param_bounds,
    check_genome,
    mutate_genome,
    crossover_genomes,
    get_param_defs,
)
from scripts.optical_ga.forward_model import (
    run_forward,
    run_surrogate,
)
from scripts.optical_ga.optical_fitness import (
    compute_fitness,
    compute_lab_fitness,
    compute_ita_fitness,
    compute_ita_no_a_fitness,
)
from scripts.optical_ga.colorimetry import (
    calculate_ita,
    classify_ita,
    reflectance_to_lab,
)


class TestGenomeEncoding:
    def test_num_params_core(self):
        """19 core parameters."""
        assert num_params(use_dermal_chromophores=False) == 19

    def test_num_params_with_dermal(self):
        """23 parameters with dermal chromophores."""
        assert num_params(use_dermal_chromophores=True) == 23

    def test_wavelengths(self):
        """81 wavelengths from 380 to 780 nm, step 5."""
        assert len(WAVELENGTHS) == 81
        assert WAVELENGTHS[0] == 380
        assert WAVELENGTHS[-1] == 780
        assert NUM_WAVELENGTHS == 81

    def test_default_genome(self):
        """Default genome returns a dict with all expected keys."""
        genome = default_genome()
        names = param_names()
        for name in names:
            assert name in genome
        assert len(genome) == 19

    def test_default_genome_with_dermal(self):
        """Default genome with dermal chromophores has 23 keys."""
        genome = default_genome(use_dermal_chromophores=True)
        assert len(genome) == 23
        # Dermal chromophores default to 0.0
        assert genome["bilirubin_layer1"] == 0.0

    def test_random_genome(self):
        """Random genome values are within bounds."""
        rng = random.Random(42)
        genome = random_genome(rng)
        for name, low, high, _, _ in CORE_PARAM_DEFS:
            val = genome[name]
            assert low <= val <= high, f"{name}: {val} not in [{low}, {high}]"

    def test_normalised_roundtrip(self):
        """Genome → normalised vector → genome preserves values."""
        genome = default_genome()
        vec = normalised_vector_from_genome(genome)
        genome2 = genome_from_normalised_vector(vec)
        for key in genome:
            assert abs(genome[key] - genome2[key]) < 1e-10, f"Mismatch for {key}"

    def test_check_genome_valid(self):
        """check_genome returns empty for valid genome."""
        genome = default_genome()
        issues = check_genome(genome)
        assert issues == []

    def test_check_genome_missing(self):
        """check_genome flags missing keys."""
        issues = check_genome({"melanin": 0.1})
        assert len(issues) > 0

    def test_check_genome_out_of_bounds(self):
        """check_genome flags out-of-bounds values."""
        genome = default_genome()
        genome["melanin"] = 10.0  # way out of bounds
        issues = check_genome(genome)
        assert any("melanin" in i for i in issues)

    def test_mutation_produces_valid_genome(self):
        """Mutation stays within bounds."""
        rng = random.Random(42)
        genome = default_genome()
        for _ in range(100):
            mutated = mutate_genome(genome, 0.5, 0.3, rng)
            issues = check_genome(mutated)
            assert issues == []

    def test_crossover_produces_valid_genomes(self):
        """Crossover children are valid genomes."""
        rng = random.Random(42)
        genome_a = default_genome()
        genome_b = random_genome(rng)
        c1, c2 = crossover_genomes(genome_a, genome_b, rng)
        assert check_genome(c1) == []
        assert check_genome(c2) == []

    def test_param_names_list(self):
        """param_names returns expected core list."""
        names = param_names()
        assert "melanin" in names
        assert "blood_layer1" in names
        assert "n_mult_layer2" in names
        assert "bilirubin_layer1" not in names  # not in core

    def test_param_names_with_dermal(self):
        """param_names includes dermal when requested."""
        names = param_names(use_dermal_chromophores=True)
        assert "bilirubin_layer1" in names
        assert "methb_layer1" in names

    def test_param_bounds(self):
        """param_bounds returns expected range."""
        low, high = param_bounds("melanin")
        assert low == 0.0
        assert high == 0.5

    def test_get_param_defs(self):
        """get_param_defs returns list of tuples."""
        defs = get_param_defs()
        assert len(defs) == 19
        assert isinstance(defs[0], tuple)
        assert len(defs[0]) == 5  # name, low, high, default, description


class TestForwardModel:
    def test_surrogate_runs(self):
        """Surrogate forward model runs without error."""
        genome = default_genome()
        lab, reflectance, rgb, bulk = run_surrogate(genome)
        assert len(lab) == 3
        assert isinstance(reflectance, np.ndarray)
        assert reflectance.shape == (81,)
        assert all(0.0 <= r <= 1.0 for r in reflectance)
        assert len(rgb) == 3

    def test_surrogate_bulk_props(self):
        """Surrogate returns per-layer mu_a and mu_s."""
        genome = default_genome()
        lab, reflectance, rgb, bulk = run_surrogate(genome)
        for key in ("layer0_mua", "layer0_mus", "layer1_mua", "layer1_mus",
                     "layer2_mua", "layer2_mus", "bulk_mu_a", "bulk_mu_s"):
            assert key in bulk
            assert len(bulk[key]) == 81

    def test_surrogate_different_genomes(self):
        """Different genomes produce different results."""
        rng = random.Random(42)
        g1 = default_genome()
        g2 = random_genome(rng)
        lab1, _, _, _ = run_surrogate(g1)
        lab2, _, _, _ = run_surrogate(g2)
        # The two outputs should differ (different parameters)
        assert lab1 != lab2

    def test_run_forward_surrogate_mode(self):
        """run_forward with mode='surrogate' works."""
        genome = default_genome()
        lab, reflectance, rgb, bulk = run_forward(genome, mode="surrogate")
        assert len(lab) == 3

    def test_run_forward_invalid_mode(self):
        """run_forward with invalid mode raises ValueError."""
        with pytest.raises(ValueError):
            run_forward(default_genome(), mode="invalid")


class TestOpticalFitness:
    def test_lab_fitness_perfect_match(self):
        """Perfect Lab match gives high fitness."""
        lab = (60.0, 10.0, 15.0)
        fitness, comps = compute_lab_fitness(lab, lab)
        assert fitness > 100.0  # near-perfect match
        assert comps["total_err"] < 1e-6

    def test_lab_fitness_poor_match(self):
        """Poor Lab match gives low fitness."""
        target = (60.0, 10.0, 15.0)
        far = (30.0, 30.0, 30.0)
        fitness, comps = compute_lab_fitness(far, target)
        assert 0.0 < fitness < 10.0  # not a perfect match

    def test_lab_fitness_a_range(self):
        """a* outside constraint range is penalised."""
        target = (60.0, 10.0, 15.0)
        far_a = (60.0, 50.0, 15.0)  # a* way too high
        fitness_no_constraint, _ = compute_lab_fitness(far_a, target, a_range=None)
        fitness_with_constraint, _ = compute_lab_fitness(far_a, target, a_range=(2, 20))
        # With constraint, penalty should reduce fitness
        assert fitness_with_constraint <= fitness_no_constraint + 1e-6

    def test_ita_fitness(self):
        """ITA mode runs without error."""
        target = (60.0, 10.0, 15.0)
        sim = (58.0, 12.0, 14.0)
        fitness, comps = compute_ita_fitness(sim, target)
        assert fitness > 0
        assert "ita_sim" in comps
        assert "ita_target" in comps

    def test_ita_no_a_fitness(self):
        """ITA_no_a mode runs without error."""
        target = (60.0, 10.0, 15.0)
        sim = (58.0, 12.0, 14.0)
        fitness, comps = compute_ita_no_a_fitness(sim, target)
        assert fitness > 0

    def test_compute_fitness_dispatch(self):
        """compute_fitness dispatches to correct mode."""
        target = (60.0, 10.0, 15.0)
        sim = (58.0, 12.0, 14.0)
        f_lab, _ = compute_fitness(sim, target, mode="lab")
        f_ita, _ = compute_fitness(sim, target, mode="ita")
        f_no_a, _ = compute_fitness(sim, target, mode="ita_no_a")
        assert f_lab > 0
        assert f_ita > 0
        assert f_no_a > 0

    def test_invalid_mode(self):
        """Invalid fitness mode raises ValueError."""
        with pytest.raises(ValueError):
            compute_fitness((60, 10, 15), (60, 10, 15), mode="invalid")


class TestColorimetry:
    def test_calculate_ita(self):
        """ITA angle calculation."""
        # L=70, b=15 → ITA ≈ arctan(20/15) ≈ 53.1 degrees
        ita = calculate_ita(70.0, 15.0)
        assert 50.0 < ita < 56.0

    def test_ita_negative_b(self):
        """ITA with negative b*."""
        ita = calculate_ita(30.0, -5.0)
        # L-50 = -20, b = -5, arctan(4) ≈ 76 degrees
        assert 70.0 < ita < 85.0

    def test_classify_ita(self):
        """ITA classification categories."""
        assert classify_ita(60) == "Very Light"
        assert classify_ita(45) == "Light"
        assert classify_ita(35) == "Intermediate"
        assert classify_ita(20) == "Tan"
        assert classify_ita(0) == "Brown"
        assert classify_ita(-40) == "Dark"


class TestGANoExternalDeps:
    """Verify the GA runs without colour-science installed (graceful fallback)."""

    def test_surrogate_returns_plausible_lab(self):
        """Surrogate returns lab-like tuple even without colour-science."""
        genome = default_genome()
        lab, reflectance, rgb, bulk = run_forward(genome, mode="surrogate")
        assert len(lab) == 3
        # L* should be in reasonable range (not NaN)
        assert np.isfinite(lab[0])
        assert np.isfinite(lab[1])
        assert np.isfinite(lab[2])
        # Reflectance should be non-negative
        assert np.all(reflectance >= 0.0)
