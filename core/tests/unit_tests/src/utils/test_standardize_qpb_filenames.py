"""
Unit tests for the standardize_qpb_filenames.py utility script.

Tests cover value normalization and spelling, filename tokenization
(checked against the project's filename parser), parameter ordering,
the two-phase renaming with rollback, sidecar updates, and full
command-line runs (dry run, apply, undo, and the safety stops) on small
synthetic data files sets built in a temporary directory.
"""

import json
import os

import pytest
import yaml
from click.testing import CliRunner

from src.utils import standardize_qpb_filenames as sqf


# =============================================================================
# HELPERS
# =============================================================================

KL_PREFIX = "KL_Brillouin_cSW0_rho1p0_EpsMSCG1e-12_mu1p0_EpsCG1e-12"


def log_contents(
    mass=0.02, order=1, config="0056800", kernel="Brillouin", rho="1.00"
) -> str:
    """Minimal qpb invert log file with the lines the parser reads."""
    lines = [
        f"Dslash operator is {kernel}",
        "Gauge field (raw_32) = /p/Nf0_b6p20_L24T48-APE/"
        f"conf_Nf0_b6p20_L24T48_apeN1a0p72.{config}",
        f"mass = {mass}",
        "Clover param = 0",
        f"rho = {rho}",
        f"KL iters = {order}",
        "Mu = 1.000",
        "Solver = CG",
        "Outer solver epsilon = 1.000000e-12",
        "Inner solver epsilon = 1.000000e-12",
        "CG done, 12 vectors in t = 10.5 sec",
    ]
    return "".join(f" {line}\n" for line in lines)


def write_group(directory, stem, with_correlators=True, **log_values):
    """Write "<stem>.txt" (and "<stem>.dat") into a set directory."""
    with open(os.path.join(directory, stem + ".txt"), "w") as file:
        file.write(log_contents(**log_values))
    if with_correlators:
        with open(os.path.join(directory, stem + ".dat"), "w") as file:
            file.write(f"correlators of {stem}\n")


def snapshot(directory):
    """{filename: contents} of every file in a directory."""
    return {
        name: open(os.path.join(directory, name)).read()
        for name in sorted(os.listdir(directory))
    }


@pytest.fixture
def specifications():
    return sqf.build_token_specifications()


@pytest.fixture
def mixed_mass_set(tmp_path):
    """
    The motivating case: an am=0.02 batch named without the bare mass and
    an am=0.04 batch named with it, plus ".yaml" and ".json" sidecars.
    """
    parent = tmp_path / "raw" / "invert"
    set_directory = parent / "KL_demo"
    set_directory.mkdir(parents=True)
    for n in (1, 2, 3):
        write_group(
            set_directory,
            f"{KL_PREFIX}_n{n}_config0056800",
            mass=0.02,
            order=n,
            config="0056800",
        )
        write_group(
            set_directory,
            f"{KL_PREFIX}_n{n}_m0p04_config0051200",
            mass=0.04,
            order=n,
            config="0051200",
        )
    (parent / "KL_demo.yaml").write_text(
        'experiment:\n  name: "KL_demo"   # set name\n  date: "2026-11-19"\n'
    )
    (parent / "KL_demo.json").write_text(
        json.dumps(
            {
                "filter_type": "filename",
                "include": [],
                "exclude": [
                    f"{KL_PREFIX}_n1_config0056800",
                    f"{KL_PREFIX}_n1_config0056800.txt",
                    "unrelated_entry",
                ],
            }
        )
    )
    return set_directory


def run_script(arguments, answers=""):
    result = CliRunner().invoke(sqf.main, arguments, input=answers)
    return result


# =============================================================================
# VALUE HELPERS
# =============================================================================


class TestComparable:
    def test_float_rounding_noise_compares_equal(self):
        assert sqf.values_agree(0.1 + 0.2, 0.3)

    def test_different_floats_differ(self):
        assert not sqf.values_agree(0.02, 0.04)

    def test_int_and_float_with_same_value_agree(self):
        assert sqf.values_agree(0, 0.0)

    def test_whitespace_in_strings_is_normalized(self):
        assert sqf.values_agree("(1, 4, 4, 4)", "(1,4,4,4)")
        assert sqf.comparable("Diagonal  KL") == "Diagonal KL"


class TestFormatNewSpelling:
    def test_decimals_follow_existing_spellings(self, specifications):
        spelling = sqf.format_new_spelling(
            specifications["Bare_mass"], 0.1, ["0p04", "0p02"]
        )
        assert spelling == "0p10"

    def test_mixed_existing_decimals_fall_back_to_shortest(self, specifications):
        spelling = sqf.format_new_spelling(
            specifications["Bare_mass"], 0.1, ["0p004", "0p02"]
        )
        assert spelling == "0p1"

    def test_whole_float_keeps_one_decimal(self, specifications):
        assert sqf.format_new_spelling(specifications["Rho_value"], 1.0, []) == "1p0"

    def test_exponent_style_parameter(self, specifications):
        spec = specifications["Outer_solver_epsilon"]
        assert sqf.format_new_spelling(spec, 1e-12, []) == "1e-12"
        assert sqf.format_new_spelling(spec, 2.5e-10, []) == "2p5e-10"

    def test_integer_parameter(self, specifications):
        assert sqf.format_new_spelling(specifications["Rational_order"], 7, []) == "7"

    def test_value_map_is_reversed(self, specifications):
        spec = specifications["Overlap_operator_method"]
        assert sqf.format_new_spelling(spec, "Diagonal KL", []) == "KL"


# =============================================================================
# FILENAME TOKENS
# =============================================================================


class TestTokenSpecifications:
    def test_labels_are_derived_from_filename_patterns(self, specifications):
        assert specifications["Bare_mass"].label == "m"
        assert specifications["Rational_order"].label == "n"
        assert specifications["Outer_solver_epsilon"].label == "EpsCG"
        assert specifications["Overlap_operator_method"].label == ""

    def test_lattice_geometry_is_excluded(self, specifications):
        assert "Lattice_geometry" not in specifications


class TestParseFilenameStem:
    def test_parameter_tokens_keep_order_and_spelling(self, specifications):
        parsed = sqf.parse_filename_stem(
            f"{KL_PREFIX}_n3_m0p04_config0051200", specifications
        )
        assert [p for p, _ in parsed.parameter_tokens][:3] == [
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Clover_coefficient",
        ]
        assert ("Bare_mass", "0p04") in parsed.parameter_tokens
        assert parsed.values["Rational_order"] == 3
        assert parsed.values["Configuration_label"] == "0051200"
        assert parsed.extra_tokens == []

    def test_extra_text_is_collected_in_order(self, specifications):
        parsed = sqf.parse_filename_stem(
            "KL_Brillouin_EpsCG1e-12_foo_n1_config0056800_bar", specifications
        )
        assert parsed.extra_tokens == ["foo", "bar"]

    def test_glued_tokens_are_split(self, specifications):
        parsed = sqf.parse_filename_stem(
            "Zolotarev_Brillouin_n4_EpsCG1e-12m0p02_config0007200", specifications
        )
        assert parsed.split_tokens == ["EpsCG1e-12m0p02"]
        assert parsed.values["Bare_mass"] == 0.02
        assert parsed.values["Outer_solver_epsilon"] == 1e-12
        assert parsed.extra_tokens == []

    def test_token_unknown_to_project_parser_is_extra_text(self, specifications):
        # The project's "_NSpinors(\d+)_" pattern needs a trailing "_"
        parsed = sqf.parse_filename_stem(
            "KL_Brillouin_n1_config0056800_NSpinors1", specifications
        )
        assert "Number_of_spinors" not in parsed.values
        assert parsed.extra_tokens == ["NSpinors1"]

    def test_repeated_parameter_stops(self, specifications):
        with pytest.raises(sqf.StandardizationError, match="appears twice"):
            sqf.parse_filename_stem("KL_Brillouin_m0p01_n1_m0p02", specifications)

    def test_empty_token_stops(self, specifications):
        with pytest.raises(sqf.StandardizationError, match="empty token"):
            sqf.parse_filename_stem("KL_Brillouin__n1", specifications)


# =============================================================================
# PARAMETER ORDER
# =============================================================================


def make_group(stem, specifications, **extra_values):
    parsed = sqf.parse_filename_stem(stem, specifications)
    return sqf.FileGroup(
        stem=stem,
        extensions=[".txt"],
        parsed=parsed,
        contents_values={},
        effective_values={**parsed.values, **extra_values},
        project_additional_text="_".join(parsed.extra_tokens),
    )


def summaries_for(groups, specifications):
    return sqf.summarize_parameters(groups, specifications, several_max=5)


class TestMergeParametersOrder:
    def test_most_common_complete_order_is_reused(self, specifications):
        stems = [
            "KL_Brillouin_m0p01_n1_config0000200",
            "KL_Brillouin_m0p02_n1_config0000200",
            "KL_Brillouin_n2_m0p01_config0000200",
        ]
        groups = [make_group(s, specifications) for s in stems]
        included = [
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Bare_mass",
            "Rational_order",
            "Configuration_label",
        ]
        order = sqf.merge_parameters_order(
            groups, included, summaries_for(groups, specifications)
        )
        assert order[2:] == ["Bare_mass", "Rational_order", "Configuration_label"]

    def test_parameter_missing_from_names_goes_before_more_varied_one(
        self, specifications
    ):
        stems = [f"KL_Brillouin_n{n}_config0000200" for n in range(1, 6)]
        groups = [
            make_group(s, specifications, Bare_mass=0.01 * (1 + i % 2))
            for i, s in enumerate(stems)
        ]
        included = [
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Rational_order",
            "Configuration_label",
            "Bare_mass",
        ]
        order = sqf.merge_parameters_order(
            groups, included, summaries_for(groups, specifications)
        )
        assert order.index("Bare_mass") < order.index("Rational_order")


# =============================================================================
# FILE OPERATIONS
# =============================================================================


class TestRenameFilesTwoPhase:
    @pytest.fixture
    def directory(self, tmp_path):
        for name in "ABC":
            (tmp_path / name).write_text(name)
        return tmp_path

    def test_swap_does_not_collide(self, directory):
        sqf.rename_files_two_phase(str(directory), [("A", "B"), ("B", "A")])
        assert snapshot(directory) == {"A": "B", "B": "A", "C": "C"}

    def test_failure_rolls_back_everything(self, directory, monkeypatch):
        original = snapshot(directory)
        real_rename, calls = os.rename, []

        def failing_rename(source, target):
            calls.append(source)
            if len(calls) == 5:  # During the second phase
                raise OSError("simulated failure")
            real_rename(source, target)

        monkeypatch.setattr(sqf.os, "rename", failing_rename)
        with pytest.raises(OSError, match="simulated"):
            sqf.rename_files_two_phase(
                str(directory), [("A", "X"), ("B", "Y"), ("C", "Z")]
            )
        monkeypatch.setattr(sqf.os, "rename", real_rename)
        assert snapshot(directory) == original

    def test_existing_target_is_never_overwritten(self, directory):
        with pytest.raises(sqf.StandardizationError, match="already exists"):
            sqf.rename_files_two_phase(str(directory), [("A", "C")])
        assert snapshot(directory) == {"A": "A", "B": "B", "C": "C"}

    def test_missing_source_stops_before_any_rename(self, directory):
        with pytest.raises(sqf.StandardizationError, match="Missing source"):
            sqf.rename_files_two_phase(str(directory), [("A", "X"), ("Q", "Y")])
        assert snapshot(directory) == {"A": "A", "B": "B", "C": "C"}


class TestSidecarUpdates:
    def test_json_entries_with_and_without_extension(self, tmp_path):
        path = tmp_path / "set.json"
        path.write_text(
            json.dumps(
                {
                    "filter_type": "filename",
                    "include": ["old.txt"],
                    "exclude": ["old", "other"],
                }
            )
        )
        replacements = sqf.update_json_filters(str(path), {"old": "new"})
        data = json.loads(path.read_text())
        assert data == {
            "filter_type": "filename",
            "include": ["new.txt"],
            "exclude": ["new", "other"],
        }
        assert replacements == {"old.txt": "new.txt", "old": "new"}

    def test_json_without_matches_is_left_untouched(self, tmp_path):
        path = tmp_path / "set.json"
        path.write_text('{"include": ["other"]}')
        assert sqf.update_json_filters(str(path), {"old": "new"}) == {}
        assert path.read_text() == '{"include": ["other"]}'

    def test_yaml_experiment_name_keeps_quotes_and_comment(self, tmp_path):
        path = tmp_path / "set.yaml"
        path.write_text(
            'experiment:\n  name: "old_set"   # comment\n  other: old_set\n'
        )
        assert sqf.replace_yaml_experiment_name(str(path), "old_set", "new_set")
        assert path.read_text() == (
            'experiment:\n  name: "new_set"   # comment\n  other: old_set\n'
        )


# =============================================================================
# COMMAND LINE RUNS
# =============================================================================


class TestDryRun:
    def test_dry_run_changes_nothing(self, mixed_mass_set):
        parent = mixed_mass_set.parent
        before, sidecars_before = snapshot(mixed_mass_set), sorted(os.listdir(parent))
        result = run_script(["-set_dir", str(mixed_mass_set), "-defaults"])
        assert result.exit_code == 0, result.output
        # Only the batch named without the bare mass needs renaming
        assert "3 of 6 file groups would be renamed" in result.output
        assert "Dry run: nothing was changed" in result.output
        assert snapshot(mixed_mass_set) == before
        assert sorted(os.listdir(parent)) == sidecars_before

    def test_missing_mass_is_added_with_existing_spelling_style(
        self, mixed_mass_set, tmp_path
    ):
        plan_path = tmp_path / "plan.yaml"
        result = run_script(
            ["-set_dir", str(mixed_mass_set), "-defaults", "-plan_out", str(plan_path)]
        )
        assert result.exit_code == 0, result.output
        plan = yaml.safe_load(plan_path.read_text())
        new_names = {entry["new"] for entry in plan["file_renames"]}
        assert f"{KL_PREFIX}_n1_m0p02_config0056800.txt" in new_names
        assert plan["status"] == "planned"

    def test_constant_known_only_from_filename_cannot_be_dropped(self, tmp_path):
        set_directory = tmp_path / "KL_fixed_n"
        set_directory.mkdir()
        for mass, spelling in ((0.01, "0p01"), (0.02, "0p02")):
            write_group(
                set_directory,
                f"KL_Brillouin_n4_m{spelling}_config0056800",
                mass=mass,
                order=4,
            )
        result = run_script(["-set_dir", str(set_directory)], answers="\n\n")
        assert result.exit_code == 0, result.output
        assert "Constants kept because" in result.output
        # "n" and "config" are constant; only "config" is in the log files
        assert "Candidates: config\n" in result.output

    def test_parameter_known_nowhere_is_omitted_not_fatal(self, tmp_path):
        set_directory = tmp_path / "KL_some_preconditioned"
        set_directory.mkdir()
        write_group(
            set_directory, "KL_Brillouin_n1_m0p01_nPrec2_config0056800", mass=0.01
        )
        write_group(set_directory, "KL_Brillouin_n1_m0p02_config0056800", mass=0.02)
        result = run_script(["-set_dir", str(set_directory), "-defaults"])
        assert result.exit_code == 0, result.output
        assert "Omitted where the value is known nowhere" in result.output
        assert "Nothing to do" in result.output


class TestSafetyStops:
    def test_filename_contents_disagreement_stops(self, tmp_path):
        set_directory = tmp_path / "KL_conflict"
        set_directory.mkdir()
        write_group(set_directory, "KL_Brillouin_n1_m0p04_config0056800", mass=0.05)
        before = snapshot(set_directory)
        result = run_script(
            ["-set_dir", str(set_directory), "-defaults", "-apply"], answers="y\n"
        )
        assert result.exit_code == 1
        assert (
            "Bare_mass is 0.04 in the filename but 0.05 in the log file"
            in result.output
        )
        assert "Nothing was changed" in result.output
        assert snapshot(set_directory) == before

    def test_duplicate_runs_under_two_names_stop(self, tmp_path):
        set_directory = tmp_path / "KL_duplicates"
        set_directory.mkdir()
        write_group(set_directory, "KL_Brillouin_n1_config0056800", mass=0.02)
        write_group(set_directory, "KL_Brillouin_config0056800_n1", mass=0.02)
        before = snapshot(set_directory)
        result = run_script(
            ["-set_dir", str(set_directory), "-defaults", "-apply"], answers="y\n"
        )
        assert result.exit_code == 1
        assert "would all be named" in result.output
        assert snapshot(set_directory) == before

    def test_leftover_temporary_file_stops(self, mixed_mass_set):
        (mixed_mass_set / f"x.txt{sqf.TEMPORARY_RENAME_SUFFIX}").write_text("")
        result = run_script(["-set_dir", str(mixed_mass_set), "-defaults"])
        assert result.exit_code == 1
        assert "Leftover temporary file" in result.output

    def test_set_directory_or_undo_is_required(self):
        result = run_script([])
        assert result.exit_code != 0
        assert "exactly one of" in result.output


class TestApplyAndUndo:
    ANSWERS = "cSW,rho,mu\ny\nKL_several_m_varying_n\ny\n"

    def apply(self, set_directory):
        result = run_script(
            ["-set_dir", str(set_directory), "-apply"], answers=self.ANSWERS
        )
        assert result.exit_code == 0, result.output
        manifests = [
            p for p in os.listdir(set_directory.parent) if "standardization" in p
        ]
        assert len(manifests) == 1
        return set_directory.parent / manifests[0]

    def test_apply_renames_files_directory_and_sidecars(self, mixed_mass_set):
        contents_before = sorted(snapshot(mixed_mass_set).values())
        manifest_path = self.apply(mixed_mass_set)
        parent = mixed_mass_set.parent
        new_directory = parent / "KL_several_m_varying_n"

        assert not mixed_mass_set.exists()
        assert sorted(os.listdir(new_directory)) == sorted(
            f"KL_Brillouin_EpsMSCG1e-12_EpsCG1e-12_n{n}_{m}.{ext}"
            for n in (1, 2, 3)
            for m in ("m0p02_config0056800", "m0p04_config0051200")
            for ext in ("txt", "dat")
        )
        assert sorted(snapshot(new_directory).values()) == contents_before

        filters = json.loads((parent / "KL_several_m_varying_n.json").read_text())
        assert filters["exclude"] == [
            "KL_Brillouin_EpsMSCG1e-12_EpsCG1e-12_n1_m0p02_config0056800",
            "KL_Brillouin_EpsMSCG1e-12_EpsCG1e-12_n1_m0p02_config0056800.txt",
            "unrelated_entry",
        ]
        notes = (parent / "KL_several_m_varying_n.yaml").read_text()
        assert 'name: "KL_several_m_varying_n"   # set name' in notes
        assert "Filenames standardized: 6 file groups renamed." in notes

        manifest = yaml.safe_load(manifest_path.read_text())
        assert manifest["status"] == "applied"
        assert manifest["dropped_constants"] == [
            "Clover_coefficient",
            "Rho_value",
            "KL_scaling_factor",
        ]
        assert len(manifest["file_renames"]) == 12

    def test_second_run_finds_nothing_to_do(self, mixed_mass_set):
        self.apply(mixed_mass_set)
        new_directory = mixed_mass_set.parent / "KL_several_m_varying_n"
        result = run_script(["-set_dir", str(new_directory), "-defaults"])
        assert result.exit_code == 0, result.output
        assert "Nothing to do" in result.output

    def test_parameters_read_by_project_parsers_are_unchanged(self, mixed_mass_set):
        from library import parsing

        def merged(directory, filename):
            lines = open(os.path.join(directory, filename)).readlines()
            values = dict(parsing.extract_scalar_parameters_from_file_contents(lines))
            for key, value in parsing.extract_scalar_parameters_from_filename(
                filename
            ).items():
                values.setdefault(key, value)
            return values

        before = {
            name: merged(mixed_mass_set, name)
            for name in os.listdir(mixed_mass_set)
            if name.endswith(".txt")
        }
        manifest = yaml.safe_load(self.apply(mixed_mass_set).read_text())
        new_directory = mixed_mass_set.parent / "KL_several_m_varying_n"
        for entry in manifest["file_renames"]:
            if entry["old"].endswith(".txt"):
                assert merged(new_directory, entry["new"]) == before[entry["old"]]

    def test_undo_restores_names_contents_and_sidecars(self, mixed_mass_set):
        parent = mixed_mass_set.parent
        files_before = snapshot(mixed_mass_set)
        filters_before = json.loads((parent / "KL_demo.json").read_text())
        manifest_path = self.apply(mixed_mass_set)

        result = run_script(["-undo", str(manifest_path)], answers="y\n")
        assert result.exit_code == 0, result.output
        assert snapshot(mixed_mass_set) == files_before
        assert json.loads((parent / "KL_demo.json").read_text()) == filters_before
        assert 'name: "KL_demo"   # set name' in (parent / "KL_demo.yaml").read_text()
        assert not (parent / "KL_several_m_varying_n").exists()
        assert yaml.safe_load(manifest_path.read_text())["status"] == "undone"

    def test_undo_refuses_manifest_not_applied(self, mixed_mass_set):
        manifest_path = self.apply(mixed_mass_set)
        assert run_script(["-undo", str(manifest_path)], answers="y\n").exit_code == 0
        result = run_script(["-undo", str(manifest_path)], answers="y\n")
        assert result.exit_code == 1
        assert "not 'applied'" in result.output

    def test_declining_final_confirmation_changes_nothing(self, mixed_mass_set):
        before = snapshot(mixed_mass_set)
        answers = "cSW,rho,mu\nn\nn\n"  # drop, no dir rename, decline apply
        result = run_script(
            ["-set_dir", str(mixed_mass_set), "-apply"], answers=answers
        )
        assert result.exit_code == 0, result.output
        assert "Nothing changed" in result.output
        assert snapshot(mixed_mass_set) == before
        assert not any(
            "standardization" in p for p in os.listdir(mixed_mass_set.parent)
        )


class TestSuggestSetName:
    def test_kernel_is_not_part_of_the_suggested_name(self, specifications):
        stems = [
            f"KL_{kernel}_n{n}_config0056800"
            for kernel in ("Brillouin", "Standard")
            for n in range(1, 8)
        ]
        groups = [make_group(s, specifications) for s in stems]
        for g in groups:
            g.new_stem = g.stem
        summaries = summaries_for(groups, specifications)
        order = [
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Rational_order",
            "Configuration_label",
        ]
        assert sqf.suggest_set_name(order, summaries, groups) == "KL_varying_n"

    def test_three_exponential_values_count_as_varying(self, specifications):
        groups = [
            make_group(f"KL_Brillouin_n3_EpsCG{eps}_config{config}", specifications)
            for eps in ("1e-08", "1e-06", "1e-04")
            for config in ("0007200", "0018000")
        ]
        for g in groups:
            g.new_stem = g.stem
        summaries = summaries_for(groups, specifications)
        assert summaries["Outer_solver_epsilon"].classification == "varying"
        assert summaries["Configuration_label"].classification == "several"
        order = [
            "Overlap_operator_method",
            "Kernel_operator_type",
            "Rational_order",
            "Outer_solver_epsilon",
            "Configuration_label",
        ]
        assert (
            sqf.suggest_set_name(order, summaries, groups)
            == "KL_several_config_varying_EpsCG"
        )
