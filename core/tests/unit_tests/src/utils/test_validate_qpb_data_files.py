"""
Unit tests for the validate_qpb_data_files.py utility script.

Tests cover main program type detection and completeness checks per
program type (invert, overlap eigenvalue estimation, the other
programs), resolving the program type of a data file set, the file
remover (quarantine, permanent deletion, refusing to empty a set), and
full command-line runs on small synthetic data file sets built in a
temporary directory, including the case where no log file shows a
recognised program type: nothing may be removed then.
"""

import os

import pytest
from click.testing import CliRunner

from library import LoggingWrapper
from src.utils import validate_qpb_data_files as v


# =============================================================================
# SYNTHETIC LOG FILES
# =============================================================================

HEADER = (
    " Partition: dc\n"
    " (Lt, Lz, Ly, Lx) = (48,24,24,24)\n"
    " Gauge field (raw_32) = /p/conf_Nf0_b6p20_L24T48_apeN1a0p72.0007200\n"
    " mass = 0.1\n"
    " Dslash operator is Brillouin\n"
)


def invert_log(complete=True):
    text = HEADER + " After 12 iterations msCG converged, t = 0.2 secs\n"
    if complete:
        text += " CG done, 12 vectors in t = 10.5 sec\n"
    return text


def eigens_log(iterations=3, converged=True, summary=True):
    text = HEADER + " Eigensolver epsilon = 1.000000e-02\n"
    for i in range(1, iterations + 1):
        text += (
            f" Lanczos iter =     {i}, lambda_min_squared = 1.18e-02 "
            "(change = 9.3e-03), lambda_max_squared = 3.99e+00 "
            "(change = 2.8e-05), target = 1.000000e-02\n"
        )
    if converged:
        text += f"\n Lanczos converged after {iterations} iters, t = 56.6 sec\n"
    if summary:
        text += (
            " Extreme eigenvalues of H^2 = (g5 Dov,m)^2, with exact overlap "
            "bounds [m^2, 4 rho^2]:\n"
            " lambda_min_squared = +1.183947663499e-02   (m^2     = 1.0e-02)\n"
            " lambda_max_squared = +3.996974691769e+00   (4 rho^2 = 4.0e+00)\n"
            " Condition number = 3.375972e+02\n"
        )
    return text


def sign_squared_log(complete=True, legacy=False):
    square = "" if legacy else "^2"
    text = HEADER + f" Done vector = 1 / 1, || Sign^2(X) - 1 ||{square} = 1.2e-10\n"
    if complete:
        text += " GW Diffs (per stochastic source):\n"
    return text


def unrecognised_log():
    return HEADER + " Some future qpb program, result = 42\n"


CORRELATORS = "".join(
    f"{t} 0 0 0 +1.000000e+00 +5.000000e-01 g5-g5\n" for t in range(4)
)


def write(directory, name, text):
    path = os.path.join(directory, name)
    with open(path, "w") as file:
        file.write(text)
    return path


def files_in(directory):
    if not os.path.isdir(directory):
        return []
    return sorted(os.listdir(directory))


@pytest.fixture
def null_logger():
    return LoggingWrapper(None, "test.log", enable_logging=False)


# =============================================================================
# PROGRAM TYPE AND COMPLETENESS
# =============================================================================


class TestLogFileClassification:
    def classify(self, tmp_path, text):
        return v.classify_log_file(write(str(tmp_path), "run.txt", text))

    def test_complete_invert_log(self, tmp_path):
        status = self.classify(tmp_path, invert_log())
        assert (status.program_type, status.complete) == ("invert", True)

    def test_interrupted_invert_log_shows_no_program_type(self, tmp_path):
        status = self.classify(tmp_path, invert_log(complete=False))
        assert (status.program_type, status.complete) == (None, False)

    def test_converged_eigens_log(self, tmp_path):
        status = self.classify(tmp_path, eigens_log())
        assert status.program_type == "overlap_eigenvalue_estimation"
        assert status.complete

    def test_eigens_log_at_iteration_cap_is_complete(self, tmp_path):
        status = self.classify(tmp_path, eigens_log(iterations=999, converged=False))
        assert status.program_type == "overlap_eigenvalue_estimation"
        assert status.complete

    def test_interrupted_eigens_log_is_incomplete(self, tmp_path):
        # Per-iteration lines contain "lambda_min_squared =" as well
        status = self.classify(tmp_path, eigens_log(converged=False, summary=False))
        assert status.program_type == "overlap_eigenvalue_estimation"
        assert not status.complete

    def test_sign_squared_log_needs_final_summary(self, tmp_path):
        assert self.classify(tmp_path, sign_squared_log()).complete
        status = self.classify(tmp_path, sign_squared_log(complete=False))
        assert status.program_type == "sign_squared_violation"
        assert not status.complete

    def test_legacy_sign_squared_log_is_recognised(self, tmp_path):
        status = self.classify(tmp_path, sign_squared_log(legacy=True))
        assert (status.program_type, status.complete) == (
            "sign_squared_violation",
            True,
        )

    def test_unrecognised_log(self, tmp_path):
        status = self.classify(tmp_path, unrecognised_log())
        assert (status.program_type, status.complete) == (None, False)


def status(program_type, complete=True):
    return v.LogFileStatus("x.txt", program_type, complete)


class TestResolveSetProgramType:
    def test_single_type(self):
        statuses = [status("invert"), status(None, False)]
        assert v.resolve_set_program_type(statuses, None) == "invert"

    def test_no_recognised_type_stops(self):
        with pytest.raises(v.StopValidation, match="MAIN_PROGRAM_TYPE_MAPPING"):
            v.resolve_set_program_type([status(None, False)] * 3, None)

    def test_several_types_stop(self):
        statuses = [status("invert"), status("overlap_eigenvalue_estimation")]
        with pytest.raises(v.StopValidation, match="several main program types"):
            v.resolve_set_program_type(statuses, None)

    def test_stored_type_wins(self):
        statuses = [status("overlap_eigenvalue_estimation")]
        assert v.resolve_set_program_type(statuses, "invert") == "invert"

    def test_legacy_non_invert_is_refined(self):
        statuses = [status("sign_squared_violation"), status("invert")]
        result = v.resolve_set_program_type(statuses, "non-invert")
        assert result == "sign_squared_violation"

    def test_legacy_non_invert_kept_without_precise_type(self):
        assert (
            v.resolve_set_program_type([status("invert")], "non-invert") == "non-invert"
        )

    def test_compatibility(self):
        assert v.is_compatible("sign_squared_values", "non-invert")
        assert not v.is_compatible("invert", "non-invert")
        assert not v.is_compatible("invert", "overlap_eigenvalue_estimation")


class TestReadStoredProgramType:
    @pytest.mark.parametrize(
        "value, expected",
        [
            ("invert", "invert"),
            ("non-invert", "non-invert"),
            ("overlap_eigenvalue_estimation", "overlap_eigenvalue_estimation"),
            ("something_else", None),
        ],
    )
    def test_values(self, tmp_path, value, expected):
        path = write(
            str(tmp_path), "metadata.md", f"Main program type: {value}\nNumber: 3\n"
        )
        assert v.read_stored_program_type(path) == expected

    def test_missing_file(self, tmp_path):
        assert v.read_stored_program_type(str(tmp_path / "metadata.md")) is None


class TestDefaultQuarantineDirectory:
    def test_raw_set_without_auxiliary_goes_to_processed(self):
        result = v.default_quarantine_directory(
            "/data/raw/eigens/KL_set", "/data/raw/eigens", False, "20261120-120000"
        )
        assert (
            result
            == "/data/processed/eigens/KL_set/auxiliary/quarantine/20261120-120000"
        )

    def test_given_auxiliary_directory_is_used(self):
        result = v.default_quarantine_directory(
            "/data/raw/eigens/KL_set", "/aux", True, "20261120-120000"
        )
        assert result == "/aux/quarantine/20261120-120000"


# =============================================================================
# FILE REMOVER
# =============================================================================


class TestFileRemover:
    @pytest.fixture
    def set_directory(self, tmp_path):
        directory = tmp_path / "set"
        directory.mkdir()
        for stem in ("good", "bad"):
            write(str(directory), f"{stem}.txt", "log")
            write(str(directory), f"{stem}.dat", "correlators")
            write(str(directory), f"{stem}.err", "error")
        return directory

    def remover(self, set_directory, tmp_path, logger, use_quarantine=True):
        return v.FileRemover(
            str(set_directory), str(tmp_path / "quarantine"), use_quarantine, logger
        )

    def bad_items(self, set_directory):
        return v.removal_items_with_siblings(
            str(set_directory / "bad.txt"), "incomplete"
        )

    @pytest.fixture
    def answer_yes(self, monkeypatch):
        monkeypatch.setattr(v, "get_yes_or_no_user_response", lambda *a, **k: True)

    @pytest.fixture
    def answer_no(self, monkeypatch):
        monkeypatch.setattr(v, "get_yes_or_no_user_response", lambda *a, **k: False)

    def test_quarantine_moves_data_files_and_deletes_error_file(
        self, set_directory, tmp_path, null_logger, answer_yes
    ):
        remover = self.remover(set_directory, tmp_path, null_logger)
        assert remover.confirm_and_remove(
            self.bad_items(set_directory), "Bad", required=True
        )
        assert files_in(set_directory) == ["good.dat", "good.err", "good.txt"]
        assert files_in(tmp_path / "quarantine") == [
            "bad.dat",
            "bad.txt",
            "quarantine_log.txt",
        ]
        log = (tmp_path / "quarantine" / "quarantine_log.txt").read_text()
        assert f"{set_directory / 'bad.txt'}\tincomplete" in log
        assert "bad.err" not in log

    def test_no_quarantine_deletes_permanently(
        self, set_directory, tmp_path, null_logger, answer_yes
    ):
        remover = self.remover(
            set_directory, tmp_path, null_logger, use_quarantine=False
        )
        remover.confirm_and_remove(self.bad_items(set_directory), "Bad", required=True)
        assert files_in(set_directory) == ["good.dat", "good.err", "good.txt"]
        assert not (tmp_path / "quarantine").exists()

    def test_removal_that_empties_the_set_is_refused(
        self, set_directory, tmp_path, null_logger, answer_yes
    ):
        items = self.bad_items(set_directory) + v.removal_items_with_siblings(
            str(set_directory / "good.txt"), "incomplete"
        )
        remover = self.remover(set_directory, tmp_path, null_logger)
        with pytest.raises(v.StopValidation, match="without any .txt or .dat"):
            remover.confirm_and_remove(items, "Everything", required=True)
        assert len(files_in(set_directory)) == 6

    def test_declining_required_removal_stops(
        self, set_directory, tmp_path, null_logger, answer_no
    ):
        remover = self.remover(set_directory, tmp_path, null_logger)
        with pytest.raises(v.StopValidation, match="cannot proceed"):
            remover.confirm_and_remove(
                self.bad_items(set_directory), "Bad", required=True
            )
        assert len(files_in(set_directory)) == 6

    def test_declining_optional_removal_keeps_files(
        self, set_directory, tmp_path, null_logger, answer_no
    ):
        remover = self.remover(set_directory, tmp_path, null_logger)
        assert not remover.confirm_and_remove(
            self.bad_items(set_directory), "Bad", required=False
        )
        assert len(files_in(set_directory)) == 6

    def test_name_clash_in_quarantine_gets_suffix(
        self, set_directory, tmp_path, null_logger, answer_yes
    ):
        (tmp_path / "quarantine").mkdir()
        write(str(tmp_path / "quarantine"), "bad.txt", "older")
        remover = self.remover(set_directory, tmp_path, null_logger)
        remover.confirm_and_remove(
            [(str(set_directory / "bad.txt"), "x")], "Bad", required=True
        )
        assert (tmp_path / "quarantine" / "bad.txt").read_text() == "older"
        assert (tmp_path / "quarantine" / "bad.txt.1").read_text() == "log"


# =============================================================================
# COMMAND LINE RUNS
# =============================================================================


class TestCommandLine:
    @pytest.fixture
    def paths(self, tmp_path):
        set_directory = tmp_path / "raw" / "eigens" / "KL_set"
        set_directory.mkdir(parents=True)
        return {
            "set": set_directory,
            "auxiliary": tmp_path / "auxiliary",
            "quarantine": tmp_path / "quarantine",
        }

    def run(self, paths, answers="", extra=()):
        arguments = [
            "-raw_dir",
            str(paths["set"]),
            "-ax_files_dir",
            str(paths["auxiliary"]),
            "-quarantine_dir",
            str(paths["quarantine"]),
            *extra,
        ]
        return CliRunner().invoke(v.main, arguments, input=answers)

    def metadata(self, paths):
        return (paths["auxiliary"] / "metadata" / "metadata.md").read_text()

    def test_eigens_set_is_kept_including_runs_at_iteration_cap(self, paths):
        directory = str(paths["set"])
        write(directory, "run_eps1e-2.txt", eigens_log())
        write(directory, "run_eps1e-4.txt", eigens_log(iterations=200))
        write(directory, "run_eps1e-8.txt", eigens_log(iterations=999, converged=False))
        write(directory, "run_eps1e-2.err", "warning")

        result = self.run(paths, answers="y\n")  # Delete the .err files
        assert result.exit_code == 0, result.output
        assert files_in(paths["set"]) == [
            "run_eps1e-2.txt",
            "run_eps1e-4.txt",
            "run_eps1e-8.txt",
        ]
        assert not paths["quarantine"].exists()
        assert "Main program type: overlap_eigenvalue_estimation" in self.metadata(
            paths
        )
        stored = paths["auxiliary"] / "metadata" / "list_of_stored_qpb_log_files.txt"
        assert len(stored.read_text().splitlines()) == 3

    def test_unrecognised_program_removes_nothing(self, paths):
        """The case that once deleted a whole eigens data file set."""
        directory = str(paths["set"])
        write(directory, "run_a.txt", unrecognised_log())
        write(directory, "run_b.txt", unrecognised_log())
        write(directory, "run_a.err", "warning")

        result = self.run(paths, answers="n\n")  # Keep the .err files
        assert result.exit_code == 1
        assert "MAIN_PROGRAM_TYPE_MAPPING" in result.output
        assert files_in(paths["set"]) == ["run_a.err", "run_a.txt", "run_b.txt"]
        assert not paths["quarantine"].exists()

    def make_invert_set(self, paths, truncated=True):
        directory = str(paths["set"])
        for stem in ("n1", "n2", "n3"):
            write(directory, f"{stem}.txt", invert_log())
            write(directory, f"{stem}.dat", CORRELATORS)
        if truncated:
            write(directory, "n4.txt", invert_log(complete=False))
            write(directory, "n4.dat", CORRELATORS)
            write(directory, "n4.err", "killed by SLURM")
            write(directory, "n1.err", "warning")

    def test_incomplete_invert_run_is_quarantined_without_its_err_file(self, paths):
        self.make_invert_set(paths)
        # Keep .err files; remove the incomplete log file
        result = self.run(paths, answers="n\ny\n")
        assert result.exit_code == 0, result.output
        assert files_in(paths["set"]) == [
            "n1.dat",
            "n1.err",
            "n1.txt",
            "n2.dat",
            "n2.txt",
            "n3.dat",
            "n3.txt",
        ]
        assert files_in(paths["quarantine"]) == [
            "n4.dat",
            "n4.txt",
            "quarantine_log.txt",
        ]
        log = (paths["quarantine"] / "quarantine_log.txt").read_text()
        assert "n4.txt\tincomplete: no 'CG done' or 'BiCGStab done'" in log
        assert "Main program type: invert" in self.metadata(paths)
        assert "Number of qpb correlator files: 3" in self.metadata(paths)

    def test_no_quarantine_option_deletes_permanently(self, paths):
        self.make_invert_set(paths)
        result = self.run(paths, answers="n\ny\n", extra=["-no_quarantine"])
        assert result.exit_code == 0, result.output
        assert "n4.txt" not in files_in(paths["set"])
        assert "n4.dat" not in files_in(paths["set"])
        assert not paths["quarantine"].exists()

    def test_declining_incomplete_removal_stops_and_keeps_files(self, paths):
        self.make_invert_set(paths)
        before = files_in(paths["set"])
        result = self.run(paths, answers="n\nn\n")
        assert result.exit_code == 1
        assert files_in(paths["set"]) == before

    def test_set_of_only_incomplete_runs_is_not_emptied(self, paths):
        directory = str(paths["set"])
        write(directory, "run_a.txt", eigens_log(converged=False, summary=False))
        write(directory, "run_b.txt", eigens_log(converged=False, summary=False))
        result = self.run(paths, answers="y\n")
        assert result.exit_code == 1
        assert "without any .txt or .dat file" in result.output
        assert files_in(paths["set"]) == ["run_a.txt", "run_b.txt"]

    def test_mixed_program_types_without_established_type_stop(self, paths):
        self.make_invert_set(paths, truncated=False)
        write(str(paths["set"]), "stray.txt", sign_squared_log())
        before = files_in(paths["set"])
        result = self.run(paths)
        assert result.exit_code == 1
        assert "several main program types" in result.output
        assert files_in(paths["set"]) == before

    def test_new_log_file_of_another_program_is_quarantined(self, paths):
        self.make_invert_set(paths, truncated=False)
        assert self.run(paths).exit_code == 0  # Establishes "invert"
        write(str(paths["set"]), "stray.txt", sign_squared_log())
        result = self.run(paths, answers="y\n")
        assert result.exit_code == 0, result.output
        assert "stray.txt" not in files_in(paths["set"])
        log = (paths["quarantine"] / "quarantine_log.txt").read_text()
        assert "'sign_squared_violation' program, set is 'invert'" in log

    def test_unsupported_file_is_quarantined(self, paths):
        write(str(paths["set"]), "run.txt", eigens_log())
        write(str(paths["set"]), "notes.md", "my notes")
        result = self.run(paths, answers="y\n")
        assert result.exit_code == 0, result.output
        assert files_in(paths["set"]) == ["run.txt"]
        assert "notes.md" in files_in(paths["quarantine"])

    def test_legacy_non_invert_metadata_is_refined(self, paths):
        write(str(paths["set"]), "run.txt", sign_squared_log(legacy=True))
        metadata_directory = paths["auxiliary"] / "metadata"
        metadata_directory.mkdir(parents=True)
        write(str(metadata_directory), "metadata.md", "Main program type: non-invert\n")
        result = self.run(paths)
        assert result.exit_code == 0, result.output
        assert "Main program type: sign_squared_violation" in self.metadata(paths)

    def test_second_run_without_new_files_can_stop_cleanly(self, paths):
        write(str(paths["set"]), "run.txt", eigens_log())
        assert self.run(paths).exit_code == 0
        result = self.run(paths, answers="n\n")  # Do not repeat validation
        assert result.exit_code == 0
        assert "no new files" in result.output
