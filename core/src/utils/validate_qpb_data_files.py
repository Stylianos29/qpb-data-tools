"""
validate_qpb_data_files.py

Summary:
    A validation script for qpb data files that checks file integrity,
    ensures consistency between file types, and keeps a single main
    program type within a data file set directory.

Input:
    Required:
        - raw_data_files_set_directory_path: Path to directory
          containing qpb data files (.txt log files, .dat correlator
          files, .err error files)
    Optional:
        - enable_logging: Flag to enable detailed logging of validation
          process
        - auxiliary_files_directory: Directory for storing metadata and
          log files
        - metadata_directory, logs_directory: Override the default
          subdirectories of the auxiliary files directory
        - log_filename: Custom name for the log file
        - no_quarantine: Delete rejected files permanently instead of
          moving them to the quarantine directory
        - quarantine_directory: Where rejected files are moved (default:
          "<auxiliary files directory>/quarantine/<timestamp>")

Output:
    1. Validated data file set directory
    2. metadata.md file containing:
        - Main program type (e.g. "invert", "sign_squared_violation",
          "overlap_eigenvalue_estimation")
        - File counts by type
        - Validation timestamp
    3. List files of validated file paths:
        - list_of_stored_qpb_log_files.txt
        - list_of_stored_qpb_error_files.txt
        - list_of_stored_qpb_correlators_files.txt (invert only)
    4. Quarantine directory, only if files were rejected: the rejected
       files and a "quarantine_log.txt" with their original paths and
       the reason for rejecting them

Main program type:
    Detected from the log file contents with the same key phrases the
    parser uses (MAIN_PROGRAM_TYPE_MAPPING). A log file is complete only
    if it also contains its program type's completion marker:
        - invert: "CG done" or "BiCGStab done"
        - overlap_eigenvalue_estimation ("eigens" programs): the final
          summary block ("Extreme eigenvalues of ..." and "Condition
          number ="), which is printed also by runs that reached the
          iteration cap without converging
        - all other programs: "per stochastic source"

Safety rules:
    1. Log files without any recognised program output are never
       removed on their own: if no log file of the set has a recognised
       program type, the script stops without touching anything.
    2. Rejected files are moved to the quarantine directory (default) or
       deleted permanently with "--no_quarantine". Error files (.err)
       are always deleted, never quarantined.
    3. Every removal lists the files and asks first. A removal that
       would leave no .txt or .dat file in the set is refused.

Functionality:
    1. Rejects files with unsupported extensions
    2. Rejects correlator files with null values (and their log files)
    3. Rejects empty files (.txt and .dat)
    4. Optionally deletes all error files
    5. Determines the main program type of the set and rejects log
       files of a different program type
    6. Rejects incomplete log files (interrupted runs)
    7. For invert sets: rejects all-zero correlator files and handles
       log/correlator files without a matching partner
    8. Caches validated file paths for future reference

Usage:
    python validate_qpb_data_files.py -raw_dir /path/to/data/files/set
                                     [-log_on] [-ax_files_dir
                                     /path/to/auxiliary/files]
                                     [-log_name custom_log_name]
                                     [-no_quarantine]
                                     [-quarantine_dir /path/to/quarantine]

Note:
    The script requires user interaction for critical decisions about
    file removal and handles error conditions by either exiting or
    allowing user intervention.
"""

import os
import sys
import glob
import shutil
import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Set, Tuple

import click

# TODO: Investigate whether to use library.utils.logging_utilities
from library import LoggingWrapper
import library.constants as constants

# The parser's own detection, so that validation and parsing always agree
from library.parsing.file_contents_parser import _extract_main_program_type
from library.validation.click_validators import (
    directory,
    validate_log_filename,
)


# =============================================================================
# CONSTANTS
# =============================================================================

LOG_EXTENSION = ".txt"
ERROR_EXTENSION = ".err"
CORRELATORS_EXTENSION = ".dat"
SUPPORTED_EXTENSIONS = (LOG_EXTENSION, ERROR_EXTENSION, CORRELATORS_EXTENSION)
DATA_EXTENSIONS = (LOG_EXTENSION, CORRELATORS_EXTENSION)

INVERT_PROGRAM_TYPE = "invert"
# Value written by earlier versions of this script for every program type
# other than "invert"; still accepted when read from metadata.md
LEGACY_NON_INVERT_PROGRAM_TYPE = "non-invert"
KNOWN_PROGRAM_TYPES = set(constants.MAIN_PROGRAM_TYPE_MAPPING.values())

# A log file is complete if it contains every phrase of at least one of
# the groups listed for its program type (whitespace is ignored)
COMPLETION_MARKERS: Dict[str, Tuple[Tuple[str, ...], ...]] = {
    INVERT_PROGRAM_TYPE: (("CG done",), ("BiCGStab done",)),
    # Final summary block, printed also by runs that reached the iteration
    # cap. (The per-iteration "Lanczos iter = ..." lines also contain
    # "lambda_min_squared =", so that phrase cannot mark completion.)
    "overlap_eigenvalue_estimation": (
        ("Extreme eigenvalues of", "Condition number ="),
    ),
}
DEFAULT_COMPLETION_MARKERS: Tuple[Tuple[str, ...], ...] = (("per stochastic source",),)

NULL_CORRELATOR_VALUE = "+0.000000e+00"
QUARANTINE_LOG_FILENAME = "quarantine_log.txt"
METADATA_FILENAME = "metadata.md"
STORED_LOG_FILES_LIST = "list_of_stored_qpb_log_files.txt"
STORED_ERROR_FILES_LIST = "list_of_stored_qpb_error_files.txt"
STORED_CORRELATORS_FILES_LIST = "list_of_stored_qpb_correlators_files.txt"
MAXIMUM_FILES_SHOWN = 10


class StopValidation(Exception):
    """Ends the validation; exit code 1 for failures, 0 for a clean stop."""

    def __init__(self, message: str, exit_code: int = 1):
        super().__init__(message)
        self.exit_code = exit_code


# =============================================================================
# HELPERS
# =============================================================================


def get_yes_or_no_user_response(prompt_text, logger=None):
    """Helper function to handle yes/no prompts with validation"""
    response = click.prompt(prompt_text, type=str)
    while response.lower() not in ["y", "yes", "", "n", "no"]:
        response = click.prompt(
            "Invalid input. Please enter 'y' for yes or 'n' for no (y[Y]/n[N])",
            type=str,
        )
    return response.lower() in ["y", "yes", ""]


def list_files(directory_path: str, extension: Optional[str] = None) -> List[str]:
    """Sorted full paths of the files (not subdirectories) in a directory."""
    pattern = "*" + (extension or "")
    return sorted(
        path
        for path in glob.glob(os.path.join(directory_path, pattern))
        if os.path.isfile(path)
    )


def sibling_path(path: str, extension: str) -> str:
    return os.path.splitext(path)[0] + extension


def read_text(path: str) -> str:
    with open(path, "r", errors="replace") as file:
        return file.read()


def default_quarantine_directory(
    raw_data_files_set_directory_path: str,
    auxiliary_files_directory: str,
    auxiliary_directory_given: bool,
    timestamp: str,
) -> str:
    """
    "<auxiliary>/quarantine/<timestamp>". Without an explicit auxiliary
    directory, a set under ".../raw/<program>/<set>" uses
    ".../processed/<program>/<set>/auxiliary", keeping quarantined files
    out of the raw data files tree.
    """
    base = auxiliary_files_directory
    marker = f"{os.sep}raw{os.sep}"
    set_path = raw_data_files_set_directory_path.rstrip(os.sep)
    if not auxiliary_directory_given and marker in set_path:
        base = os.path.join(
            set_path.replace(marker, f"{os.sep}processed{os.sep}", 1), "auxiliary"
        )
    return os.path.join(base, "quarantine", timestamp)


# =============================================================================
# PROGRAM TYPE AND COMPLETENESS
# =============================================================================


@dataclass
class LogFileStatus:
    path: str
    program_type: Optional[str]  # None: no recognised program output
    complete: bool


def _normalized(text: str) -> str:
    return "".join(text.split())


def detect_program_type(text: str) -> Optional[str]:
    """Main program type of a log file, as the parser detects it."""
    return _extract_main_program_type(text.splitlines())


def completion_markers(program_type: Optional[str]) -> Tuple[Tuple[str, ...], ...]:
    if program_type is None:
        return DEFAULT_COMPLETION_MARKERS
    return COMPLETION_MARKERS.get(program_type, DEFAULT_COMPLETION_MARKERS)


def is_complete(program_type: Optional[str], text: str) -> bool:
    """Whether the log contains all phrases of one of the marker groups."""
    normalized_text = _normalized(text)
    return any(
        all(_normalized(marker) in normalized_text for marker in group)
        for group in completion_markers(program_type)
    )


def describe_completion_markers(program_type: Optional[str]) -> str:
    return " or ".join(
        " + ".join(f"'{marker}'" for marker in group)
        for group in completion_markers(program_type)
    )


def classify_log_file(path: str) -> LogFileStatus:
    try:
        text = read_text(path)
    except OSError:
        return LogFileStatus(path, None, False)
    program_type = detect_program_type(text)
    complete = program_type is not None and is_complete(program_type, text)
    return LogFileStatus(path, program_type, complete)


def is_compatible(file_program_type: str, set_program_type: str) -> bool:
    if set_program_type == LEGACY_NON_INVERT_PROGRAM_TYPE:
        return file_program_type != INVERT_PROGRAM_TYPE
    return file_program_type == set_program_type


def resolve_set_program_type(
    statuses: Sequence[LogFileStatus], stored_program_type: Optional[str]
) -> str:
    """
    The main program type of the set: the one stored in metadata.md (a
    legacy "non-invert" is refined when the files show a single precise
    type), otherwise the single type found among the log files. Stops if
    the log files show several types, or none at all.
    """
    found = sorted({s.program_type for s in statuses if s.program_type})

    if stored_program_type and stored_program_type != LEGACY_NON_INVERT_PROGRAM_TYPE:
        return stored_program_type

    if stored_program_type == LEGACY_NON_INVERT_PROGRAM_TYPE:
        found = [t for t in found if t != INVERT_PROGRAM_TYPE]
        if not found:
            return LEGACY_NON_INVERT_PROGRAM_TYPE

    if not found:
        raise StopValidation(
            f"None of the {len(statuses)} log file(s) to validate contains "
            "output of a recognised qpb main program, so nothing can be "
            "checked and nothing was removed. If this is a new qpb program, "
            "add its key phrase to MAIN_PROGRAM_TYPE_MAPPING in "
            "library/constants/patterns.py."
        )
    if len(found) > 1:
        counts = ", ".join(
            f"{t}: {sum(s.program_type == t for s in statuses)}" for t in found
        )
        raise StopValidation(
            f"Log files of several main program types found ({counts}). "
            "These cannot coexist in one data file set. Please manually "
            "review and move the files of the other program type(s)."
        )
    return found[0]


def read_stored_program_type(metadata_file: str) -> Optional[str]:
    if not os.path.exists(metadata_file) or os.path.getsize(metadata_file) == 0:
        return None
    with open(metadata_file, "r") as file:
        for line in file:
            if line.startswith("Main program type:"):
                value = line.split("Main program type:", 1)[1].strip()
                if (
                    value in KNOWN_PROGRAM_TYPES
                    or value == LEGACY_NON_INVERT_PROGRAM_TYPE
                ):
                    return value
                return None
    return None


# =============================================================================
# CORRELATOR FILE CHECKS
# =============================================================================


def has_null_value(correlators_file_path: str) -> bool:
    """Any line with a null value in the 5th column."""
    with open(correlators_file_path, "r") as file:
        return any(
            line.split()[4] == NULL_CORRELATOR_VALUE
            for line in file
            if len(line.split()) >= 5
        )


def has_only_zero_values(correlators_file_path: str) -> bool:
    """Every non-empty line has null values in the 5th and 6th columns."""
    with open(correlators_file_path, "r") as file:
        return all(
            line.split()[4:6] == [NULL_CORRELATOR_VALUE, NULL_CORRELATOR_VALUE]
            for line in file
            if line.strip()
        )


# =============================================================================
# FILE REMOVAL (QUARANTINE OR DELETE)
# =============================================================================


class FileRemover:
    """
    Removes rejected files after listing them and asking the user. Files
    are moved to the quarantine directory unless quarantining is off;
    error files are always deleted. A removal that would leave the set
    without any .txt or .dat file is refused.
    """

    def __init__(
        self,
        set_directory: str,
        quarantine_directory: str,
        use_quarantine: bool,
        logger: LoggingWrapper,
    ):
        self.set_directory = set_directory
        self.quarantine_directory = quarantine_directory
        self.use_quarantine = use_quarantine
        self.logger = logger

    @property
    def action(self) -> str:
        return "Move to quarantine" if self.use_quarantine else "Permanently delete"

    def confirm_and_remove(
        self,
        items: Sequence[Tuple[str, str]],
        description: str,
        required: bool,
    ) -> bool:
        """
        Remove (path, reason) items after confirmation. If declined: stops
        the validation when "required", otherwise keeps the files.
        Returns whether the files were removed.
        """
        # Drop duplicates and files that no longer exist, keeping order
        unique: Dict[str, str] = {}
        for path, reason in items:
            if os.path.exists(path) and path not in unique:
                unique[path] = reason
        if not unique:
            return False

        self._refuse_if_set_would_be_emptied(set(unique))
        self._show(unique, description)

        consequence = (
            "Selecting 'n' will exit the program"
            if required
            else "Selecting 'n' keeps them"
        )
        error_note = (
            " (.err files are always deleted)"
            if self.use_quarantine and any(p.endswith(ERROR_EXTENSION) for p in unique)
            else ""
        )
        if not get_yes_or_no_user_response(
            f"{self.action} these {len(unique)} file(s){error_note}?\n"
            f"{consequence} (y[Y]/n[N])"
        ):
            if required:
                raise StopValidation(
                    f"Validation cannot proceed with {description.lower()} present "
                    "inside the qpb data file set directory. Exiting the program."
                )
            self.logger.info(
                f"Kept {len(unique)} file(s): {description}.", to_console=True
            )
            return False

        for path, reason in unique.items():
            self._remove(path, reason)
        return True

    def delete_error_files(self, paths: Sequence[str]) -> int:
        """Delete error files without further confirmation."""
        deleted = 0
        for path in paths:
            try:
                os.remove(path)
                deleted += 1
                self.logger.info(f"Deleted .err file: {os.path.basename(path)}")
            except OSError as error:
                self.logger.error(
                    f"Error deleting file {path}: {error}", to_console=True
                )
        return deleted

    def _refuse_if_set_would_be_emptied(self, removing: Set[str]) -> None:
        remaining = [
            path
            for path in list_files(self.set_directory)
            if path.endswith(DATA_EXTENSIONS) and path not in removing
        ]
        if not remaining:
            raise StopValidation(
                f"Removing these {len(removing)} file(s) would leave the data file "
                "set without any .txt or .dat file. Nothing was removed; please "
                "inspect the data file set by hand."
            )

    def _show(self, unique: Dict[str, str], description: str) -> None:
        self.logger.warning(f"{description}: {len(unique)} file(s).", to_console=True)
        for index, (path, reason) in enumerate(unique.items()):
            line = f"{os.path.basename(path)}  [{reason}]"
            self.logger.info(f"To be removed: {line}")
            if index < MAXIMUM_FILES_SHOWN:
                click.echo(f"     {line}")
        if len(unique) > MAXIMUM_FILES_SHOWN:
            click.echo(
                f"     ... and {len(unique) - MAXIMUM_FILES_SHOWN} more "
                "(all are listed in the log file, if logging is enabled)"
            )

    def _remove(self, path: str, reason: str) -> None:
        name = os.path.basename(path)
        try:
            if path.endswith(ERROR_EXTENSION) or not self.use_quarantine:
                os.remove(path)
                self.logger.info(f"Deleted: {name} [{reason}]")
            else:
                target = self._quarantine_target(name)
                shutil.move(path, target)
                with open(
                    os.path.join(self.quarantine_directory, QUARANTINE_LOG_FILENAME),
                    "a",
                ) as log:
                    log.write(f"{path}\t{reason}\n")
                self.logger.info(f"Moved to quarantine: {name} [{reason}]")
        except OSError as error:
            self.logger.error(f"Error removing file {path}: {error}", to_console=True)

    def _quarantine_target(self, name: str) -> str:
        os.makedirs(self.quarantine_directory, exist_ok=True)
        target = os.path.join(self.quarantine_directory, name)
        counter = 1
        while os.path.exists(target):
            target = os.path.join(self.quarantine_directory, f"{name}.{counter}")
            counter += 1
        return target


# =============================================================================
# VALIDATION STEPS
# =============================================================================


def check_set_has_log_files(set_directory: str) -> None:
    if not os.listdir(set_directory):
        raise StopValidation(
            f"Data file set directory '{set_directory}' is empty. Validation "
            "cannot proceed without any files to validate."
        )
    if not list_files(set_directory, LOG_EXTENSION):
        raise StopValidation(
            f"No qpb log files (*.txt) found in '{set_directory}'. Validation "
            "cannot proceed without qpb log files."
        )


def reject_unsupported_files(set_directory: str, remover: FileRemover, logger) -> None:
    all_files = list_files(set_directory)
    logger.info(
        f"Total number of files in raw data file set directory: {len(all_files)}"
    )
    subdirectories = [
        entry
        for entry in os.listdir(set_directory)
        if os.path.isdir(os.path.join(set_directory, entry))
    ]
    if subdirectories:
        logger.warning(f"Subdirectories are ignored: {subdirectories}", to_console=True)

    unsupported = [p for p in all_files if not p.endswith(SUPPORTED_EXTENSIONS)]
    if not unsupported:
        logger.info("No unsupported file types found in raw data file set directory.")
        return
    remover.confirm_and_remove(
        [(p, "unsupported extension") for p in unsupported],
        "Files with unsupported extensions",
        required=True,
    )


def reject_null_correlator_files(
    set_directory: str, remover: FileRemover, logger
) -> None:
    null_files = []
    for path in list_files(set_directory, CORRELATORS_EXTENSION):
        try:
            if has_null_value(path):
                null_files.append(path)
                logger.info(f"Null correlator file found: {os.path.basename(path)}")
        except OSError as error:
            logger.error(
                f"Error reading correlator file {path}: {error}", to_console=True
            )
    if not null_files:
        logger.info("No correlator files with null values found.")
        return
    items = []
    for path in null_files:
        items.append((path, "null values in the 5th column"))
        items.append(
            (sibling_path(path, LOG_EXTENSION), "log file of a null correlator file")
        )
    remover.confirm_and_remove(
        items, "Correlator files with null values (with their log files)", required=True
    )


def reject_empty_files(set_directory: str, remover: FileRemover, logger) -> None:
    empty = [
        path
        for path in list_files(set_directory)
        if path.endswith(DATA_EXTENSIONS) and os.path.getsize(path) == 0
    ]
    if not empty:
        logger.info("No empty files found in raw data file set directory.")
        return
    remover.confirm_and_remove(
        [(p, "empty file") for p in empty], "Empty .txt or .dat files", required=True
    )


def load_stored_file_lists(metadata_directory: str, logger) -> Dict[str, List[str]]:
    stored: Dict[str, List[str]] = {}
    for filename in (
        STORED_LOG_FILES_LIST,
        STORED_ERROR_FILES_LIST,
        STORED_CORRELATORS_FILES_LIST,
    ):
        path = os.path.join(metadata_directory, filename)
        stored[filename] = []
        try:
            if not os.path.exists(path):
                with open(path, "w"):
                    logger.info(f"Created empty file: {filename}", to_console=True)
            elif os.path.getsize(path) == 0:
                logger.info(f"File: {filename} exists but it is empty.")
            else:
                with open(path, "r") as file:
                    stored[filename] = [line.strip() for line in file if line.strip()]
                logger.info(f"Contents of file: {filename} passed to a list.")
        except OSError as error:
            logger.error(f"Error processing {filename}: {error}", to_console=True)
    return stored


def select_files_to_validate(
    set_directory: str, stored: Dict[str, List[str]], logger
) -> Tuple[List[str], List[str], List[str]]:
    """(log, error, correlator) files not validated before, or all of them."""
    current = {
        STORED_LOG_FILES_LIST: list_files(set_directory, LOG_EXTENSION),
        STORED_ERROR_FILES_LIST: list_files(set_directory, ERROR_EXTENSION),
        STORED_CORRELATORS_FILES_LIST: list_files(set_directory, CORRELATORS_EXTENSION),
    }
    new = {
        name: [p for p in paths if p not in set(stored[name])]
        for name, paths in current.items()
    }
    labels = {
        STORED_LOG_FILES_LIST: "log",
        STORED_ERROR_FILES_LIST: "error",
        STORED_CORRELATORS_FILES_LIST: "correlator",
    }
    for name, paths in new.items():
        if paths:
            logger.info(
                f"Found {len(paths)} {labels[name]} files not listed.", to_console=True
            )

    if not (new[STORED_LOG_FILES_LIST] or new[STORED_CORRELATORS_FILES_LIST]):
        logger.warning("No new files found to validate.", to_console=True)
        if not get_yes_or_no_user_response(
            "Would you like to repeat validation of already existing qpb data files?\n"
            "Selecting 'n' will exit the program (y[Y]/n[N])"
        ):
            raise StopValidation(
                "Exiting validation as no new files were found.", exit_code=0
            )
        logger.info(
            "Repeating validation of already existing data files.", to_console=True
        )
        new = current

    return (
        new[STORED_LOG_FILES_LIST],
        new[STORED_ERROR_FILES_LIST],
        new[STORED_CORRELATORS_FILES_LIST],
    )


def offer_error_files_deletion(
    set_directory: str, new_error_files: Sequence[str], remover: FileRemover, logger
) -> None:
    all_error_files = list_files(set_directory, ERROR_EXTENSION)
    if not all_error_files:
        return
    total = len(all_error_files)
    found = (
        f"Found {len(new_error_files)} new .err error files ({total} total in data file set).\n"
        if new_error_files
        else f"Found {total} .err error files in data file set.\n"
    )
    if get_yes_or_no_user_response(
        found + "Error files are diagnostic outputs and not required for validation.\n"
        "Do you want to delete ALL .err files? (y[Y]/n[N])"
    ):
        deleted = remover.delete_error_files(all_error_files)
        logger.info(
            f"A total of {deleted} qpb error files were deleted.", to_console=True
        )
    else:
        logger.info(f"Keeping all {total} qpb error files.")


def removal_items_with_siblings(path: str, reason: str) -> List[Tuple[str, str]]:
    """A log file and its .err and .dat files."""
    return [
        (path, reason),
        (sibling_path(path, ERROR_EXTENSION), "error file of a rejected log file"),
        (
            sibling_path(path, CORRELATORS_EXTENSION),
            "correlator file of a rejected log file",
        ),
    ]


def validate_log_files(
    log_files: Sequence[str], metadata_file: str, remover: FileRemover, logger
) -> Tuple[str, List[str]]:
    """
    Determine the set's main program type, reject log files of other
    program types and incomplete ones. Returns the program type and the
    remaining log files.
    """
    statuses = [classify_log_file(path) for path in log_files]
    stored_program_type = read_stored_program_type(metadata_file)
    if stored_program_type:
        logger.info(
            f"Main program type found in metadata file: '{stored_program_type}'."
        )

    program_type = resolve_set_program_type(statuses, stored_program_type)
    logger.info(
        f"Main program type of the data file set: '{program_type}'.", to_console=True
    )

    incompatible = [
        s
        for s in statuses
        if s.program_type is not None
        and not is_compatible(s.program_type, program_type)
    ]
    if incompatible:
        items = []
        for status in incompatible:
            items += removal_items_with_siblings(
                status.path, f"'{status.program_type}' program, set is '{program_type}'"
            )
        remover.confirm_and_remove(
            items, "Log files of a different main program type", required=True
        )

    # Files of the set's program type are checked against its completion
    # markers; files without recognised program output are incomplete too
    # (typically runs interrupted before printing any result)
    incomplete = [s for s in statuses if s not in incompatible and not s.complete]
    if incomplete:
        reason = f"incomplete: no {describe_completion_markers(program_type)}"
        items = []
        for status in incomplete:
            items += removal_items_with_siblings(status.path, reason)
        remover.confirm_and_remove(
            items,
            "Incomplete or corrupted log files (with their .err/.dat files)",
            required=True,
        )
    else:
        logger.info("No incomplete or corrupted qpb log files found.")

    rejected = {s.path for s in incompatible} | {s.path for s in incomplete}
    return program_type, [p for p in log_files if p not in rejected]


def validate_invert_correlator_files(
    log_files: Sequence[str],
    correlator_files: Sequence[str],
    remover: FileRemover,
    logger,
) -> None:
    correlator_files = [p for p in correlator_files if os.path.exists(p)]

    # Correlator files with only zero values
    zero_files = []
    for path in correlator_files:
        try:
            if has_only_zero_values(path):
                zero_files.append(path)
                logger.info(
                    f"File contains only zeros in columns 5-6: {os.path.basename(path)}"
                )
        except OSError as error:
            logger.error(
                f"Error checking zeros in file {path}: {error}", to_console=True
            )
    if zero_files:
        items = []
        for path in zero_files:
            items.append((path, "only zero correlator values"))
            items.append(
                (
                    sibling_path(path, LOG_EXTENSION),
                    "log file of a zero-value correlator file",
                )
            )
        if not remover.confirm_and_remove(
            items,
            "Correlator files with all zero values (with their log files)",
            required=False,
        ):
            logger.info(
                "No correlator files with all zero values were removed.",
                to_console=True,
            )

    # Log and correlator files without a matching partner
    log_files = [p for p in log_files if os.path.exists(p)]
    correlator_files = [p for p in correlator_files if os.path.exists(p)]
    log_stems = {os.path.splitext(p)[0] for p in log_files}
    correlator_stems = {os.path.splitext(p)[0] for p in correlator_files}
    logs_only = log_stems - correlator_stems
    correlators_only = correlator_stems - log_stems

    if logs_only:
        if not correlator_files and len(logs_only) == len(log_files):
            logger.info(
                "Detected intentional invert no-correlators data file set "
                f"({len(log_files)} log files, 0 correlator files). "
                "Skipping correlator matching validation.",
                to_console=True,
            )
        else:
            remover.confirm_and_remove(
                [
                    (stem + LOG_EXTENSION, "no matching correlator file")
                    for stem in sorted(logs_only)
                ],
                "qpb log files without matching correlator files",
                required=False,
            )
    else:
        logger.info("No unmatched qpb log files found.")

    if correlators_only:
        remover.confirm_and_remove(
            [
                (stem + CORRELATORS_EXTENSION, "no matching log file")
                for stem in sorted(correlators_only)
            ],
            "qpb correlator files without matching log files",
            required=False,
        )
    else:
        logger.info("No unmatched qpb correlator files found.")


def store_validated_file_lists(
    set_directory: str, metadata_directory: str, program_type: str, logger
) -> None:
    log_files = list_files(set_directory, LOG_EXTENSION)
    error_files = list_files(set_directory, ERROR_EXTENSION)
    correlator_files = list_files(set_directory, CORRELATORS_EXTENSION)
    is_invert = program_type == INVERT_PROGRAM_TYPE

    if is_invert and correlator_files and len(log_files) != len(correlator_files):
        logger.warning(
            f"Number of log files ({len(log_files)}) does not match number of "
            f"correlator files ({len(correlator_files)}). "
            "This may indicate missing or extra files.",
            to_console=True,
        )

    outputs = {STORED_LOG_FILES_LIST: log_files, STORED_ERROR_FILES_LIST: error_files}
    if is_invert:
        outputs[STORED_CORRELATORS_FILES_LIST] = correlator_files
    for filename, paths in outputs.items():
        output_path = os.path.join(metadata_directory, filename)
        try:
            existed = os.path.exists(output_path)
            with open(output_path, "w") as file:
                file.writelines(f"{path}\n" for path in paths)
            logger.info(f"{'Updated' if existed else 'Created'} file list: {filename}")
        except OSError as error:
            logger.error(f"Error writing to {filename}: {error}", to_console=True)

    with open(os.path.join(metadata_directory, METADATA_FILENAME), "w") as file:
        file.write(f"Main program type: {program_type}")
        file.write(f"\nNumber of qpb log files: {len(log_files)}")
        file.write(f"\nNumber of qpb error files: {len(error_files)}")
        if is_invert:
            file.write(f"\nNumber of qpb correlator files: {len(correlator_files)}")
        file.write("\n")
        file.write(
            "\nValidation completed at: "
            f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )


def run_validation(
    set_directory: str, metadata_directory: str, remover: FileRemover, logger
) -> None:
    check_set_has_log_files(set_directory)
    reject_unsupported_files(set_directory, remover, logger)
    reject_null_correlator_files(set_directory, remover, logger)
    reject_empty_files(set_directory, remover, logger)

    stored = load_stored_file_lists(metadata_directory, logger)
    log_files, error_files, correlator_files = select_files_to_validate(
        set_directory, stored, logger
    )
    offer_error_files_deletion(set_directory, error_files, remover, logger)

    log_files = [p for p in log_files if os.path.exists(p)]
    program_type, log_files = validate_log_files(
        log_files, os.path.join(metadata_directory, METADATA_FILENAME), remover, logger
    )
    if program_type == INVERT_PROGRAM_TYPE:
        validate_invert_correlator_files(log_files, correlator_files, remover, logger)

    store_validated_file_lists(set_directory, metadata_directory, program_type, logger)


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


@click.command()
@click.option(
    "-raw_dir",
    "--raw_data_files_set_directory_path",
    "raw_data_files_set_directory_path",
    required=True,
    callback=directory.must_exist,
    help="Path to the raw data file set directory",
)
@click.option(
    "-ax_files_dir",
    "--auxiliary_files_directory",
    "auxiliary_files_directory",
    default=None,
    callback=directory.can_create,
    help="Directory where the script's log file will be stored.",
)
@click.option(
    "-metadata_dir",
    "--metadata_directory",
    "metadata_directory",
    required=False,
    default=None,
    callback=directory.can_create,
    help="Directory for metadata files",
)
@click.option(
    "-log_on",
    "--enable_logging",
    "enable_logging",
    is_flag=True,
    default=False,
    help="Enable logging.",
)
@click.option(
    "-logs_dir",
    "--logs_directory",
    "logs_directory",
    required=False,
    default=None,
    callback=directory.can_create,
    help="Directory for log files",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=validate_log_filename,
    help="Specific name for the script's log file.",
)
@click.option(
    "-no_quarantine",
    "--no_quarantine",
    "no_quarantine",
    is_flag=True,
    default=False,
    help=(
        "Delete rejected files permanently instead of moving them to the "
        "quarantine directory (.err files are always deleted)."
    ),
)
@click.option(
    "-quarantine_dir",
    "--quarantine_directory",
    "quarantine_directory",
    default=None,
    help=(
        "Where rejected files are moved. Default: "
        "'<auxiliary files directory>/quarantine/<timestamp>'."
    ),
)
def main(
    raw_data_files_set_directory_path,
    auxiliary_files_directory,
    metadata_directory,
    enable_logging,
    logs_directory,
    log_filename,
    no_quarantine,
    quarantine_directory,
):
    # HANDLE EMPTY INPUT ARGUMENTS

    # Default to input file's parent directory if no auxiliary directory
    # specified
    auxiliary_directory_given = auxiliary_files_directory is not None
    if auxiliary_files_directory is None:
        auxiliary_files_directory = os.path.dirname(raw_data_files_set_directory_path)

    # If no log filename is provided, generate a default name
    if log_filename is None:
        script_name = os.path.basename(sys.argv[0])
        log_filename = script_name.replace(".py", "_python_script.log")

    # Setup subdirectory paths
    if metadata_directory is None:
        metadata_directory = os.path.join(auxiliary_files_directory, "metadata")
    if logs_directory is None:
        logs_directory = os.path.join(auxiliary_files_directory, "logs")
    if quarantine_directory is None:
        quarantine_directory = default_quarantine_directory(
            raw_data_files_set_directory_path,
            auxiliary_files_directory,
            auxiliary_directory_given,
            datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        )

    # Ensure subdirectories exist (the quarantine one only when used)
    os.makedirs(metadata_directory, exist_ok=True)
    os.makedirs(logs_directory, exist_ok=True)

    # INITIATE SCRIPT AND LOGGING

    click.echo("   -- Validating raw qpb data file set initiated.")
    logger = LoggingWrapper(logs_directory, log_filename, enable_logging)
    logger.initiate_script_logging()

    remover = FileRemover(
        raw_data_files_set_directory_path,
        quarantine_directory,
        use_quarantine=not no_quarantine,
        logger=logger,
    )

    try:
        run_validation(
            raw_data_files_set_directory_path, metadata_directory, remover, logger
        )
    except StopValidation as stop:
        if stop.exit_code == 0:
            logger.info(str(stop), to_console=True)
        else:
            logger.error(str(stop), to_console=True)
        logger.terminate_script_logging()
        sys.exit(stop.exit_code)

    if os.path.isdir(quarantine_directory):
        click.echo(f"   Rejected files were moved to: {quarantine_directory}")

    # FINALIZE SCRIPT AND LOGGING

    logger.terminate_script_logging()
    click.echo("   -- Validating raw qpb data file set completed.")


if __name__ == "__main__":
    main()
