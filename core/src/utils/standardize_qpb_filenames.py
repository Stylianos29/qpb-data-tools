"""
standardize_qpb_filenames.py

Summary:
    Renames the files of a raw qpb data files set so that their names
    are uniform and follow the QPB-runner filename conventions, without
    losing any information that is recorded only in the filenames.

    QPB-runner ("scan.sh") builds filenames as:
        <method>_<kernel>_<printed constants>_<outer..inner varying>[_<extra>]
    with labels from "MODIFIABLE_PARAMETERS_LABELS_DICTIONARY" and "."
    replaced by "p". Over time, the files of one set can end up named
    inconsistently (e.g. the bare mass printed only for a later batch).
    This script recomputes every name from the actual parameter values.

Input:
    Required (one of):
        - data_files_set_directory: Path to the data files set directory
          (e.g. data_files/raw/invert/KL_several_config_varying_n)
        - undo_manifest: Path to a manifest written by a previous
          "--apply" run, to reverse it
    Optional:
        - apply: Perform the renaming (default is a dry run)
        - use_defaults: Do not ask questions; keep all constants already
          in the filenames, add nothing, keep the directory name
        - several_max: Max number of distinct values for a parameter to
          be classified "several" instead of "varying" (display and
          directory name suggestion only)
        - preview_count: Number of renames shown in the preview (0: all)
        - plan_output_path: Write the full planned mapping (YAML) during
          a dry run
        - manifest_directory: Where manifests are written on "--apply"
          (default: parent directory of the data files set)
        - enable_logging, logs_directory, log_filename

Output:
    Dry run: analysis table, decisions, preview of renames.
    Apply:   renamed files (and optionally renamed set directory with
             its sidecar files), updated ".json" filter files, a note
             appended to the ".yaml" sidecar, and a manifest (YAML)
             that "--undo" can reverse.

Safety rules:
    1. Parameter values come from the log file contents first; the
       filename only fills the gaps.
    2. Any disagreement between filename and contents, any filename the
       project's own filename parser reads differently, any missing
       value, or any name collision stops the script before anything is
       touched.
    3. Every new name is re-parsed with the project's filename parser and
       must give back exactly the intended parameters and extra text.
    4. A constant parameter can be dropped from the names only if every
       log file records it under the same parameter name.
    5. Renames are done in two phases (through temporary names) with
       rollback on failure, and never overwrite an existing file.
    6. Extra text (filename text that is not a parameter, e.g.
       "deflation") is kept as a suffix unless you choose to drop it from
       all filenames. It is recorded in no log file, so dropping it is
       allowed only if the names stay unique; if they would collide, the
       parameters that tell the files apart are offered for adding.

Usage:
    python standardize_qpb_filenames.py -set_dir <path>            # dry run
    python standardize_qpb_filenames.py -set_dir <path> -apply
    python standardize_qpb_filenames.py -undo <manifest.yaml>
"""

import os
import re
import sys
import json
import datetime
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

import click
import yaml

from library import LoggingWrapper
import library.constants as constants
from library.parsing import (
    extract_scalar_parameters_from_filename,
    extract_scalar_parameters_from_file_contents,
)
from library.parsing._shared_parsing_utils import _convert_to_type
from library.validation.click_validators import (
    directory,
    validate_log_filename,
)
from src.processing._param_transform_config import SOLVER_PARAMETER_RESOLUTION_RULES


# =============================================================================
# CONSTANTS
# =============================================================================

DATA_FILE_EXTENSIONS = (".txt", ".dat", ".err")
LOG_FILE_EXTENSION = ".txt"

# Always the first two tokens of every filename, never dropped
ALWAYS_KEPT_PARAMETERS: Tuple[str, ...] = (
    "Overlap_operator_method",
    "Kernel_operator_type",
)

# Filename parameters stored in the log files under a different name. Used
# only to check filename/contents agreement, never to allow dropping.
FILENAME_TO_CONTENTS_ALIASES = {
    "Rational_order": ("KL_diagonal_order", "Neuberger_order", "Zolotarev_order"),
}

# Some programs print a single, generic "Solver epsilon" / "Solver max iters"
# line. The processing step resolves it (SOLVER_PARAMETER_RESOLUTION_RULES,
# per overlap method and invert/non-invert) to the same canonical parameter
# as one of these filename parameters, so the log file records that
# filename parameter too (e.g. "EpsMSCG" for non-invert KL programs).
GENERIC_SOLVER_PARAMETERS = (
    "Generic_solver_epsilon",
    "Generic_solver_max_iterations",
)
CANONICAL_TO_FILENAME_SOLVER_PARAMETER = {
    "MSCG_epsilon": "Inner_solver_epsilon",
    "MSCG_max_iterations": "Inner_solver_max_iterations",
    "CG_epsilon": "Outer_solver_epsilon",
    "CG_max_iterations": "Outer_solver_max_iterations",
}

# Not handled as filename tokens: the "Lattice_geometry" filename regex uses
# group names different from its key, so the project's parser can't read it.
EXCLUDED_PARAMETERS = ("Lattice_geometry",)

# Parameters taken from the contents only if they already appear in some
# filename (they describe the run environment, not the physics)
CONTENTS_VALUES_IGNORED_UNLESS_IN_FILENAMES = ("MPI_geometry",)

# Names used in tables/prompts for value-only tokens
DISPLAY_LABELS = {
    "Overlap_operator_method": "method",
    "Kernel_operator_type": "kernel",
}

# Extra text: filename text that is not a parameter (e.g. "deflation"),
# recorded in no log file
EXTRA_TEXT_LABEL = "extra"
NO_EXTRA_TEXT = "(none)"

# Exponential-format parameters (tolerances like EpsCG) are rarely varied,
# so 3 or more distinct values already count as "varying"
EXPONENTIAL_FORMAT_SEVERAL_MAX = 2

TEMPORARY_RENAME_SUFFIX = ".standardizing_tmp"
MANIFEST_FILENAME_TEMPLATE = "{set_name}_filenames_standardization_{timestamp}.yaml"
FLOAT_SIGNIFICANT_DIGITS = 10
SCRIPT_NAME = "standardize_qpb_filenames.py"


class StandardizationError(Exception):
    """Raised when the data files set cannot be safely standardized."""


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass(frozen=True)
class TokenSpecification:
    """How a parameter is written as a single filename token."""

    parameter: str
    label: str  # Literal prefix of the token ("" for value-only tokens)
    regex: "re.Pattern"
    value_type: type
    value_map: Dict[str, Any]
    is_exponent_style: bool

    @property
    def display_label(self) -> str:
        return DISPLAY_LABELS.get(self.parameter, self.label)


@dataclass
class ParsedFilename:
    """Token-level reading of a filename stem."""

    parameter_tokens: List[Tuple[str, str]]  # (parameter, value spelling)
    values: Dict[str, Any]
    extra_tokens: List[str]
    split_tokens: List[str] = field(default_factory=list)  # Glued, now split


@dataclass
class FileGroup:
    """All data files sharing the same stem (.txt, .dat, .err)."""

    stem: str
    extensions: List[str]
    parsed: ParsedFilename
    contents_values: Dict[str, Any]  # Same parameter name as in filenames
    effective_values: Dict[str, Any]
    project_additional_text: str
    new_stem: Optional[str] = None  # Assigned by build_new_stems()

    @property
    def target_stem(self) -> str:
        """The new stem; fails loudly if read before it is assigned."""
        if self.new_stem is None:
            raise StandardizationError(f"'{self.stem}': new name not assigned yet.")
        return self.new_stem


@dataclass
class ParameterSummary:
    parameter: str
    display_label: str
    distinct_values: List[Any]
    groups_with_value: int  # Fewer than all: not applicable to the others
    groups_with_token: int
    groups_in_log_files: int
    # Every group with this parameter in its filename also records it in its
    # log file under the same name, so the filename token can be dropped
    recoverable_from_log_files: bool
    classification: str  # "constant", "several" or "varying"


@dataclass
class RenamePlan:
    set_directory: str
    new_set_directory: Optional[str]
    parameters_order: List[str]
    kept_constants: List[str]
    dropped_constants: List[str]
    added_parameters: List[str]
    file_renames: List[Tuple[str, str]]  # Filenames inside the set directory
    stem_renames: Dict[str, str]
    sidecar_renames: List[Tuple[str, str]] = field(default_factory=list)
    # Extra texts removed from the filenames (empty: extra text kept)
    dropped_extra_text: List[str] = field(default_factory=list)


# =============================================================================
# VALUE HELPERS
# =============================================================================


def comparable(value: Any) -> Any:
    """Normalize a value so that equal parameter values compare equal."""
    if isinstance(value, float):
        return float(f"{value:.{FLOAT_SIGNIFICANT_DIGITS}g}")
    if isinstance(value, str):
        # Collapse whitespace; "(1, 4, 4, 4)" and "(1,4,4,4)" compare equal
        return re.sub(r"\s*,\s*", ",", " ".join(value.split()))
    if isinstance(value, (list, tuple)):
        return tuple(comparable(item) for item in value)
    return value


def values_agree(first: Any, second: Any) -> bool:
    return comparable(first) == comparable(second)


def sort_values(values) -> List[Any]:
    return sorted(values, key=lambda v: (type(v).__name__, v))


def describe_values(values: List[Any], maximum_shown: int = 6) -> str:
    shown = [str(v) for v in values[:maximum_shown]]
    if len(values) > maximum_shown:
        shown.append(f"... (+{len(values) - maximum_shown})")
    return ", ".join(shown)


# =============================================================================
# FILENAME TOKENS
# =============================================================================


def build_token_specifications() -> Dict[str, TokenSpecification]:
    """Derive single-token regexes from FILENAME_SCALAR_PATTERNS_DICTIONARY."""
    specifications = {}
    for parameter, info in constants.FILENAME_SCALAR_PATTERNS_DICTIONARY.items():
        if parameter in EXCLUDED_PARAMETERS:
            continue
        # Token-level patterns: underscores are the token separators
        pattern = info["pattern"].strip("_")
        specifications[parameter] = TokenSpecification(
            parameter=parameter,
            label=pattern.split("(?P<", 1)[0],
            regex=re.compile(pattern),
            value_type=info["type"],
            value_map=info.get("value_map", {}),
            is_exponent_style="e[+-]" in pattern,
        )
    return specifications


def convert_token_value(specification: TokenSpecification, raw_value: str) -> Any:
    """Convert a token's value text the way the project's parser does."""
    mapped_value = specification.value_map.get(raw_value, raw_value)
    if specification.parameter == "MPI_geometry":
        return f"(1, {', '.join(mapped_value)})"
    return _convert_to_type(
        mapped_value, specification.value_type, specification.parameter
    )


def match_token(
    token: str, specifications: Dict[str, TokenSpecification]
) -> List[Tuple[TokenSpecification, "re.Match"]]:
    """
    Match a token to parameter(s). A token matching no parameter as a whole
    is tried as a glued sequence of parameter tokens (a missing "_", e.g.
    "EpsCG1e-12m0p02"); it is split only if exactly one split exists.
    Returns [] for extra text.
    """
    whole = [
        (spec, match)
        for spec in specifications.values()
        if (match := spec.regex.fullmatch(token))
    ]
    if len(whole) > 1:
        names = ", ".join(spec.parameter for spec, _ in whole)
        raise StandardizationError(
            f"Token '{token}' matches several parameters ({names})."
        )
    if whole:
        return whole

    decompositions = []

    def search(rest, accumulated):
        if len(decompositions) > 1:
            return
        if not rest:
            if len(accumulated) > 1:
                decompositions.append(list(accumulated))
            return
        for end in range(1, len(rest) + 1):
            for spec in specifications.values():
                match = spec.regex.fullmatch(rest[:end])
                if match:
                    accumulated.append((spec, match))
                    search(rest[end:], accumulated)
                    accumulated.pop()

    search(token, [])
    return decompositions[0] if len(decompositions) == 1 else []


def project_parse(stem: str) -> Tuple[Dict[str, Any], str]:
    """Run the project's filename parser; return (parameters, extra text)."""
    try:
        # ".txt" is stripped by the parser (".err" is not)
        parameters = extract_scalar_parameters_from_filename(stem + LOG_FILE_EXTENSION)
    except Exception as error:  # The parser is the reference: any failure stops
        raise StandardizationError(
            f"'{stem}': the project's filename parser failed ({error})."
        )
    additional_text = "_".join(parameters.pop("Additional_text", []))
    return parameters, additional_text


def parse_filename_stem(
    stem: str, specifications: Dict[str, TokenSpecification]
) -> ParsedFilename:
    """
    Split a stem into parameter tokens (in order) and extra tokens, using the
    project's filename parser as the reference: a token it does not read as
    a parameter is extra text, and any other disagreement stops.
    """
    project_values, project_extra = project_parse(stem)

    items = []  # (token, [(parameter, spelling, value)]) ; [] means extra
    for token in stem.split("_"):
        if not token:
            raise StandardizationError(f"'{stem}': empty token (double '_').")
        try:
            matches = match_token(token, specifications)
        except StandardizationError as error:
            raise StandardizationError(f"'{stem}': {error}")
        parameters = []
        for specification, match in matches:
            spelling = match.group(specification.parameter)
            value = convert_token_value(specification, spelling)
            if value is None:
                raise StandardizationError(
                    f"'{stem}': cannot convert '{match.group(0)}'."
                )
            parameters.append((specification.parameter, spelling, value))
        # Defer to the project's parser on what counts as a parameter
        if any(p not in project_values for p, _, _ in parameters):
            parameters = []
        items.append((token, parameters))

    parameter_tokens, values, extra_tokens, split_tokens = [], {}, [], []
    for token, parameters in items:
        if not parameters:
            extra_tokens.append(token)
            continue
        if len(parameters) > 1:
            split_tokens.append(token)
        for parameter, spelling, value in parameters:
            if parameter in values:
                raise StandardizationError(
                    f"'{stem}': parameter '{parameter}' appears twice."
                )
            parameter_tokens.append((parameter, spelling))
            values[parameter] = value

    differing = mismatching_parameters(values, project_values)
    if differing or project_extra != "_".join(extra_tokens):
        raise StandardizationError(
            f"'{stem}': read differently by the project's filename parser "
            f"(parameters: {differing or 'none'}; extra text: "
            f"'{project_extra}' vs '{'_'.join(extra_tokens)}')."
        )
    return ParsedFilename(parameter_tokens, values, extra_tokens, split_tokens)


def mismatching_parameters(
    expected: Dict[str, Any], obtained: Dict[str, Any]
) -> List[str]:
    """Parameters whose values differ, or that exist on one side only."""
    return sorted(
        parameter
        for parameter in set(expected) | set(obtained)
        if parameter not in expected
        or parameter not in obtained
        or not values_agree(expected[parameter], obtained[parameter])
    )


# =============================================================================
# ANALYSIS
# =============================================================================


def read_contents_values(
    log_file_path: str,
    specifications: Dict[str, TokenSpecification],
    overlap_method: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Return (direct, aliased) values from a log file: "direct" uses the same
    parameter name as the filename patterns (or a generic solver parameter
    that the processing step resolves to the same canonical parameter),
    "aliased" a different one.
    """
    with open(log_file_path, "r", errors="replace") as file:
        lines = file.readlines()
    try:
        contents = extract_scalar_parameters_from_file_contents(lines)
    except (ValueError, TypeError):
        return {}, {}

    direct = {p: contents[p] for p in specifications if p in contents}

    is_invert = contents.get("Main_program_type") == "invert"
    rules: Dict[str, str] = (
        {}
        if overlap_method is None
        else SOLVER_PARAMETER_RESOLUTION_RULES.get((overlap_method, is_invert), {})
    )
    for generic in GENERIC_SOLVER_PARAMETERS:
        canonical = rules.get(generic)
        if generic not in contents or canonical is None:
            continue
        target = CANONICAL_TO_FILENAME_SOLVER_PARAMETER.get(canonical)
        if target in specifications and target not in direct:
            direct[target] = contents[generic]

    aliased = {}
    for parameter, aliases in FILENAME_TO_CONTENTS_ALIASES.items():
        for alias in aliases:
            if alias in contents:
                aliased[parameter] = contents[alias]
                break
    return direct, aliased


def collect_file_groups(set_directory: str) -> Tuple[Dict[str, List[str]], List[str]]:
    """Group data files by stem; return (groups, ignored filenames)."""
    groups, ignored = defaultdict(list), []
    for entry in sorted(os.listdir(set_directory)):
        if not os.path.isfile(os.path.join(set_directory, entry)):
            continue
        if entry.endswith(TEMPORARY_RENAME_SUFFIX):
            raise StandardizationError(
                f"Leftover temporary file '{entry}' found: a previous run was "
                "interrupted. Inspect the directory before continuing."
            )
        stem, extension = os.path.splitext(entry)
        if extension in DATA_FILE_EXTENSIONS:
            groups[stem].append(extension)
        else:
            ignored.append(entry)
    return dict(groups), ignored


def analyze_data_files_set(
    set_directory: str,
    specifications: Dict[str, TokenSpecification],
    logger: LoggingWrapper,
) -> Tuple[List[FileGroup], List[str]]:
    """Read every file group; stop on any filename/contents disagreement."""
    stems_extensions, ignored = collect_file_groups(set_directory)
    if not stems_extensions:
        raise StandardizationError(f"No data files found in '{set_directory}'.")

    click.echo(f"   Reading {len(stems_extensions)} file groups ...")
    problems, groups = [], []
    for stem, extensions in sorted(stems_extensions.items()):
        # 1. Token reading, checked against the project's filename parser
        try:
            parsed = parse_filename_stem(stem, specifications)
        except StandardizationError as error:
            problems.append(str(error))
            continue

        # 2. Contents first, filename only for the gaps; any conflict stops
        direct, aliased = {}, {}
        if LOG_FILE_EXTENSION in extensions:
            direct, aliased = read_contents_values(
                os.path.join(set_directory, stem + LOG_FILE_EXTENSION),
                specifications,
                parsed.values.get("Overlap_operator_method"),
            )
        for parameter in CONTENTS_VALUES_IGNORED_UNLESS_IN_FILENAMES:
            if parameter not in parsed.values:
                direct.pop(parameter, None)

        for parameter, filename_value in parsed.values.items():
            contents_value = direct.get(parameter, aliased.get(parameter))
            if contents_value is not None and not values_agree(
                filename_value, contents_value
            ):
                problems.append(
                    f"'{stem}': {parameter} is {filename_value} in the filename "
                    f"but {contents_value} in the log file."
                )

        groups.append(
            FileGroup(
                stem=stem,
                extensions=sorted(extensions),
                parsed=parsed,
                contents_values=direct,
                effective_values={**parsed.values, **direct},
                project_additional_text="_".join(parsed.extra_tokens),
            )
        )

    if ignored:
        logger.warning(f"Ignored non-data files: {ignored}", to_console=False)
        click.echo(f"   Note: {len(ignored)} non-data file(s) ignored: {ignored[:5]}")
    if problems:
        raise StandardizationError(
            "The data files set cannot be standardized safely:\n  - "
            + "\n  - ".join(problems[:50])
            + (f"\n  ... and {len(problems) - 50} more." if len(problems) > 50 else "")
        )
    return groups, ignored


def summarize_parameters(
    groups: List[FileGroup],
    specifications: Dict[str, TokenSpecification],
    several_max: int,
) -> Dict[str, ParameterSummary]:
    summaries = {}
    all_parameters = {p for g in groups for p in g.effective_values}
    for parameter in all_parameters:
        values = {
            comparable(g.effective_values[parameter])
            for g in groups
            if parameter in g.effective_values
        }
        count = len(values)
        maximum_for_several = (
            EXPONENTIAL_FORMAT_SEVERAL_MAX
            if parameter in constants.PARAMETERS_WITH_EXPONENTIAL_FORMAT
            else several_max
        )
        classification = (
            "constant"
            if count == 1
            else "several" if count <= maximum_for_several else "varying"
        )
        summaries[parameter] = ParameterSummary(
            parameter=parameter,
            display_label=specifications[parameter].display_label,
            distinct_values=sort_values(values),
            groups_with_value=sum(parameter in g.effective_values for g in groups),
            groups_with_token=sum(parameter in g.parsed.values for g in groups),
            groups_in_log_files=sum(parameter in g.contents_values for g in groups),
            recoverable_from_log_files=all(
                parameter in g.contents_values
                for g in groups
                if parameter in g.parsed.values
            ),
            classification=classification,
        )
    return summaries


def extra_text_counts(groups: List[FileGroup]) -> Counter:
    """How many file groups carry each extra text ("(none)": no extra text)."""
    return Counter(g.project_additional_text or NO_EXTRA_TEXT for g in groups)


def print_analysis_table(
    summaries: Dict[str, ParameterSummary],
    number_of_groups: int,
    groups: Optional[List[FileGroup]] = None,
) -> None:
    header = (
        f"   {'Parameter':<32}{'Label':<14}{'#':>4}  {'Class':<9}"
        f"{'In names':>10}  {'In logs':<8}Values"
    )
    click.echo("\n" + header)
    click.echo("   " + "-" * (len(header) + 10))

    def sort_key(summary):
        order = {"constant": 0, "several": 1, "varying": 2}
        return (
            order[summary.classification],
            -summary.groups_with_token,
            summary.parameter,
        )

    for summary in sorted(summaries.values(), key=sort_key):
        in_names = f"{summary.groups_with_token}/{number_of_groups}"
        in_logs = f"{summary.groups_in_log_files}/{number_of_groups}"
        flags = ""
        if summary.groups_with_value < number_of_groups:
            flags += f"  [n/a for {number_of_groups - summary.groups_with_value}]"
        if 0 < summary.groups_with_token < summary.groups_with_value:
            flags += "  <- inconsistent"
        click.echo(
            f"   {summary.parameter:<32}{summary.display_label:<14}"
            f"{len(summary.distinct_values):>4}  {summary.classification:<9}"
            f"{in_names:>10}  {in_logs:<8}"
            f"{describe_values(summary.distinct_values)}{flags}"
        )

    has_extra_text = bool(groups) and any(g.project_additional_text for g in groups)
    if groups and has_extra_text:
        counts = extra_text_counts(groups)
        with_text = sum(1 for g in groups if g.project_additional_text)
        classification = "constant" if len(counts) == 1 else "varies"
        in_names = f"{with_text}/{number_of_groups}"
        in_logs = f"0/{number_of_groups}"
        values = [f"{text} ({count})" for text, count in counts.most_common()]
        click.echo(
            f"   {'(extra text)':<32}{EXTRA_TEXT_LABEL:<14}{len(counts):>4}  "
            f"{classification:<9}{in_names:>10}  {in_logs:<8}"
            f"{describe_values(values)}"
        )

    click.echo(
        "   (In names / In logs: file groups with the parameter in the filename /"
        " in the log file; n/a: value known nowhere, token omitted."
        + (" extra: filename text that is not a parameter." if has_extra_text else "")
        + ")\n"
    )


# =============================================================================
# DECISIONS
# =============================================================================


def ask_for_labels(
    question: str,
    candidates: List[ParameterSummary],
    use_defaults: bool,
) -> List[str]:
    """Ask for a comma-separated subset of candidate labels; default none."""
    if not candidates or use_defaults:
        return []
    by_label = {c.display_label: c.parameter for c in candidates}
    choices = ", ".join(by_label)
    while True:
        answer = click.prompt(
            f"   {question}\n   Candidates: {choices}\n"
            "   Enter labels separated by commas, 'all', or press Enter for none",
            default="",
            show_default=False,
        ).strip()
        if not answer:
            return []
        if answer.lower() == "all":
            return list(by_label.values())
        labels = [label.strip() for label in answer.split(",") if label.strip()]
        unknown = [label for label in labels if label not in by_label]
        if unknown:
            click.echo(f"   Unknown label(s): {unknown}. Try again.")
            continue
        return [by_label[label] for label in labels]


def decide_included_parameters(
    summaries: Dict[str, ParameterSummary],
    number_of_groups: int,
    use_defaults: bool,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    """Return (included, kept constants, dropped constants, added)."""
    problems = []
    for parameter in ALWAYS_KEPT_PARAMETERS:
        summary = summaries.get(parameter)
        if summary is None or summary.groups_with_value < number_of_groups:
            problems.append(f"'{parameter}' is unknown for some file groups.")

    in_names = [s for s in summaries.values() if s.groups_with_token > 0]
    mandatory = [
        s.parameter
        for s in in_names
        if s.classification != "constant" and s.parameter not in ALWAYS_KEPT_PARAMETERS
    ]
    constants_in_names = [
        s
        for s in in_names
        if s.classification == "constant" and s.parameter not in ALWAYS_KEPT_PARAMETERS
    ]
    droppable = [s for s in constants_in_names if s.recoverable_from_log_files]
    forced = [s for s in constants_in_names if not s.recoverable_from_log_files]
    addable = [
        s
        for s in summaries.values()
        if s.groups_with_token == 0
        and s.classification != "constant"
        and s.parameter not in ALWAYS_KEPT_PARAMETERS
    ]

    if forced:
        click.echo(
            "   Constants kept because not every log file records them under "
            "the same name (dropping would lose information): "
            + ", ".join(s.display_label for s in forced)
        )
    dropped = ask_for_labels(
        "Which constant parameters should be DROPPED from the filenames?",
        droppable,
        use_defaults,
    )
    added = ask_for_labels(
        "These parameters vary but appear in no filename. Which should be ADDED?",
        addable,
        use_defaults,
    )

    kept_constants = [
        s.parameter for s in constants_in_names if s.parameter not in dropped
    ]
    included = list(ALWAYS_KEPT_PARAMETERS) + mandatory + kept_constants + added
    not_applicable = [
        f"{summaries[p].display_label} ({number_of_groups - summaries[p].groups_with_value})"
        for p in included
        if summaries[p].groups_with_value < number_of_groups
    ]
    if not_applicable:
        click.echo(
            "   Omitted where the value is known nowhere (not applicable), "
            "number of file groups: " + ", ".join(not_applicable)
        )
    if problems:
        raise StandardizationError(
            "\n  - ".join(["Cannot build uniform names:"] + problems)
        )
    return included, kept_constants, dropped, added


def find_name_collisions(
    groups: List[FileGroup], parameters: List[str], keep_extra_text: bool
) -> List[List[FileGroup]]:
    """File groups that the given parameters (and extra text) cannot tell apart."""
    by_key: Dict[Tuple[Any, ...], List[FileGroup]] = defaultdict(list)
    for g in groups:
        key = tuple(comparable(g.effective_values.get(p)) for p in parameters)
        if keep_extra_text:
            key += (g.project_additional_text,)
        by_key[key].append(g)
    return [colliding for colliding in by_key.values() if len(colliding) > 1]


def decide_extra_text(
    groups: List[FileGroup],
    summaries: Dict[str, ParameterSummary],
    included: List[str],
    use_defaults: bool,
) -> Tuple[bool, List[str]]:
    """
    Ask whether to keep the extra text or drop it from all filenames.
    Returns (keep extra text, parameters to add so that it can be dropped).
    If dropping would make names collide, the parameters that tell the
    colliding file groups apart are offered; otherwise the text is kept.
    """
    if use_defaults or not any(g.project_additional_text for g in groups):
        return True, []

    click.echo("\n   Extra text in the filenames (recorded in no log file):")
    for text, count in extra_text_counts(groups).most_common():
        click.echo(f"     {text}: {count}/{len(groups)} file groups")
    choice = click.prompt(
        "   Keep the extra text as it is, or drop it from all filenames?",
        type=click.Choice(["keep", "drop"]),
        default="keep",
    )
    if choice == "keep":
        return True, []

    collisions = find_name_collisions(groups, included, keep_extra_text=False)
    if not collisions:
        return False, []

    candidates = [
        p
        for p in summaries
        if p not in included
        and p not in ALWAYS_KEPT_PARAMETERS
        and any(
            len({comparable(g.effective_values.get(p)) for g in colliding}) > 1
            for colliding in collisions
        )
    ]
    colliding_count = sum(len(colliding) for colliding in collisions)
    if candidates and not find_name_collisions(
        groups, included + candidates, keep_extra_text=False
    ):
        labels = ", ".join(summaries[p].display_label for p in candidates)
        click.echo(
            f"   Without the extra text, {colliding_count} file groups would share "
            f"names; {labels} tells them apart."
        )
        if click.confirm(
            f"   Add {labels} to the filenames and drop the extra text?", default=True
        ):
            return False, candidates
        click.echo("   Keeping the extra text.")
        return True, []

    click.echo(
        f"   Keeping the extra text: without it, {colliding_count} file groups would "
        "share names and no parameter tells them apart."
    )
    return True, []


def merge_parameters_order(
    groups: List[FileGroup],
    included: List[str],
    summaries: Dict[str, ParameterSummary],
) -> List[str]:
    """
    Order of the parameter tokens, keeping the set's own order: the most
    common existing order that has exactly the included parameters, if any;
    otherwise a consensus of the token orders of all current filenames, with
    parameters present in no filename inserted before the first parameter
    with more values.
    """
    body = [p for p in included if p not in ALWAYS_KEPT_PARAMETERS]
    sequences = Counter(
        tuple(p for p, _ in g.parsed.parameter_tokens if p in body) for g in groups
    )
    complete = [s for s, _ in sequences.most_common() if set(s) == set(body)]
    if complete:
        return list(ALWAYS_KEPT_PARAMETERS) + list(complete[0])

    precedes, positions, first_seen = Counter(), defaultdict(list), {}
    for sequence, count in sequences.items():
        length = len(sequence)
        for index, parameter in enumerate(sequence):
            first_seen.setdefault(parameter, len(first_seen))
            positions[parameter].extend(
                [index / (length - 1) if length > 1 else 0.5] * count
            )
            for later in sequence[index + 1 :]:
                precedes[(parameter, later)] += count

    # Greedy: next is the parameter preceded by the fewest remaining ones,
    # where "q precedes p" if it does so in most filenames with both
    def majority_precedes(q, p):
        return precedes[(q, p)] > precedes[(p, q)]

    remaining, order = set(positions), []
    while remaining:
        best = min(
            remaining,
            key=lambda p: (
                sum(majority_precedes(q, p) for q in remaining if q != p),
                mean(positions[p]),
                first_seen[p],
            ),
        )
        order.append(best)
        remaining.remove(best)

    for parameter in [p for p in body if p not in order]:
        size = len(summaries[parameter].distinct_values)
        position = next(
            (
                i
                for i, p in enumerate(order)
                if len(summaries[p].distinct_values) > size
                and summaries[p].classification != "constant"
            ),
            len(order),
        )
        order.insert(position, parameter)
    return list(ALWAYS_KEPT_PARAMETERS) + order


def format_new_spelling(
    specification: TokenSpecification, value: Any, known_spellings: List[str]
) -> str:
    """Spelling for a value that appears in no current filename."""
    if specification.value_map:
        reverse = {v: k for k, v in specification.value_map.items()}
        return reverse.get(value, str(value))
    if specification.parameter == "MPI_geometry":
        return "".join(re.findall(r"\d+", str(value))[1:])
    if specification.value_type is int:
        return str(int(value))
    if specification.value_type is float:
        if specification.is_exponent_style:
            mantissa, exponent = f"{value:.{FLOAT_SIGNIFICANT_DIGITS - 1}e}".split("e")
            mantissa = mantissa.rstrip("0").rstrip(".")
            exponent = int(exponent)
            text = f"{mantissa}e{'-' if exponent < 0 else '+'}{abs(exponent)}"
            return text.replace(".", "p")
        # Reuse the number of decimals of the existing spellings, if uniform
        decimals = {len(s.split("p", 1)[1]) if "p" in s else 0 for s in known_spellings}
        if len(decimals) == 1:
            candidate = f"{value:.{decimals.pop()}f}"
            if values_agree(float(candidate), value):
                return candidate.replace(".", "p")
        text = f"{value:.{FLOAT_SIGNIFICANT_DIGITS}f}".rstrip("0")
        if text.endswith("."):
            text += "0"
        return text.replace(".", "p")
    return str(value)


def build_new_stems(
    groups: List[FileGroup],
    order: List[str],
    specifications: Dict[str, TokenSpecification],
    keep_extra_text: bool = True,
) -> List[str]:
    """Assign new stems; return the list of newly created spellings."""
    spellings = defaultdict(Counter)
    for g in groups:
        for parameter, spelling in g.parsed.parameter_tokens:
            spellings[(parameter, comparable(g.parsed.values[parameter]))][
                spelling
            ] += 1
    known_by_parameter = defaultdict(list)
    for (parameter, _), counter in spellings.items():
        known_by_parameter[parameter].extend(counter)

    new_spellings = set()
    for g in groups:
        tokens = []
        for parameter in order:
            if parameter not in g.effective_values:
                continue  # Not applicable to this group
            specification = specifications[parameter]
            value = g.effective_values[parameter]
            counter = spellings.get((parameter, comparable(value)))
            if counter:
                spelling = counter.most_common(1)[0][0]
            else:
                spelling = format_new_spelling(
                    specification, value, known_by_parameter[parameter]
                )
                new_spellings.add(
                    f"{specification.label}{spelling} ({parameter}={value})"
                )
            tokens.append(f"{specification.label}{spelling}")
        extra_tokens = g.parsed.extra_tokens if keep_extra_text else []
        g.new_stem = "_".join(tokens + extra_tokens)
    return sorted(new_spellings)


def verify_new_stems(
    groups: List[FileGroup],
    order: List[str],
    summaries: Dict[str, ParameterSummary],
    keep_extra_text: bool = True,
) -> None:
    """Uniqueness and round-trip checks; raise on any problem."""
    problems = []
    by_new_stem = defaultdict(list)
    for g in groups:
        by_new_stem[g.target_stem].append(g)

    for new_stem, colliding in by_new_stem.items():
        if len(colliding) < 2:
            continue
        differing = sorted(
            {
                summaries[p].display_label
                for p in summaries
                if len({comparable(g.effective_values.get(p)) for g in colliding}) > 1
            }
        )
        problems.append(
            f"{len(colliding)} groups would all be named '{new_stem}' "
            f"(e.g. {[g.stem for g in colliding[:3]]}). Parameters that tell them "
            f"apart: {differing or 'none known'}."
        )

    for g in groups:
        expected = {p: g.effective_values[p] for p in order if p in g.effective_values}
        expected_extra_text = g.project_additional_text if keep_extra_text else ""
        obtained, extra_text = project_parse(g.target_stem)
        differing = mismatching_parameters(expected, obtained)
        if differing or extra_text != expected_extra_text:
            problems.append(
                f"'{g.target_stem}' does not parse back correctly "
                f"(parameters: {differing or 'ok'}; extra text: '{extra_text}' "
                f"vs '{expected_extra_text}')."
            )
        # Filename-only values must survive
        lost = [
            p
            for p in g.parsed.values
            if p not in g.contents_values and p not in obtained
        ]
        if lost:
            problems.append(f"'{g.stem}': renaming would lose {lost}.")

    if problems:
        raise StandardizationError(
            "The proposed names are not safe:\n  - " + "\n  - ".join(problems[:30])
        )


# =============================================================================
# SET DIRECTORY AND SIDECARS
# =============================================================================


def find_sidecar_files(set_directory: str) -> List[str]:
    """Files next to the set directory named "<set name>.<extension>"."""
    parent, set_name = os.path.split(set_directory.rstrip(os.sep))
    return sorted(
        os.path.join(parent, entry)
        for entry in os.listdir(parent)
        if os.path.isfile(os.path.join(parent, entry))
        and os.path.splitext(entry)[0] == set_name
    )


def suggest_set_name(
    order: List[str],
    summaries: Dict[str, ParameterSummary],
    groups: List[FileGroup],
) -> str:
    method_token = groups[0].target_stem.split("_", 1)[0]
    # Method and kernel are never part of the set name: a set holds a single
    # method and one or both kernels by convention
    named = [p for p in order if p not in ALWAYS_KEPT_PARAMETERS]
    several = [
        summaries[p].display_label
        for p in named
        if summaries[p].classification == "several"
    ]
    varying = [
        summaries[p].display_label
        for p in named
        if summaries[p].classification == "varying"
    ]
    name = method_token
    if several:
        name += "_several_" + "_and_".join(several)
    if varying:
        name += "_varying_" + "_and_".join(varying)
    return name


def decide_set_directory_rename(
    set_directory: str, suggestion: str, use_defaults: bool
) -> Optional[str]:
    current_name = os.path.basename(set_directory.rstrip(os.sep))
    if use_defaults:
        return None
    click.echo(f"\n   Data files set directory: '{current_name}'")
    click.echo(f"   Suggested name from the parameter classification: '{suggestion}'")
    if not click.confirm(
        "   Rename the data files set directory (and its sidecar files)?", default=False
    ):
        return None
    parent = os.path.dirname(set_directory.rstrip(os.sep))
    while True:
        new_name = click.prompt("   New directory name", default=suggestion).strip()
        if not new_name or os.sep in new_name or new_name in (".", ".."):
            click.echo("   Invalid name.")
            continue
        if new_name == current_name:
            return None
        target = os.path.join(parent, new_name)
        if os.path.exists(target):
            click.echo(f"   '{target}' already exists. Choose another name.")
            continue
        return target


# =============================================================================
# PLANNING
# =============================================================================


def plan_standardization(
    set_directory: str,
    several_max: int,
    use_defaults: bool,
    logger: LoggingWrapper,
) -> Tuple[RenamePlan, List[FileGroup]]:
    specifications = build_token_specifications()
    groups, _ = analyze_data_files_set(set_directory, specifications, logger)
    summaries = summarize_parameters(groups, specifications, several_max)
    print_analysis_table(summaries, len(groups), groups)

    included, kept, dropped, added = decide_included_parameters(
        summaries, len(groups), use_defaults
    )
    keep_extra_text, added_for_extra_text = decide_extra_text(
        groups, summaries, included, use_defaults
    )
    included += added_for_extra_text
    added += added_for_extra_text

    order = merge_parameters_order(groups, included, summaries)
    new_spellings = build_new_stems(groups, order, specifications, keep_extra_text)
    verify_new_stems(groups, order, summaries, keep_extra_text)
    has_extra_text = any(g.parsed.extra_tokens for g in groups)

    click.echo(
        "   Filename pattern: "
        + "_".join(
            (
                f"{{{summaries[p].display_label}}}"
                if not specifications[p].label
                else f"{specifications[p].label}{{}}"
            )
            for p in order
        )
        + ("_{extra text}" if has_extra_text and keep_extra_text else "")
    )
    if dropped:
        click.echo(
            "   Dropped constants: "
            + ", ".join(summaries[p].display_label for p in dropped)
        )
    dropped_extra_text = (
        []
        if keep_extra_text
        else sorted({g.project_additional_text for g in groups} - {""})
    )
    if dropped_extra_text:
        click.echo("   Dropped extra text: " + ", ".join(dropped_extra_text))
    if new_spellings:
        click.echo(
            "   New value spellings (not in any current filename): "
            + ", ".join(new_spellings[:10])
        )
    split_tokens = sorted({t for g in groups for t in g.parsed.split_tokens})
    if split_tokens:
        click.echo(
            "   Tokens missing a '_' separator, to be split: "
            + ", ".join(split_tokens[:10])
        )
    if keep_extra_text and any(
        g.parsed.extra_tokens
        and g.parsed.parameter_tokens
        and g.stem.split("_")[-len(g.parsed.extra_tokens) :] != g.parsed.extra_tokens
        for g in groups
    ):
        click.echo("   Extra text found mid-name is moved to the end of the name.")

    file_renames, stem_renames = [], {}
    for g in groups:
        if g.target_stem != g.stem:
            stem_renames[g.stem] = g.target_stem
            file_renames.extend(
                (g.stem + ext, g.target_stem + ext) for ext in g.extensions
            )

    new_set_directory = None
    if stem_renames or not use_defaults:
        new_set_directory = decide_set_directory_rename(
            set_directory, suggest_set_name(order, summaries, groups), use_defaults
        )
    sidecar_renames = []
    if new_set_directory:
        new_name = os.path.basename(new_set_directory)
        for sidecar in find_sidecar_files(set_directory):
            extension = os.path.splitext(sidecar)[1]
            sidecar_renames.append(
                (sidecar, os.path.join(os.path.dirname(sidecar), new_name + extension))
            )

    plan = RenamePlan(
        set_directory=set_directory,
        new_set_directory=new_set_directory,
        parameters_order=order,
        kept_constants=kept,
        dropped_constants=dropped,
        added_parameters=added,
        file_renames=file_renames,
        stem_renames=stem_renames,
        sidecar_renames=sidecar_renames,
        dropped_extra_text=dropped_extra_text,
    )
    return plan, groups


def print_plan_preview(
    plan: RenamePlan, groups: List[FileGroup], preview_count: int
) -> None:
    renamed = [g for g in groups if g.target_stem != g.stem]
    click.echo(
        f"\n   {len(renamed)} of {len(groups)} file groups would be renamed "
        f"({len(plan.file_renames)} files)."
    )
    shown = renamed if preview_count == 0 else renamed[:preview_count]
    for g in shown:
        click.echo(f"     {g.stem}{{{','.join(g.extensions)}}}")
        click.echo(f"       -> {g.target_stem}")
    if len(shown) < len(renamed):
        click.echo(
            f"     ... ({len(renamed) - len(shown)} more; use -preview 0 or -plan_out)"
        )
    if plan.new_set_directory:
        click.echo(
            f"\n   Directory: {plan.set_directory}\n         -> {plan.new_set_directory}"
        )
    for old, new in plan.sidecar_renames:
        click.echo(f"   Sidecar:   {os.path.basename(old)} -> {os.path.basename(new)}")

    processed = processed_directory_for(plan.set_directory)
    if processed and (plan.file_renames or plan.new_set_directory):
        click.echo(
            f"\n   WARNING: processed outputs exist in '{processed}'. They refer to "
            "the current filenames and should be regenerated after renaming."
        )


def processed_directory_for(set_directory: str) -> Optional[str]:
    marker = f"{os.sep}raw{os.sep}"
    if marker not in set_directory:
        return None
    candidate = set_directory.replace(marker, f"{os.sep}processed{os.sep}", 1)
    return candidate if os.path.isdir(candidate) else None


def plan_to_dictionary(plan: RenamePlan, status: str) -> Dict[str, Any]:
    return {
        "tool": SCRIPT_NAME,
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "status": status,
        "set_directory": plan.set_directory,
        "new_set_directory": plan.new_set_directory,
        "parameters_order": plan.parameters_order,
        "kept_constants": plan.kept_constants,
        "dropped_constants": plan.dropped_constants,
        "added_parameters": plan.added_parameters,
        "dropped_extra_text": plan.dropped_extra_text,
        "sidecar_renames": [{"old": o, "new": n} for o, n in plan.sidecar_renames],
        "json_filter_updates": [],
        "yaml_experiment_name_updated": None,
        "yaml_note_appended": None,
        "file_renames": [{"old": o, "new": n} for o, n in plan.file_renames],
    }


def write_yaml(path: str, data: Dict[str, Any]) -> None:
    with open(path, "w") as file:
        yaml.safe_dump(data, file, sort_keys=False, width=1000)


# =============================================================================
# FILE OPERATIONS
# =============================================================================


def rename_files_two_phase(directory_path: str, renames: List[Tuple[str, str]]) -> None:
    """
    Rename (old, new) filenames inside a directory. All sources are first
    moved to temporary names, then to their targets, so chains like A->B,
    B->C cannot collide. Never overwrites; rolls back on any failure.
    """
    sources = {old for old, _ in renames}
    for old, new in renames:
        if not os.path.isfile(os.path.join(directory_path, old)):
            raise StandardizationError(f"Missing source file '{old}'.")
        if os.path.exists(os.path.join(directory_path, new)) and new not in sources:
            raise StandardizationError(f"Target '{new}' already exists.")
        if os.path.exists(os.path.join(directory_path, old + TEMPORARY_RENAME_SUFFIX)):
            raise StandardizationError(f"Temporary name for '{old}' already exists.")

    path = lambda name: os.path.join(directory_path, name)  # noqa: E731
    staged, finished = [], []
    try:
        for old, new in renames:
            os.rename(path(old), path(old + TEMPORARY_RENAME_SUFFIX))
            staged.append((old, new))
        for old, new in renames:
            if os.path.exists(path(new)):
                raise StandardizationError(f"Target '{new}' appeared during renaming.")
            os.rename(path(old + TEMPORARY_RENAME_SUFFIX), path(new))
            finished.append((old, new))
    except Exception:
        for old, new in reversed(finished):
            os.rename(path(new), path(old + TEMPORARY_RENAME_SUFFIX))
        for old, _ in reversed(staged):
            os.rename(path(old + TEMPORARY_RENAME_SUFFIX), path(old))
        raise


def rename_path(old: str, new: str) -> None:
    if os.path.exists(new):
        raise StandardizationError(f"Target '{new}' already exists.")
    os.rename(old, new)


def update_json_filters(json_path: str, stem_map: Dict[str, str]) -> Dict[str, str]:
    """Rewrite filename entries of a filter JSON; return {old: new} entries."""
    with open(json_path) as file:
        data = json.load(file)
    replacements = {}

    def map_entry(entry):
        if not isinstance(entry, str):
            return entry
        stem, extension = os.path.splitext(entry)
        if extension not in DATA_FILE_EXTENSIONS:
            stem, extension = entry, ""
        if stem in stem_map:
            replacements[entry] = stem_map[stem] + extension
            return replacements[entry]
        return entry

    if isinstance(data, dict):
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [map_entry(entry) for entry in value]
    if replacements:
        with open(json_path, "w") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    return replacements


def replace_yaml_experiment_name(yaml_path: str, old_name: str, new_name: str) -> bool:
    with open(yaml_path) as file:
        text = file.read()
    pattern = re.compile(
        r"^(\s*name:\s*)([\"']?)" + re.escape(old_name) + r"\2(\s*(?:#.*)?)$",
        re.MULTILINE,
    )
    new_text, count = pattern.subn(
        lambda m: f"{m.group(1)}{m.group(2)}{new_name}{m.group(2)}{m.group(3)}", text
    )
    if count:
        with open(yaml_path, "w") as file:
            file.write(new_text)
    return bool(count)


def append_yaml_note(yaml_path: str, lines: List[str]) -> None:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(yaml_path, "a") as file:
        file.write("\n# " + "-" * 70 + "\n")
        file.write(f"# [{stamp}] {SCRIPT_NAME}\n")
        for line in lines:
            file.write(f"#   {line}\n")


def sidecars_with_extension(set_directory: str, extension: str) -> List[str]:
    return [p for p in find_sidecar_files(set_directory) if p.endswith(extension)]


# =============================================================================
# APPLY AND UNDO
# =============================================================================


def apply_plan(
    plan: RenamePlan, manifest_directory: str, logger: LoggingWrapper
) -> str:
    set_name = os.path.basename(plan.set_directory.rstrip(os.sep))
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    manifest_path = os.path.join(
        manifest_directory,
        MANIFEST_FILENAME_TEMPLATE.format(set_name=set_name, timestamp=timestamp),
    )
    manifest = plan_to_dictionary(plan, status="in_progress")
    os.makedirs(manifest_directory, exist_ok=True)
    write_yaml(manifest_path, manifest)
    logger.info(f"Manifest written (in progress): {manifest_path}")

    # 1. Data files (rolled back automatically if any rename fails)
    try:
        rename_files_two_phase(plan.set_directory, plan.file_renames)
    except Exception as error:
        manifest["status"] = "failed_rolled_back"
        write_yaml(manifest_path, manifest)
        raise StandardizationError(f"Renaming failed and was rolled back: {error}")
    logger.info(f"Renamed {len(plan.file_renames)} files.")

    try:
        apply_sidecar_changes(plan, manifest, manifest_path, set_name)
    except Exception as error:
        raise StandardizationError(
            f"The data files WERE renamed, but updating the sidecars/directory "
            f"failed ({error}). Manifest left as 'in_progress': {manifest_path}"
        )
    manifest["status"] = "applied"
    write_yaml(manifest_path, manifest)
    logger.info(f"Manifest written (applied): {manifest_path}")
    return manifest_path


def apply_sidecar_changes(
    plan: RenamePlan, manifest: Dict[str, Any], manifest_path: str, set_name: str
) -> None:

    # 2. Filter JSON and YAML notes (before sidecars are renamed)
    for json_path in sidecars_with_extension(plan.set_directory, ".json"):
        replacements = update_json_filters(json_path, plan.stem_renames)
        if replacements:
            final_path = dict(plan.sidecar_renames).get(json_path, json_path)
            manifest["json_filter_updates"].append(
                {"path": final_path, "replacements": replacements}
            )
    for yaml_path in sidecars_with_extension(plan.set_directory, ".yaml"):
        final_path = dict(plan.sidecar_renames).get(yaml_path, yaml_path)
        if plan.new_set_directory and replace_yaml_experiment_name(
            yaml_path, set_name, os.path.basename(plan.new_set_directory)
        ):
            manifest["yaml_experiment_name_updated"] = final_path
        note = [
            f"Filenames standardized: {len(plan.stem_renames)} file groups renamed."
        ]
        if plan.dropped_constants:
            note.append(f"Dropped constants: {', '.join(plan.dropped_constants)}")
        if plan.dropped_extra_text:
            note.append(f"Dropped extra text: {', '.join(plan.dropped_extra_text)}")
        if plan.new_set_directory:
            note.append(f"Set renamed from '{set_name}'.")
        note.append(f"Manifest: {manifest_path}")
        append_yaml_note(yaml_path, note)
        manifest["yaml_note_appended"] = final_path

    # 3. Sidecars and set directory
    for old, new in plan.sidecar_renames:
        rename_path(old, new)
    if plan.new_set_directory:
        rename_path(plan.set_directory, plan.new_set_directory)


def undo_from_manifest(manifest_path: str, logger: LoggingWrapper) -> None:
    with open(manifest_path) as file:
        manifest = yaml.safe_load(file)
    if manifest.get("tool") != SCRIPT_NAME:
        raise StandardizationError(
            f"'{manifest_path}' is not a {SCRIPT_NAME} manifest."
        )
    if manifest.get("status") != "applied":
        raise StandardizationError(
            f"Manifest status is '{manifest.get('status')}', not 'applied'. "
            "An interrupted run must be inspected by hand."
        )

    set_directory = manifest["set_directory"]
    current_directory = manifest["new_set_directory"] or set_directory
    if not os.path.isdir(current_directory):
        raise StandardizationError(f"Directory '{current_directory}' not found.")

    click.echo(
        f"   Undoing {len(manifest['file_renames'])} file renames in '{current_directory}'."
    )
    if not click.confirm("   Proceed?", default=False):
        click.echo("   Nothing changed.")
        return

    # Reverse order of apply_plan
    if manifest["new_set_directory"]:
        rename_path(manifest["new_set_directory"], set_directory)
    for entry in reversed(manifest["sidecar_renames"]):
        rename_path(entry["new"], entry["old"])

    def original_path(path):
        for entry in manifest["sidecar_renames"]:
            if entry["new"] == path:
                return entry["old"]
        return path

    for update in manifest["json_filter_updates"]:
        reverse = {new: old for old, new in update["replacements"].items()}
        json_path = original_path(update["path"])
        with open(json_path) as file:
            data = json.load(file)
        for key, value in data.items():
            if isinstance(value, list):
                data[key] = [reverse.get(entry, entry) for entry in value]
        with open(json_path, "w") as file:
            json.dump(data, file, indent=2)
            file.write("\n")
    if manifest["yaml_experiment_name_updated"]:
        replace_yaml_experiment_name(
            original_path(manifest["yaml_experiment_name_updated"]),
            os.path.basename(manifest["new_set_directory"]),
            os.path.basename(set_directory),
        )
    if manifest["yaml_note_appended"]:
        append_yaml_note(
            original_path(manifest["yaml_note_appended"]),
            [f"Standardization undone. Manifest: {manifest_path}"],
        )

    rename_files_two_phase(
        set_directory, [(e["new"], e["old"]) for e in manifest["file_renames"]]
    )
    manifest["status"] = "undone"
    write_yaml(manifest_path, manifest)
    logger.info(f"Undone: {manifest_path}")
    click.echo("   Undo completed.")


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================


@click.command()
@click.option(
    "-set_dir",
    "--data_files_set_directory",
    "data_files_set_directory",
    default=None,
    callback=directory.must_exist,
    help="Path to the raw data files set directory.",
)
@click.option(
    "-apply",
    "--apply",
    "apply",
    is_flag=True,
    default=False,
    help="Rename for real (after a final confirmation). Default: dry run.",
)
@click.option(
    "-undo",
    "--undo_manifest",
    "undo_manifest",
    default=None,
    type=click.Path(exists=True, dir_okay=False),
    help="Reverse a previous '--apply' using its manifest.",
)
@click.option(
    "-defaults",
    "--use_defaults",
    "use_defaults",
    is_flag=True,
    default=False,
    help="No questions: keep constants, add nothing, keep directory name.",
)
@click.option(
    "-several_max",
    "--several_max",
    "several_max",
    default=5,
    show_default=True,
    type=click.IntRange(min=2),
    help=(
        "Max distinct values for the 'several' class (else 'varying'). "
        "Exponential-format parameters always use 2."
    ),
)
@click.option(
    "-preview",
    "--preview_count",
    "preview_count",
    default=10,
    show_default=True,
    type=click.IntRange(min=0),
    help="Number of renames shown in the preview (0: all).",
)
@click.option(
    "-plan_out",
    "--plan_output_path",
    "plan_output_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Write the full planned mapping (YAML) during a dry run.",
)
@click.option(
    "-manifest_dir",
    "--manifest_directory",
    "manifest_directory",
    default=None,
    callback=directory.can_create,
    help="Where manifests are written. Default: parent of the set directory.",
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
    default=None,
    callback=directory.can_create,
    help="Directory for log files.",
)
@click.option(
    "-log_name",
    "--log_filename",
    "log_filename",
    default=None,
    callback=validate_log_filename,
    help="Specific name for the script's log file.",
)
def main(
    data_files_set_directory,
    apply,
    undo_manifest,
    use_defaults,
    several_max,
    preview_count,
    plan_output_path,
    manifest_directory,
    enable_logging,
    logs_directory,
    log_filename,
):
    if (data_files_set_directory is None) == (undo_manifest is None):
        raise click.UsageError("Pass exactly one of '-set_dir' or '-undo'.")

    reference = data_files_set_directory or os.path.abspath(undo_manifest)
    if logs_directory is None:
        logs_directory = os.path.join(os.path.dirname(reference.rstrip(os.sep)), "logs")
    if enable_logging:
        os.makedirs(logs_directory, exist_ok=True)
    logger = LoggingWrapper(logs_directory, log_filename, enable_logging)
    logger.initiate_script_logging()

    applying = False
    try:
        if undo_manifest:
            applying = True
            undo_from_manifest(os.path.abspath(undo_manifest), logger)
            return

        set_directory = data_files_set_directory.rstrip(os.sep)
        click.echo(
            f"   -- Standardizing filenames of '{set_directory}'"
            f"{'' if apply else ' (dry run)'}"
        )
        plan, groups = plan_standardization(
            set_directory, several_max, use_defaults, logger
        )
        print_plan_preview(plan, groups, preview_count)

        if not plan.file_renames and not plan.new_set_directory:
            click.echo("\n   Nothing to do: the filenames are already standardized.")
            return

        if not apply:
            if plan_output_path:
                write_yaml(plan_output_path, plan_to_dictionary(plan, status="planned"))
                click.echo(f"\n   Full plan written to '{plan_output_path}'.")
            click.echo(
                "\n   Dry run: nothing was changed. Re-run with -apply to rename."
            )
            return

        if not click.confirm(
            f"\n   Rename {len(plan.file_renames)} files"
            f"{' and the set directory' if plan.new_set_directory else ''} now?",
            default=False,
        ):
            click.echo("   Nothing changed.")
            return
        applying = True
        manifest_path = apply_plan(
            plan,
            manifest_directory or os.path.dirname(set_directory),
            logger,
        )
        click.echo(f"\n   Done. Manifest (for -undo): {manifest_path}")

    except StandardizationError as error:
        logger.error(str(error))
        click.echo(f"\n   STOPPED: {error}", file=sys.stderr)
        if not applying:
            click.echo("   Nothing was changed.", file=sys.stderr)
        sys.exit(1)
    finally:
        logger.terminate_script_logging()


if __name__ == "__main__":
    main()
