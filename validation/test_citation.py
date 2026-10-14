"""
.github/citation.py credits every person in the git history once, the most
commits first, and leaves out AI tools and bots.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

_GITHUB = Path(__file__).parents[1] / ".github"
sys.path.insert(0, str(_GITHUB))  # citation.py imports ai_credit.py next to it
_SPEC = importlib.util.spec_from_file_location("citation", _GITHUB / "citation.py")
assert _SPEC is not None and _SPEC.loader is not None
citation = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(citation)

PEOPLE = (
    "Bassel Mabsout <7851037+bmabsout@users.noreply.github.com>",
    "Claude Martin <claude.martin@gmail.com>",
    "anshm <anshm@umich.edu>",
)
NOT_PEOPLE = (
    "Claude <noreply@anthropic.com>",
    "Claude Sonnet 5 <noreply@anthropic.com>",
    "Copilot Autofix powered by AI <175728472+Copilot@users.noreply.github.com>",
    "dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
    "GitHub <noreply@github.com>",
)


@pytest.mark.parametrize("identity", PEOPLE)
def test_a_person_is_an_author(identity):
    assert citation.is_person(identity)


@pytest.mark.parametrize("identity", NOT_PEOPLE)
def test_an_ai_tool_or_a_bot_is_not_an_author(identity):
    assert not citation.is_person(identity)


def test_authors_are_ranked_by_credited_commits_then_by_first_credit():
    a, b, c = (f"{n} <{n.lower()}@example.com>" for n in ("Ann Lee", "Bo Kim", "Cy Wu"))
    commits = [[c], [a, *NOT_PEOPLE], [b], [a, NOT_PEOPLE[0]], [b], [c]]
    assert citation.ranked_people(commits) == [c, a, b]


def test_a_person_counts_once_per_commit():
    """Ann is the author and a co-author of one commit; Bo authored two."""
    ann, bo = "Ann Lee <ann@example.com>", "Bo Kim <bo@example.com>"
    assert citation.ranked_people([[ann, ann], [bo], [bo]]) == [bo, ann]


def test_two_people_with_one_name_are_two_authors():
    """.mailmap keeps two Alex Smiths apart, so the citation does too: each
    gets an entry with the email, as CFF requires different entries. A name
    that nobody else has gets no email."""
    alex1, alex2 = "Alex Smith <a@one.org>", "Alex Smith <a@two.org>"
    sam = "Sam Roe <sam@example.com>"
    people = citation.ranked_people([[alex1], [alex2], [sam], [sam], [sam]])
    assert people == [sam, alex1, alex2]
    authors = citation.citation(people).split("authors:\n")[1]
    assert authors.split("  - ")[1:] == [
        'given-names: "Sam"\n    family-names: "Roe"\n',
        'given-names: "Alex"\n    family-names: "Smith"\n    email: "a@one.org"\n',
        'given-names: "Alex"\n    family-names: "Smith"\n    email: "a@two.org"\n',
    ]


def test_names_that_differ_only_in_spaces_get_the_email():
    """cff_person splits a name at any run of spaces, so these two names give
    the same entry and need the email to differ."""
    people = ["Alex Smith <a@one.org>", "Alex  Smith <a@two.org>"]
    text = citation.citation(people)
    assert 'email: "a@one.org"' in text and 'email: "a@two.org"' in text


def test_a_credit_without_an_email_is_left_out(tmp_path, monkeypatch):
    """git check-mailmap takes only "Name <email>". A trailer without an
    email, such as "Co-authored-by: Claude", is not an author."""
    monkeypatch.chdir(tmp_path)
    git = ["git", "-c", "user.name=Ann Lee", "-c", "user.email=ann@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    message = "Add a file\n\nCo-authored-by: Claude\n"
    commit = [*git, "-c", "commit.gpgsign=false", "commit", "-q", "--allow-empty"]
    subprocess.run([*commit, "-m", message], check=True)
    assert citation.ranked_people(citation.identities()) == [
        "Ann Lee <ann@example.com>"
    ]


@pytest.mark.parametrize(
    ("name", "entry"),
    [
        ("anshm", '  - alias: "anshm"\n'),
        ("Bassel Mabsout", '  - given-names: "Bassel"\n    family-names: "Mabsout"\n'),
        (
            "Abdallah El Hamawi",
            '  - given-names: "Abdallah"\n    family-names: "El Hamawi"\n',
        ),
        (
            "Mary Ann van der Berg",
            '  - given-names: "Mary Ann"\n    family-names: "van der Berg"\n',
        ),
    ],
)
def test_the_family_name_keeps_its_particles(name, entry):
    assert citation.cff_person(name) == entry


def test_the_file_lists_the_top_five_authors():
    """people come most commits first, so the authors are the first five."""
    people = [f"Person {c} <{c}@example.com>" for c in "ABCDEFG"]
    text = citation.citation(people)
    assert text.count("given-names") == 5
    assert 'family-names: "E"' in text and 'family-names: "F"' not in text


def test_the_file_lists_each_author_once():
    text = citation.citation(
        ["Bassel Mabsout <b@example.com>", "anshm <a@example.com>"]
    )
    assert text.startswith("# Written by .github/citation.py")
    assert text.count("family-names") == 1 and text.count("alias") == 1
    assert "cff-version: 1.2.0\n" in text and "authors:\n" in text
