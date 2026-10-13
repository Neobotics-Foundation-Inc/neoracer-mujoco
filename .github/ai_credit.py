"""
Find and remove the lines that credit an AI tool in a commit message or a pull
request description.

People may credit an AI tool in their own commits and pull requests. The commit
on main names only people: a pull request is squash-merged with its title and
description as the commit message, and the AI credit workflow removes these
lines from the description first.

A line credits an AI tool if it is one of these:
- a Co-authored-by trailer whose email belongs to an AI tool, for example
  `Co-authored-by: Claude <noreply@anthropic.com>`;
- another trailer that names an AI tool, such as `Assisted-by: Copilot`;
- a footer such as "Generated with Claude Code" or "Made with Cursor";
- a link to an AI session, such as `https://claude.ai/code/session_...`.

A trailer of a person is kept, even if the person's name is Claude.

Usage, from the workflow:
    python3 .github/ai_credit.py clean < text > cleaned
    python3 .github/ai_credit.py check < text    # exit 1 if a line credits a tool
"""

import re
import sys

# A word that names an AI tool. A name alone does not decide (a person may be
# called Claude or Devin); the email does.
AI_WORD = re.compile(
    r"\b(claude|anthropic|copilot|cursor|codex|chatgpt|openai|gpt[-\w.]*|gemini|jules"
    r"|aider|devin|windsurf|codeium|cody|tabnine|codewhisperer|amazon q|kiro|cline"
    r"|openhands|junie|augment|sweep|qwen|deepseek|grok|llm|ai)\b",
    re.IGNORECASE,
)

# Email domains that only AI tools use in Co-authored-by trailers.
AI_DOMAINS = frozenset(
    {
        "anthropic.com",
        "openai.com",
        "cursor.com",
        "cursor.sh",
        "aider.chat",
        "cognition.ai",
        "codeium.com",
        "windsurf.com",
        "tabnine.com",
        "augmentcode.com",
        "all-hands.dev",
    }
)

TRAILER = re.compile(
    r"^\s*co-authored-by:\s*(?P<name>[^<]*?)\s*<(?P<email>[^>]*)>\s*$", re.IGNORECASE
)
# A GitHub noreply address: 123+login@users.noreply.github.com or login@...
GITHUB_NOREPLY = re.compile(
    r"^(\d+\+)?(?P<login>[^@]+)@users\.noreply\.github\.com$", re.IGNORECASE
)
# Trailers that some tools use instead of Co-authored-by.
OTHER_TRAILER = re.compile(
    r"^\s*(assisted|generated|co-developed|ai-assisted)-by:", re.IGNORECASE
)
FOOTER = re.compile(
    r"^[\W_]*(generated|created|made|written|built|authored|co-authored)\s+(with|by|using|in)\b",
    re.IGNORECASE,
)
SESSION_LINK = re.compile(
    r"^\s*([\w-]+:\s*)?<?https?://(claude\.ai/code|chatgpt\.com/codex|cursor\.com/agents"
    r"|app\.devin\.ai/sessions|jules\.google\.com)\S*\s*$",
    re.IGNORECASE,
)
SEPARATOR = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$")


def ai_email(email: str) -> bool:
    """True if the email belongs to an AI tool, not to a person."""
    email = email.strip().lower()
    if noreply := GITHUB_NOREPLY.match(email):
        login = noreply["login"]
        # A GitHub App or the Copilot account. A person's own login is kept.
        return login == "copilot" or (
            login.endswith("[bot]") and AI_WORD.search(login) is not None
        )
    return email.rpartition("@")[2] in AI_DOMAINS


def credits_ai(line: str) -> bool:
    """True if the line credits an AI tool."""
    if trailer := TRAILER.match(line):
        return ai_email(trailer["email"])
    if OTHER_TRAILER.match(line) or FOOTER.match(line):
        return AI_WORD.search(line) is not None
    return SESSION_LINK.match(line) is not None


def clean(text: str) -> str:
    """The text without the lines that credit an AI tool, and without the
    separators and blank lines that those lines leave at the end."""
    lines = [line for line in text.splitlines() if not credits_ai(line)]
    if len(lines) == len(text.splitlines()):
        return text
    while lines and (not lines[-1].strip() or SEPARATOR.match(lines[-1])):
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def main(argv: list[str]) -> int:
    text = sys.stdin.read()
    match argv:
        case ["clean"]:
            sys.stdout.write(clean(text))
            return 0
        case ["check"]:
            found = [line for line in text.splitlines() if credits_ai(line)]
            for line in found:
                print(f"credits an AI tool: {line.strip()}")
            return 1 if found else 0
        case _:
            print(__doc__, file=sys.stderr)
            return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
