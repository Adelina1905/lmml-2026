# User-established project rules

## Git identity and attribution

- Create commits using the user's configured Git name and email. Current identity: `posticaianmarius <posticaianmarius@gmail.com>`.
- Push to GitHub using the user's authenticated account. Git commit identity and the account authenticating a push are separate; both must belong to the user.
- Do not use an agent or bot identity as a commit author, committer, or account authenticating a push.
- Do not add agent or AI co-author trailers, signatures, or attribution to commit messages.
- If the user's GitHub authentication is unavailable or belongs to another account, resolve that before pushing; do not substitute an agent account.
- This identity rule applies whenever a commit or push is otherwise authorized; it is not a standing instruction to commit or push every change.
- Every Git push requires the user's explicit confirmation for that push. Permission to commit is not permission to push. Prepare and describe the commits and destination before requesting confirmation; do not push until the user confirms.

## Task folders

- When the user provides a new competition task, automatically create its folder at the repository root before starting task work.
- Name the folder `Task name (Country)`, using the task's original title and the country's full English name. Example: `Ngiyakuzwa (South Africa)`.
- Determine the country from the supplied task description. If it is missing or ambiguous, ask for the country instead of guessing.
- Remove characters that Windows does not allow in folder names and trim trailing spaces or periods.
- Reuse an existing matching task folder. Keep that task's input files, code, results, and solution notes inside it.
- Automatically save the complete task statement pasted by the user as `Task.md` inside that folder before starting the analysis. Preserve its original wording, Markdown formatting, requirements, input links, flag format, and scoring details.
- Keep conversation instructions and solution commentary out of `Task.md`. Create `Solving.md` only when the user explicitly requests it.
