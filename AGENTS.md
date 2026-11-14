# User-established project rules

## Git identity and attribution

- Create commits using the user's configured Git name and email. Current identity: `posticaianmarius <posticaianmarius@gmail.com>`.
- Push to GitHub using the user's authenticated account. Git commit identity and the account authenticating a push are separate; both must belong to the user.
- Do not use an agent or bot identity as a commit author, committer, or account authenticating a push.
- Do not add agent or AI co-author trailers, signatures, or attribution to commit messages.
- If the user's GitHub authentication is unavailable or belongs to another account, resolve that before pushing; do not substitute an agent account.
- This identity rule applies whenever a commit or push is otherwise authorized; it is not a standing instruction to commit or push every change.
- Every Git push requires the user's explicit confirmation for that push. Permission to commit is not permission to push. Prepare and describe the commits and destination before requesting confirmation; do not push until the user confirms.

## Code formatting

Use compact, single-line code formatting by default. Keep statements, expressions, declarations, function calls, constructor calls, and argument lists on one line. Do not split arguments across multiple lines unless the user explicitly asks for multiline formatting or the code would otherwise be invalid. Do not place a space between control-flow keywords and the opening parenthesis. Write if(condition), for(...), foreach(...), while(...), and switch(...). Place spaces around operators. Write value * 2f, value - 1f, a + b, a / b, a == b, and a = b instead of removing the spaces. Place spaces around increment and decrement operators. Write i ++ and i -- instead of i++ and i--. Avoid unnecessary blank lines and vertical spacing. Prefer concise, consistent, and easy-to-scan formatting across all generated code.
