# AI Development Rules

These rules apply to the entire repository and to every AI coding agent working on it.

## Language

- Use English for all new or modified repository content, including code comments, documentation, messages, and filenames.
- Do not add Chinese text to the repository. Conversation with the developer may use their preferred language.

## Explicit C++ Types

- Do not use the `auto` keyword in new or modified C++ code.
- Write explicit types for variables, parameters, return types, iterators, and range-based loops.
- This also excludes `decltype(auto)`, `auto` structured bindings, and generic lambda parameters declared with `auto`.
- Prefer readable type aliases when an explicit type would otherwise be unwieldy.

## Scope and Approval

- By default, modify only core implementation and public API code in `src/` and `include/` that is needed for the requested task.
- Changes outside core code require explicit developer approval unless the developer has already requested or authorized those changes.
- In particular, do not independently create documentation, architecture descriptions, development logs, changelog entries, reports, or other supporting artifacts.
- Ask for approval before creating such files. Do not treat them as automatic deliverables of a coding task.
- Running existing builds and tests is allowed; do not add their generated output to version control.
- Keep changes focused. Do not perform repository-wide cleanup of existing language or typing violations unless explicitly requested.
