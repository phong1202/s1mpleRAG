# Phase 3: a default user and conversation history

> **For whoever executes this:** do the tasks in order, each one by TDD (red test → code → green test). Do not commit on your own; when told to, group the work into a few large commits. No step calls OpenAI.

**Goal:** conversations are stored on the server, tied to a user, can be reopened and deleted, and old answers keep their citations. For now every request belongs to a pre-created admin user, until there is a login.

**Source:** [`docs/be-conversations-handoff.md`](../be-conversations-handoff.md), written by the FE session from the decisions settled with the repo owner. The contract in section 3.3 of that document is what the FE is building against, so this plan **does not change the contract**. Wherever a change is unavoidable, that document is updated first.

**Architecture:** the current user comes from a dependency, `get_current_user`. In `AUTH_MODE=auto_admin` mode (the default, and the only value for now), the dependency returns the admin user with a fixed id. When a login arrives, only the body of this dependency changes. Conversations and messages are tied to a user; another user's data returns 404 as if it did not exist.

## Global Constraints

- The request and response shapes of `POST /query` do not change; all of its existing tests stay green.
- A question that fails still does not go into the history (as in R4).
- Citations are stored exactly as `POST /query` returned them (`CitationOut`, including `quote`), not as references to chunks: reloading the corpus gives every chunk a new id, whereas the documents and page numbers stay the same.
- Documents are still shared by every user (section 3.4 of the handoff document).
- `auto_admin` amounts to no authentication: log a WARNING at startup, and the README says clearly that the API must not be exposed to an outside network.
- No change to `web/`.

## Out of scope

As in section 5 of the handoff document: login, passwords, tokens, permissions beyond `is_admin`; separating documents per user; renaming, searching and sharing conversations; cancelling the request when the client disconnects.

## Files

| File | Work |
| --- | --- |
| `app/models/user.py`, `app/models/conversation.py` | `User`; `Conversation.user_id`, `title`, `updated_at`; `Message.citations` |
| `alembic/versions/…_users_and_conversation_owners.py` | the `users` table, seed the admin, new columns, backfill old conversations to the admin, index `(user_id, updated_at desc)` |
| `app/config.py`, `.env.example` | `AUTH_MODE` |
| `app/auth.py` | new: the admin's fixed id, `get_current_user` |
| `app/repositories/conversation_repository.py` | filter by user; title, `updated_at`, citations; list, read, delete |
| `app/schemas/conversation.py` | new: `UserOut`, `ConversationSummary`, `ConversationDetail`, `MessageOut` |
| `app/controllers/conversation_controller.py` | new: `GET /me`, `GET /conversations`, `GET /conversations/{id}`, `DELETE /conversations/{id}` |
| `app/services/query_service.py`, `app/controllers/query_controller.py` | `POST /query` receives the user, checks the owner, writes the title and citations |
| `app/app.py` | WARNING at startup when `auto_admin` |
| `README.md` | API table, limitations section |
| the matching tests | |

---

### Task 1: the user and automatic authentication

- The `users` table as in section 3.1 of the handoff document. The migration seeds the admin with the fixed id `00000000-0000-4000-8000-000000000001`, username `admin`, display name `Admin`, `is_admin = true`.
- `AUTH_MODE: Literal["auto_admin"] = "auto_admin"`.
- `get_current_user` returns the admin; if the admin is not in the DB (not migrated yet), 500 with a clear message, without creating it.
- `GET /me`.
- [x] **Failing tests:** `GET /me` returns the right admin; with the admin deleted from the DB, 500 with a message; starting the app with `auto_admin` logs one WARNING line.
- [x] **Implement, tests green.**

### Task 2: conversations tied to a user, with a title and citations

- `conversations.user_id` (not null, old conversations backfilled to the admin in the same migration), `title` (the first question, cut to 120 characters), `updated_at`; `messages.citations` (jsonb, null).
- `POST /query`: a new conversation carries the current user's `user_id`; another user's `conversation_id` → 404 `Conversation not found`; the first turn sets `title`; every turn updates `updated_at` and stores the answer's `citations` (`[]` when there are no citations).
- [x] **Failing tests:** a new conversation belongs to the admin; the first turn sets the title, later turns do not change it; `updated_at` increases after every turn; the assistant message's `citations` equal exactly the `citations` the response returned; another user's conversation → 404; the existing tests of `POST /query` are still green.
- [x] **Implement, tests green; the migration runs on the dev database (with existing conversations) and on an empty database.**

### Task 3: reading and deleting conversations

- `GET /conversations?limit&offset`: the current user's, `updated_at` descending, with `message_count`; `data` is `PaginatedData`, as in `GET /documents`.
- `GET /conversations/{id}`: every message, oldest first.
- `DELETE /conversations/{id}`: deletes the conversation and its messages; traces remain, with `conversation_id` null.
- Another user's conversation: 404 on all three.
- [x] **Failing tests:** as in section 7 of the handoff document, points 3 to 6.
- [x] **Implement, tests green.**

### Task 4: documentation

- [x] `README.md`: endpoint table; the limitations section records the risk of `auto_admin`.
- [x] `docs/be-conversations-handoff.md`: record the "done" status and anything that differs from the proposal (if any).
- [x] `docs/system-design.md`.
- [x] The English version of this plan.

## Definition of Done

Section 7 of the handoff document, all eight points.

## Outcome (2026-10-17)

Built exactly to the contract in the handoff document, with nothing changed. All 8 points in section 7 have tests. 525 tests green, `ruff` clean.

- **Migration `a2c10a08e2a2`** has been run on the dev database. The dev database had no conversations yet, so the part that assigns old conversations to the admin was tried separately: downgrade the migration, insert a sample conversation with two messages, run the migration again. Result: the conversation belongs to the admin, its title is the first question, and the old messages' citations are empty. The sample data has been deleted.
- **`updated_at` uses `clock_timestamp()`, not `now()`:** `now()` is the time the transaction started, the same for every write in one transaction.
- **Departures from the plan:** one migration shared by Tasks 1 and 2, so that the database runs it only once.
