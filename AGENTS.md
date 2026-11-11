# Instructions for coding agents

This file is for AI coding agents that work on magna for a contributor.
[CONTRIBUTING.md](CONTRIBUTING.md) is the full guide; read it first.
The points below are the ones that most often cost a round of review.

## Before you write code

- Check whether the change needs an agreed plan
  (CONTRIBUTING.md, "Agree on a plan before a larger change").
  If it does and no maintainer has accepted a plan on an issue, draft the issue for the person you work for
  instead of the pull request.
- For an optimization, measure the simplest existing way to get the same result before you write code.
  The change is worth making only if it clearly beats that.
- Read [How it works](docs/software/architecture/index.md) for the part of the kernel you will change.

## While you work

- Keep to the design rules in CONTRIBUTING.md. Algorithms go in the C++ kernel, the kernel holds no
  protocol-specific code, and decoders see only reported measurement data.
- Change only what the purpose needs. Do not reformat, rename or reorganize other code,
  and do not add options, fallbacks or compatibility layers that nobody asked for.
- Rebuild after every change under `core/magna/native/`, then run the tests and the docs check
  (CONTRIBUTING.md, "Build and test").
- A bug fix comes with a test that fails without the fix. Run it once without the fix to see it fail.
- When the user manual says what a feature supports, make that true for every supported task, or name the exceptions.
- Write magna in lowercase, also at the start of a sentence.
- Never commit `.venv`, `build/`, or the built extension and `native-build.json` in `core/magna/`.

## The pull request description

Fill in every section of the template, briefly. CI reads parts of it.

- **Scope.** Tick every part the diff touches and list each other changed file with a reason.
  The *Review summary* check compares this with the diff.
- **Results of a seeded run.** Answer "unchanged" only after checking. The *Seeded results* check
  runs fixed seeded tasks on `main` and on the pull request, and warns when that answer is wrong.
  You can run the same comparison against a checkout of `main` built with your own Python:

  ```sh
  git worktree add --detach ../magna-main main
  MAGNA_PYTHON="$PWD/.venv/bin/python" ../magna-main/scripts/python.sh ../magna-main/scripts/build_native_core.py
  ./scripts/python.sh scripts/seeded_results.py --base ../magna-main
  ```

  Counters and fingerprints may change while the results stay the same; say which fields moved.
- **Changelog.** One line under "Unreleased" that says what a user notices, not how the code does it.
  Leave out internal names, counters and "results unchanged". For example, write
  "Sampling is faster when many draws are rejected." rather than
  "Skip output queries for zero accepted mass, reducing `resources.queries`."
- **Measurements.** Name the commit, the machine and the command for every number,
  and compare with the simplest existing way you measured first.
- **AI tools.** Say which tools helped. The person who opens the pull request answers for every line.

Open, update or comment on a pull request or issue only when the person you work for asks you to.
