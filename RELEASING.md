# Releasing magna

This page is for maintainers. It explains what version numbers mean, how changes reach `main`
and how to publish a release.

## Version numbers

Versions have the form `0.MINOR.PATCH` until the interface is declared stable at 1.0.

| Release | Contains | Example |
|---|---|---|
| Patch, such as `0.1.1` | Bug fixes and documentation corrections, with no change to the interface or requirements | a wrong result corrected |
| Minor, such as `0.2.0` | New features, interface changes, or changed requirements such as a higher minimum Python version | Python 3.11 required |

A version labels a release, not a pull request. Merged pull requests collect on `main`
under "Unreleased" in [CHANGELOG.md](CHANGELOG.md), and only a release pull request changes the version.
The version appears in two files, which must agree: `version` in `pyproject.toml`
and `PACKAGE_VERSION` in `core/magna/file_api.py`.

## How the repository is protected

- `main` accepts changes only through pull requests, which are squash-merged after CI passes.
  GitHub deletes the merged branch.
- Ordinary pushes cannot create branches or tags. Contributors work in their forks;
  maintainers push branches and tags with the repository's deploy key.

A deploy key is an SSH key that GitHub accepts for this repository only.
Create one with a passphrase on each computer you publish from, add its public half under
Settings → Deploy keys with write access, and give it a host alias in `~/.ssh/config`:

```text
Host github-magna
  HostName github.com
  User git
  IdentityFile ~/.ssh/deploy_magna
  IdentitiesOnly yes
  AddKeysToAgent no
  UseKeychain no
```

SSH then asks for the passphrase on every push to `git@github-magna:itpyi/magna.git`.
Keep the key out of `ssh-agent`, and do not store this address as a remote,
so that nothing is pushed with it by accident. (`UseKeychain` is a macOS option; omit it elsewhere.)

## Changing main as a maintainer

Work in a clone of `https://github.com/itpyi/magna.git` on a short-lived branch:

```sh
git switch main
git pull --ff-only
git switch -c short-description
# change, build and test as in CONTRIBUTING.md
git add -A
git commit
git push git@github-magna:itpyi/magna.git short-description
```

Open the pull request on GitHub, wait for CI and squash-merge it. Then update the clone;
a squash merge creates a new commit, so the local branch is removed with `-D`:

```sh
git switch main
git pull --ff-only
git branch -D short-description
```

## Publishing a release

1. In a release pull request, set the new version in `pyproject.toml` and `core/magna/file_api.py`,
   and in `CHANGELOG.md` turn "Unreleased" into the version and date and start a new empty "Unreleased" section.
2. Merge it after CI passes.
3. Tag the merge commit and push the tag:

   ```sh
   git switch main
   git pull --ff-only
   git tag -a v0.2.0 -m "magna 0.2.0"
   git push git@github-magna:itpyi/magna.git v0.2.0
   ```

4. On GitHub, open Releases → Draft a new release, choose the tag and paste its `CHANGELOG.md` section as the notes.
   Leave the release label at None; Pre-release is only for test versions published before a real release.
   GitHub marks the newest ordinary release as Latest.

magna is not yet published on PyPI. The [installation guide](docs/software/install.md#install-a-released-version)
shows how users install a release from its tag.
