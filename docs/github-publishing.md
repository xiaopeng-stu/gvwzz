# Publishing GVWZZ v1 on GitHub

The package version is `1.0.0`, and the first public release and Git tag are named
`v1`. This guide uses the repository name `gvwzz`. Replace `YOUR-USERNAME` with
your GitHub username or organization. These example URLs are placeholders; the
source package does not claim that the repository already exists.

## 1. Prepare the source

Install Git and sign in to your GitHub account. Extract the source archive, then
open PowerShell in the `gvwzz-v1` directory containing `README.md`.

Run the release checks with the Python interpreter you intend to use:

```powershell
python -m pip install -r requirements-native.txt
python build_native.py
python run_tests.py --require-native --differential
```

Native builds on Windows need Microsoft C++ Build Tools with the Desktop
development with C++ workload. The README covers Linux and macOS setup.

Keep the existing `LICENSE`. Update project ownership or citation information
only with accurate details. The `.gitignore` excludes local results, environments,
caches, and binaries; generated C++ sources belong in the repository.

## 2. Create an empty public repository

Open <https://github.com/new>, select your account, enter `gvwzz`, and choose
**Public**. Do not initialize a README, license, or `.gitignore` on GitHub: these
files already exist in the source tree. Click **Create repository**.

In PowerShell, from the source directory:

```powershell
git init -b main
git add .
git status
git commit -m "Release GVWZZ v1"
git remote add origin https://github.com/YOUR-USERNAME/gvwzz.git
git push -u origin main
```

Review the staged paths printed by `git status` before committing. If Git asks
for an author identity, configure your own name and email, then retry the commit:

```powershell
git config user.name "Your Name"
git config user.email "your-git-email@example.com"
```

Authenticate through the Git credential helper or GitHub CLI when prompted.
GitHub account passwords are not used as HTTPS Git passwords.

After the push, your repository will be available at:

```text
https://github.com/YOUR-USERNAME/gvwzz
```

Open the **Actions** tab to inspect the test workflow. Its matrix checks Python
fallback and native builds on Linux, Windows, and macOS. Investigate failed jobs
before describing a platform as tested.

## 3. Publish the v1 release

After the checks pass, tag the reviewed commit and push the tag:

```powershell
git tag -a v1 -m "GVWZZ v1 (Python package 1.0.0)"
git push origin v1
```

On the repository page, open **Releases**, then **Draft a new release**. Select
the existing `v1` tag, enter **GVWZZ v1** as the title, and use
`docs/release-notes.md` as the description. Attach the supplied
`gvwzz-v1-source.zip` if you want a named download. You can obtain its SHA-256
checksum with `Get-FileHash path/to/gvwzz-v1-source.zip -Algorithm SHA256`.
GitHub also generates source ZIP and tar archives from the tag. Publish the
release after reviewing its files and notes.

The release page will be:

```text
https://github.com/YOUR-USERNAME/gvwzz/releases/tag/v1
```

Keep the `v1` tag on this reviewed commit. Use a new version and tag for later
changes so experimental results can refer to a stable source snapshot.

## Optional: use GitHub CLI

This is an alternative to creating the empty repository in the browser. Start
from a local repository with the initial commit already made, and do not create
a second repository if you have completed the preceding steps:

```powershell
gh auth login
gh repo create YOUR-USERNAME/gvwzz --public --source=. --remote=origin --push
```

After the tests pass and the `v1` tag has been pushed, create a draft release:

```powershell
gh release create v1 --verify-tag --title "GVWZZ v1" --notes-file docs/release-notes.md --draft
```

Open the draft on GitHub, attach the archive and checksum if desired, and publish
it. No automatic release publication is configured in the test workflow.

## Official references

- [Adding locally hosted code to GitHub](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)
- [Managing releases](https://docs.github.com/en/repositories/releasing-projects-on-github/managing-releases-in-a-repository)
- [GitHub CLI: gh repo create](https://cli.github.com/manual/gh_repo_create)
- [GitHub CLI: gh release create](https://cli.github.com/manual/gh_release_create)
