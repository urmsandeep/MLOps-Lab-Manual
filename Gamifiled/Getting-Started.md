# 📋 MLOps Quest — Pre-Class Setup (DO BEFORE SESSION 1)

> **Time required:** 30–45 minutes
> **Difficulty:** Medium — mostly tedious, not hard
> **Deadline:** Finish this BEFORE Session 1 starts. Class time is for learning MLOps, not fighting with GitHub.

---

## 🎯 Why this matters

You will lose XP if you show up without this setup done. Your classmates who came prepared will be passing Exercise 1 while you are still creating a GitHub account. Do NOT wait until class.

By the end of this doc, you will have:

- ✅ A working GitHub account
- ✅ Git installed on your laptop
- ✅ Python 3.10+ installed
- ✅ A successful test push to a repo (proving your setup works)
- ✅ The class repo cloned and ready

---

## 🤖 Note about using AI tools

You can use ChatGPT / Claude / Copilot to help you through this — that's fine, even encouraged. But paste the **actual error message** when asking AI for help. Do not ask "why is git not working" — ask "I got this exact error: `<paste error>`, what does it mean?"

If you copy-paste commands from AI without reading them, you will paste something that breaks your system. Always read before executing.

---

# PART 1 — Install the tools (10 min)

## 1.1 Install Git

**Windows users:**

1. Download Git from: https://git-scm.com/download/win
2. Run the installer. Click "Next" through every screen — defaults are fine.
3. Open **"Git Bash"** from the Start menu (this is your terminal for the whole course)
4. Type this to verify:

```bash
git --version
```

You should see something like `git version 2.43.0`. Any version 2.x is fine.

**Mac users:**

```bash
# Open Terminal app (Cmd+Space, type "Terminal")
git --version
```

If it prompts to install Xcode Command Line Tools, click Install. Wait ~5 min.

**Ubuntu/WSL users:**

```bash
sudo apt update
sudo apt install git -y
git --version
```

## 1.2 Install Python 3.10 or higher

Check if you already have it:

```bash
python --version
# OR
python3 --version
```

If either shows `Python 3.10.x` or higher, you're good. Skip to 1.3.

If not, download from: https://www.python.org/downloads/

During install on Windows: **⚠️ CHECK THE BOX "Add Python to PATH"** — this is the #1 mistake people make.

Verify after install:

```bash
python --version
```

## 1.3 Install pip packages you'll need

```bash
pip install pandas numpy scikit-learn joblib mlflow pyyaml
```

If you get a "command not found: pip" error:
```bash
python -m pip install pandas numpy scikit-learn joblib mlflow pyyaml
```

Wait for the install to finish (2-5 min). Some warnings are OK. Errors are not.

---

# PART 2 — Create a GitHub account (10 min)

## 2.1 Sign up

Go to: https://github.com/signup

- **Email**: Use an email you check regularly. School emails work.
- **Password**: Strong password. Save it in a password manager or write it down safely.
- **Username**: ⚠️ **Choose carefully — this will be PUBLIC and permanent.** Use a professional-looking username:
  - ✅ Good: `arjun-patel`, `priya-ml-dev`, `ravi-codes-1999`
  - ❌ Avoid: `xXGamerBoyXx`, `tempuser123`, `asdfasdf`

  Remember: this will show up on your public leaderboard and your code commits. Future employers might see it. **Choose a username you're OK with having for years.**

## 2.2 Verify your email

GitHub sends a 6-digit code. Enter it. If you don't see it, check spam.

## 2.3 Skip the onboarding questions

GitHub will ask questions like "what's your job", "how many developers on your team". Click "Skip" / "Continue" through all of these — they don't affect anything.

## 2.4 Verify you can log in

Sign out, then sign back in. Make sure your username and password work. **Write your username down somewhere you won't lose it.**

---

# PART 3 — Connect git to GitHub (15 min — the hardest part)

> 💡 **Why this is hard:** GitHub no longer accepts your GitHub password from the command line. You need a special thing called a Personal Access Token (PAT). This confuses everyone. It confused your instructor. You are not alone.

## 3.1 Tell git who you are

Open your terminal (Git Bash on Windows, Terminal on Mac/Linux):

```bash
git config --global user.name "YOUR_GITHUB_USERNAME"
git config --global user.email "YOUR_EMAIL_USED_FOR_GITHUB"
```

Replace `YOUR_GITHUB_USERNAME` with the username you chose in Part 2. Use the **same** email you signed up with.

Verify:

```bash
git config --global user.name
git config --global user.email
```

Both should print what you typed.

## 3.2 Create a Personal Access Token (PAT)

A PAT is like a password, but only for command-line git. You will paste it instead of your real password when GitHub asks.

**Steps:**

1. Log into GitHub in your browser
2. Go to: https://github.com/settings/tokens?type=beta
3. Click the green button **"Generate new token"**
4. Fill in:
   - **Token name**: `mlops-quest-laptop` (any name works, but this is descriptive)
   - **Expiration**: `90 days` (or "Custom" → end of semester)
   - **Repository access**: Select **"All repositories"**
     - (For a paranoid approach, choose "Only select repositories" → `MLOps-Lab-Gamified` after you fork it. But "All repositories" is simpler for a first-time user.)
   - **Repository permissions** (scroll down):
     - **Contents**: `Read and write`
     - **Metadata**: Will auto-check as `Read-only` — leave it
     - **Pull requests**: `Read and write`
     - Leave everything else at "No access"
5. Scroll all the way down. Click green **"Generate token"**
6. ⚠️ **You will see a long string starting with `github_pat_...`**
7. ⚠️ **COPY IT NOW. Paste it into a text file on your laptop.**
8. ⚠️ **This is the ONLY time GitHub will show you this token. If you close the page without copying, you have to generate a new one.**

Save it in a file called `github-token.txt` on your desktop, or in your password manager. **Never share this with anyone or paste it into a chat.**

## 3.3 Test your token

We're going to clone a public repo to test auth works:

```bash
# Go to your home directory
cd ~

# Create a folder for this course
mkdir mlops-course
cd mlops-course
```

Now try to clone something (this is just a test — we'll clone the real repo later):

```bash
git clone https://github.com/urmsandeep/MLOps-Lab-Manual.git
```

This should just work without asking for credentials (public repo, anonymous clone is fine).

If you see `fatal: destination path 'MLOps-Lab-Manual' already exists` — just `cd MLOps-Lab-Manual` and you're fine.

**Now the real test — can you push?** We'll make a tiny change and try to push it back. Since you don't own that repo, this should **fail with a specific error**, which confirms your auth is reaching GitHub correctly.

```bash
cd MLOps-Lab-Manual
echo "test" > _test_file.txt
git add _test_file.txt
git commit -m "test commit"
git push
```

When prompted:
- **Username**: your GitHub username
- **Password**: paste your **PAT** (not your real password) — note: pasting a PAT doesn't show any characters on screen, that's normal

You should see:
```
remote: Permission to urmsandeep/MLOps-Lab-Manual.git denied to YOUR_USERNAME
```

**That error is GOOD.** It means GitHub recognized your token — it just correctly denied you push access to someone else's repo. Your auth works. 🎉

If you instead see `Authentication failed` or `invalid username or token`, something's wrong with your PAT. Re-do section 3.2 carefully.

Cleanup:
```bash
cd ..
rm -rf MLOps-Lab-Manual
```

---

# PART 4 — Fork the class repo and register (5 min)

## 4.1 Fork the repo

1. In your browser (logged in as YOUR account), go to: **https://github.com/urmsandeep/MLOps-Lab-Gamified**
2. Click the **"Fork"** button in the top-right
3. On the "Create a new fork" page, **leave all defaults**. Just click the green **"Create fork"** button
4. After ~3 seconds, you'll land on `https://github.com/YOUR_USERNAME/MLOps-Lab-Gamified` — this is YOUR personal copy

## 4.2 Clone your fork

In your terminal:

```bash
cd ~/mlops-course

# Replace YOUR_USERNAME with your actual GitHub username
git clone https://github.com/YOUR_USERNAME/MLOps-Lab-Gamified.git

cd MLOps-Lab-Gamified
```

If asked for credentials, use your username and paste your PAT as the password.

## 4.3 Register yourself in students.yml

Open `students.yml` in any text editor (Notepad, VS Code, nano, whatever). You'll see:

```yaml
students:
  - urmsandeep          # instructor (for testing)
  # - alice-ml
  # - bob-dev
```

Add a new line with your GitHub username:

```yaml
students:
  - urmsandeep
  - YOUR_USERNAME_HERE
```

⚠️ **Important formatting rules:**
- The line must start with `  - ` (two spaces, then dash, then space)
- NO quotes around your username
- NO trailing spaces after your username
- Use the exact same capitalization as your actual GitHub username

Save the file.

## 4.4 Push your registration

```bash
git add students.yml
git commit -m "Register: YOUR_USERNAME"
git push origin main
```

If asked for credentials:
- Username: your GitHub username
- Password: paste your PAT

Expected output (the last few lines):
```
To https://github.com/YOUR_USERNAME/MLOps-Lab-Gamified.git
   XXXXXXX..YYYYYYY  main -> main
```

## 4.5 Open a Pull Request

1. Go back to your browser
2. Visit: `https://github.com/YOUR_USERNAME/MLOps-Lab-Gamified`
3. You'll see a yellow/blue banner: *"This branch is 1 commit ahead of urmsandeep:main"*
4. Click the **"Contribute"** dropdown → **"Open pull request"**
5. On the next page:
   - Title: `Register: YOUR_USERNAME`
   - Description: `Signing up for the quest.`
6. Click green **"Create pull request"**
7. You'll be redirected to a page showing your PR on the instructor's repo

## 4.6 Wait for instructor approval

Your instructor will merge your PR before Session 1. You'll get a notification email from GitHub.

**You do not need to do anything else.** You're registered.

---

# PART 5 — Verify everything is ready

Before you close this doc, run through this checklist:

- [ ] I can open a terminal and `git --version` shows version 2.x or higher
- [ ] I can run `python --version` and see 3.10 or higher
- [ ] I have a GitHub account with a username I'm OK with long-term
- [ ] I saved my Personal Access Token somewhere safe
- [ ] I successfully did `git push` on a test (even if it failed with "permission denied", that means auth worked)
- [ ] I forked `urmsandeep/MLOps-Lab-Gamified` to MY account
- [ ] I cloned MY fork locally
- [ ] I added my username to `students.yml`
- [ ] I pushed the change to my fork
- [ ] I opened a Pull Request to the instructor's repo
- [ ] I can see my PR at `https://github.com/urmsandeep/MLOps-Lab-Gamified/pulls`

If all boxes are checked, **you are ready for Session 1.**

---

# 🚨 Help! I'm stuck!

**Before asking the instructor, try these in order:**

## Common errors and fixes

| Error message | What it means | Fix |
|---|---|---|
| `git: command not found` | Git isn't installed or not on PATH | Reinstall git, restart terminal |
| `Authentication failed` | Wrong password / bad PAT | Re-generate PAT in Part 3.2 |
| `Permission denied to YOUR_USERNAME` | You're trying to push to a repo you don't own | You forgot to fork — do Part 4.1 |
| `refusing to merge unrelated histories` | You cloned the wrong repo | Delete the folder, start over from 4.2 |
| `Updates were rejected` | Someone else pushed in between | Run `git pull origin main` then `git push` again |
| `pip: command not found` | Python not on PATH | Use `python -m pip install ...` instead |
| `ModuleNotFoundError: No module named 'X'` | You forgot a pip install | Run `pip install X` |

## Debugging workflow

1. **Read the error message slowly.** The answer is almost always in the last 2-3 lines.
2. **Copy the exact error** into Google or an AI chatbot. Do not paraphrase.
3. **Ask a classmate.** Someone else has hit the same issue. Pair up.
4. **Only then ask the instructor** — and when you do, paste:
   - The exact command you ran
   - The exact error you got
   - What you've already tried

This is how real engineers ask for help. Practicing it now makes you better at your job later.

---

# 📞 Contact

Stuck for more than 30 minutes? Ask help of Sandeep with:

- Your GitHub username
- Your OS (Windows/Mac/Linux)
- A screenshot of the error

---

**See you in Session 1. Be ready to ship.** 🚀
