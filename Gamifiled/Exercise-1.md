# 🎮 MLOps Quest — Session 1 Guide

> **Duration:** 1 hours
> **Goal:** Pass Exercise 1 and earn your first 10+ XP
> **Prerequisite:** You must have completed `Getting-Started.md`. If you haven't, stop here and do that first.

---

## 🎯 What you'll do today

1. Keep a watch on the Game board 
2. Train a baseline regression model for QuickFoods delivery time
3. Push your model to your fork
4. Earn XP: **10 base + up to 9 bonus = up to 19 XP (XP Experience Points) **
5. Watch your name appear on the leaderboard projected in class

---

## 📺 The three URLs you need (instructor will put on the whiteboard)

| What | Example URL |
|---|---|
| 🏆 **Leaderboard** (watch on your second monitor or phone) | `https://urmsandeep.github.io/MLOps-Lab-Gamified/` |
| 📊 **MLflow server** (for tracking your experiments) | `https://xxxx-xx.ngrok-free.app` *(unique per class — get it from whiteboard)* |
| 📂 **Class repo** (for reference) | `https://github.com/urmsandeep/MLOps-Lab-Gamified` |

---

# MINUTE 0–10 — Get situated

## Step 1: Confirm you can see the leaderboard

Open the leaderboard URL in your browser. You should see a dark arcade-style page showing students on a podium. If your name (GitHub username) is in the full leaderboard list, you're registered and ready.

**If your name is NOT in the list:**
- Check with the instructor — your Pull Request might not be merged yet
- In the meantime, continue with Step 2 below. You can push code even before you're on the roster; the grader will pick you up as soon as you're added.

## Step 2: Open your terminal and go to the repo

```bash
cd ~/mlops-course/MLOps-Lab-Gamified
```

## Step 3: Sync your fork with the latest class repo

The instructor may have updated the class repo since you forked. Pull in the latest:

```bash
# Add the instructor's repo as "upstream" (do this ONCE, first session only)
git remote add upstream https://github.com/urmsandeep/MLOps-Lab-Gamified.git

# Pull latest changes
git fetch upstream
git merge upstream/main
```

If you get a merge conflict error, ask a classmate for help or raise your hand. For most people, this will complete cleanly.

Push the synced state to your own fork:

```bash
git push origin main
```

---

# MINUTE 10–30 — Exercise 1: Train the baseline

## Step 4: Read the exercise brief

```bash
cd exercises/ex01
cat README.md
```

Skim this for 2 minutes. Key facts:
- **Goal**: produce `exercises/ex01/model.pkl` that predicts QuickFoods delivery time
- **Pass condition**: RMSE on a secret held-out set must be below 15.0
- You cannot see the held-out data, so don't try to cheat — the test is fair but opaque

## Step 5: Install dependencies (if you haven't already)

```bash
cd starter
pip install -r requirements.txt
```

Wait for install to finish. Ignore yellow warnings. Red errors mean trouble.

## Step 6: Run the starter code AS-IS to see what happens

```bash
python train.py
```

Expected output (last few lines):

```
Local validation RMSE: 3.13
Grader threshold: 15.0 (you need to be below this on held-out data)
✓ Model saved to exercises/ex01/model.pkl
```

**If you see RMSE around 3.0–4.0**: congrats, the starter code already produces a passing model. You can push it as-is for instant XP.

**If you see an error**: read the error message. Most common problem: forgot to install dependencies. Run `pip install -r requirements.txt` again.

## Step 7 (optional but recommended): Improve the model

The starter uses `LinearRegression`. Try `RandomForestRegressor` for better scores — same interface, usually much lower RMSE:

```python
# In train.py, change this line:
# OLD:
# model = LinearRegression()

# NEW:
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
```

Re-run:
```bash
python train.py
```

You should see RMSE drop to around 2.0 or less. Better numbers ≠ more XP in this exercise (pass is binary), but you'll feel good. Save it for hyperparameter tuning glory in Exercise 5.

## Step 8: Verify the model was saved

```bash
# Go up one level
cd ..
ls -la model.pkl
```

You should see the file, a few KB in size.

---

# MINUTE 30–40 — Push and earn XP

## Step 9: Commit and push your model

Go back to the repo root:

```bash
# Go back to repo root
cd ~/mlops-course/MLOps-Lab-Gamified

# Check what you're about to commit
git status
```

You should see `exercises/ex01/model.pkl` listed as untracked.

```bash
git add exercises/ex01/model.pkl
git commit -m "Ex01: baseline model"
git push origin main
```

If asked for credentials:
- Username: your GitHub username
- Password: paste your Personal Access Token (the one you saved in Pre-Class Part 3.2)

Expected final line of output:
```
   XXXXXXX..YYYYYYY  main -> main
```

## Step 10: WATCH THE LEADERBOARD 👀

Open the leaderboard URL. Within 2 minutes (the grader runs every 2 min), your row will:
- Turn the Ex01 🎯 cell green
- Add 10 XP to your total
- Possibly add bonus XP:
  - **+5 XP** if you're the first person to pass Ex01 today (🩸 First Blood)
  - **+3 XP / +2 XP** if you're 2nd or 3rd (🥈🥉)
  - **+2 XP** if you passed on your very first push (✨ Flawless)

## If your row doesn't turn green within 3 minutes

Don't panic. Try these in order:

1. **Hard refresh the leaderboard**: `Ctrl+Shift+R` (Windows) or `Cmd+Shift+R` (Mac)
2. **Verify your push landed**: go to `https://github.com/YOUR_USERNAME/MLOps-Lab-Gamified/tree/main/exercises/ex01` — do you see `model.pkl`? If no, your push didn't work.
3. **Check the grader logs**: go to `https://github.com/urmsandeep/MLOps-Lab-Gamified/actions` → click the latest run → click `grade` job → look for your username in the logs
4. **Raise your hand**. Ask the instructor to investigate.

---

# MINUTE 40–90 — Keep climbing

## Step 11: Preview what's coming in Exercises 2–10

Now that you're on the board, take 5 min to skim the other exercises so you know what's ahead:

```bash
cd ~/mlops-course/MLOps-Lab-Gamified/exercises
ls
```

You'll see folders for ex02, ex03, etc. (once the instructor adds them — may be session-by-session).

## Step 12: Start Exercise 2 (MLflow tracking)

If the instructor has opened Exercise 2 for this session, read its README and begin. Key thing for Ex2: you need the MLflow tracking URL from the whiteboard.

Configure it in your shell:

**Windows (Git Bash):**
```bash
export MLFLOW_TRACKING_URI="https://xxxx-xx.ngrok-free.app"
```

**Mac/Linux:**
```bash
export MLFLOW_TRACKING_URI="https://xxxx-xx.ngrok-free.app"
```

Then in your Python code:
```python
import mlflow
# The URI is picked up from environment automatically
mlflow.set_experiment("quickfoods-ex02-YOURUSERNAME")
with mlflow.start_run():
    # ... train your model ...
    mlflow.log_metric("rmse", rmse)
    mlflow.sklearn.log_model(model, "model")
```

Replace `YOURUSERNAME` with your actual GitHub username so the instructor can tell whose runs are whose.

---

# MINUTE 90–120 — Wrap up

## Step 13: Before leaving class

- [ ] Screenshot the leaderboard with your row visible — save as your progress proof
- [ ] Commit and push any work-in-progress code to your fork (even if not passing yet — you can continue at home)
- [ ] Note any exercise where you got stuck — ask during Q&A

## Step 14: Homework

- Finish any exercise you didn't complete in class
- The grader runs 24/7, so you can keep pushing and earning XP outside class
- **But remember: First Blood bonuses only fire ONCE per exercise.** If a classmate passes Ex02 at 10 PM tonight, they get First Blood, not you tomorrow.

---

# 🎲 Tips for maximizing XP

## Speed matters, but not as much as you think

Don't panic-push broken code. A failed attempt burns your Flawless bonus (+2 XP). It's usually worth spending 5 extra minutes to get it right first try.

## Read the exercise READMEs fully before starting

Each exercise has a "Hints" section. Reading it saves you 15 min of trial-and-error.

## Help your neighbor, but don't give them your code

You'll both learn more if they struggle with your help than if they copy your push. Plus, if 10 students all submit the exact same model.pkl, the instructor notices.

## Use the class MLflow to compare with classmates

Once in MLflow (Ex02+), you can see everyone's runs side by side. "Why is Arjun's RMSE 1.2 when mine is 3.5?" — go look at his hyperparameters. Learn from the leader.

## Don't get discouraged if you're at the bottom early on

Someone has to be on row #45. The leaderboard rewards **consistent shipping** over the whole course, not just Session 1. The biggest climbers usually win medium-term.

---

# 📞 When to ask for help

## Immediately raise your hand if:

- You can't push code (auth error)
- Your Python environment is broken
- The error message makes no sense to you even after 5 min of reading

## Try to solve yourself first if:

- Your model's RMSE is too high (this is actually a learning opportunity)
- You're unsure which sklearn class to use (Google / docs exist)
- You want to do something fancier than the baseline

## For AI tool users

Copilot / ChatGPT / Claude will all happily help you through any of this. Rules of thumb:

- ✅ OK: "Explain what this error means: `<paste full error>`"
- ✅ OK: "What's the difference between LinearRegression and RandomForest?"
- ✅ OK: "Why is my RMSE not improving?"
- ⚠️ Risky: "Write me code that passes Exercise 1" — you'll learn nothing, and if the AI produces a valid-looking model that happens to underperform on the held-out set, you won't know why it failed
- ❌ Bad idea: Pasting your PAT or any credentials into an AI chat

---

# 🏆 Good luck, operator

Your first push. Your first XP. Your first public portfolio commit showing a real MLOps system.

**GAME ON.** 🚀
