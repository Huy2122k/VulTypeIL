import os
import re
import time

import pandas as pd
import requests
from tqdm import tqdm

# ============================================
# CONFIG
# ============================================

GITHUB_TOKEN = ""   # <-- optional: put your token here
OUTPUT_DIR = "incremental_tasks"
NUM_TASKS = 5

train_path = "dataset/train.xlsx"
valid_path = "dataset/valid.xlsx"
test_path  = "dataset/test.xlsx"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================
# GITHUB API HELPER
# ============================================

def extract_repo_info(url):
    """
    Extract 'user', 'repo', 'sha' from commit URL
    Example: https://github.com/user/repo/commit/abcd1234
    """
    pattern = r"github\.com/([^/]+)/([^/]+)/commit/([0-9a-fA-F]+)"
    m = re.search(pattern, url)
    if not m:
        return None, None, None
    return m.group(1), m.group(2), m.group(3)


def get_commit_time(user, repo, sha, session, cache):
    """
    Query GitHub API to get commit timestamp.
    Cached so repeated SHAs don't cause extra requests.
    """
    if sha in cache:
        return cache[sha]

    url = f"https://api.github.com/repos/{user}/{repo}/commits/{sha}"

    headers = {}
    if GITHUB_TOKEN:
        headers["Authorization"] = f"Bearer {GITHUB_TOKEN}"

    while True:
        r = session.get(url, headers=headers)
        if r.status_code == 200:
            data = r.json()
            commit_time = data["commit"]["author"]["date"]
            cache[sha] = commit_time
            return commit_time

        elif r.status_code == 403:
            # rate limit → sleep
            print("Hit rate limit → waiting 60s")
            time.sleep(60)
        else:
            print("Error", r.status_code, r.text)
            cache[sha] = None
            return None


# ============================================
# LOAD ALL DATA
# ============================================

def load_all_data():
    train = pd.read_excel(train_path)
    valid = pd.read_excel(valid_path)
    test  = pd.read_excel(test_path)

    train["source"] = "train"
    valid["source"] = "valid"
    test["source"]  = "test"

    all_data = pd.concat([train, valid, test], ignore_index=True)
    return all_data


# ============================================
# MAIN PROCESS
# ============================================

def process_and_split():
    session = requests.Session()
    cache = {}

    df = load_all_data()
    df["commit_time"] = None

    print("Processing commit timestamps...")
    for i in tqdm(range(len(df))):
        url = df.loc[i, "git_url"]
        user, repo, sha = extract_repo_info(url)
        if user is None:
            df.loc[i, "commit_time"] = None
            continue
        ts = get_commit_time(user, repo, sha, session, cache)
        df.loc[i, "commit_time"] = ts

    print("Sorting by time...")
    df = df.sort_values("commit_time").reset_index(drop=True)

    print("Assigning tasks...")
    df["task"] = pd.qcut(df.index, NUM_TASKS, labels=False) + 1

    print("Saving results into 5 tasks...")
    for task_id in range(1, NUM_TASKS + 1):
        sub = df[df["task"] == task_id]

        train_sub = sub[sub["source"] == "train"]
        valid_sub = sub[sub["source"] == "valid"]
        test_sub  = sub[sub["source"] == "test"]

        train_sub.to_excel(f"{OUTPUT_DIR}/task{task_id}_train.xlsx", index=False)
        valid_sub.to_excel(f"{OUTPUT_DIR}/task{task_id}_valid.xlsx", index=False)
        test_sub.to_excel(f"{OUTPUT_DIR}/task{task_id}_test.xlsx", index=False)

        print(f"Saved task {task_id}: {len(sub)} samples")

    print("\n📌 DONE! Dataset has been split into 5 tasks EXACTLY following the paper.")


# ============================================
# RUN
# ============================================

if __name__ == "__main__":
    process_and_split()
