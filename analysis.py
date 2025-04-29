# dreamer_action_analysis/main.py
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from collections import Counter
import embodied
from sklearn.preprocessing import StandardScaler
from statsmodels.nonparametric.smoothers_lowess import lowess
import statsmodels.formula.api as smf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.utils import shuffle
import json

# === AGENT LOADING ===
def load_dreamer_agent(logdir):
    config = embodied.Config(
        'dreamerv3/configs/atari.yaml',
        'dreamerv3/configs/atari_enduro.yaml',
    )
    config = embodied.Config(config)
    config = config.update({'logdir': logdir})
    agent = embodied.Agent(config.obs_space, config.act_space, config)
    agent.load(logdir)
    return agent

# === DATA PREPARATION ===
def prepare_dreamer_input(npz_path):
    data = np.load(npz_path)
    image_seq = data["image"]
    actions = data["action"]
    grayscale_seq = np.mean(image_seq, axis=-1, keepdims=True).astype(np.uint8)
    return {
        'image': grayscale_seq,
        'actions': np.argmax(actions, axis=-1)
    }

# === FEATURE EXTRACTION ===
def extract_from_dreamer(agent, obs_seq):
    action_logits = []
    latent_feats = []
    for obs in obs_seq:
        out = agent.policy({'image': obs[None]})
        action_logits.append(out['logits'][0])
        latent_feats.append(agent._state.deter.copy())
    return np.stack(action_logits), np.stack(latent_feats)

# === ANALYSIS TOOLS ===
def plot_action_distributions(human_actions, model_actions, action_names):
    human_count = Counter(human_actions)
    model_count = Counter(model_actions)
    indices = range(len(action_names))
    plt.bar(indices, [human_count.get(i, 0) for i in indices], alpha=0.5, label="Human")
    plt.bar(indices, [model_count.get(i, 0) for i in indices], alpha=0.5, label="Model")
    plt.xticks(indices, action_names, rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.title("Action distribution: Human vs Model")
    plt.savefig("results/action_distribution.png")
    plt.close()

def compare_values_by_human_action(human_actions, action_values, action_names):
    data = []
    for i, ha in enumerate(human_actions):
        for a in range(len(action_names)):
            data.append({
                'human_action': action_names[ha],
                'model_action': action_names[a],
                'value': action_values[i][a]
            })
    df = pd.DataFrame(data)
    sns.barplot(data=df, x='model_action', y='value', hue='human_action')
    plt.xticks(rotation=45)
    plt.title("Action Value Distribution by Human Action")
    plt.tight_layout()
    plt.savefig("results/action_values_by_human_action.png")
    plt.close()

def smooth_and_scale(values):
    smoothed = lowess(values, np.arange(len(values)), frac=0.005, return_sorted=False)
    return StandardScaler().fit_transform(smoothed.reshape(-1, 1)).flatten()

def run_statistical_test(human_actions, action_values, action_names):
    records = []
    for i in range(len(human_actions)):
        for j, name in enumerate(action_names):
            records.append({
                'dqn_action': name,
                'human_action': action_names[human_actions[i]],
                'value': action_values[i][j]
            })
    df = pd.DataFrame(records)
    model = smf.ols('value ~ C(dqn_action) + C(human_action) + C(dqn_action):C(human_action)', data=df).fit()
    with open("results/statistical_summary.txt", "w") as f:
        f.write(model.summary().as_text())

def run_lasso_decoding(features, actions):
    selected = np.isin(actions, [2, 3, 4, 5, 6, 7, 8])
    binary_labels = np.array([1 if a in [2, 4, 5, 7] else 0 for a in actions[selected]])
    X = features[selected]
    pca = PCA(n_components=100)
    X_pca = pca.fit_transform(X)
    clf = LogisticRegression(penalty='l1', solver='saga', max_iter=10000)
    acc = cross_val_score(clf, X_pca, binary_labels, cv=5).mean()
    null_scores = []
    for _ in range(100):
        permuted = shuffle(binary_labels, random_state=0)
        score = cross_val_score(clf, X_pca, permuted, cv=5).mean()
        null_scores.append(score)
    result = {
        "LASSO_accuracy": acc,
        "Max_null_accuracy": float(np.max(null_scores)),
        "Significant": acc > np.max(null_scores)
    }
    with open("results/lasso_decoding_result.json", "w") as f:
        json.dump(result, f, indent=2)

# === MAIN EXECUTION ===
LOGDIR = "./logdir/YOUR_MODEL_HERE"
DATA_DIR = "./behavioral_data_2kframe"
ACTION_NAMES = ["NOOP", "FIRE", "RIGHT", "LEFT", "DOWN", "DOWNRIGHT", "DOWNLEFT", "RIGHTFIRE", "LEFTFIRE"]

os.makedirs("results", exist_ok=True)
agent = load_dreamer_agent(LOGDIR)

for subdir, _, files in os.walk(DATA_DIR):
    for file in files:
        if file.endswith(".npz"):
            npz_path = os.path.join(subdir, file)
            print(f"Processing {npz_path}...")

            data = prepare_dreamer_input(npz_path)
            model_logits, latent_feats = extract_from_dreamer(agent, data['image'])
            model_actions = np.argmax(model_logits, axis=1)
            human_actions = data['actions']

            # Figure A
            plot_action_distributions(human_actions, model_actions, ACTION_NAMES)

            # Figure B
            action_probs = model_logits - np.mean(model_logits, axis=1, keepdims=True)
            compare_values_by_human_action(human_actions, action_probs, ACTION_NAMES)

            # Statistical analysis
            run_statistical_test(human_actions, action_probs, ACTION_NAMES)

            # LASSO decoding test
            run_lasso_decoding(latent_feats, human_actions)

            break
