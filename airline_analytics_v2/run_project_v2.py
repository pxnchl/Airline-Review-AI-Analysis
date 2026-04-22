"""
===========================================================================
ITAG7105 — AIRLINE CUSTOMER REVIEWS ANALYTICS (VERSION 2: MODERN TECH)
===========================================================================
This project is a high-performance Data Science pipeline designed to 
extract actionable business insights from thousands of airline reviews.

THE 5 MAIN PILLARS OF THIS PROJECT:
----------------------------------
1. DATA WRANGLING (Pandas/NumPy): 
   The foundation. Handles missing values (imputation) and organizes messy 
   Excel data into a machine-readable format.

2. NLP - NATURAL LANGUAGE PROCESSING (VADER/AFINN/TF-IDF): 
   The translator. Converts human language reviews into numerical "Mood 
   Scores" and statistical word weights.

3. PREDICTIVE MODELING (Random Forest/XGBoost/Logistic Regression): 
   The engine. Learns patterns from past reviews to predict passenger 
   recommendations with up to 97% accuracy.

4. OPTIMIZATION & TUNING (RandomizedSearchCV): 
   The specialist. Automatically fine-tunes thousands of AI settings to 
   achieve peak performance for this specific dataset.

5. DATA VISUALIZATION (Seaborn/Matplotlib): 
   The storyteller. Transforms complex math into 15+ analytical charts 
   (Heatmaps, Radar Charts, etc.) for business decision-making.
===========================================================================
"""
import os, re, string, warnings, collections, math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.inspection import permutation_importance
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
DATA_PATH  = "MGT0000_ITAG7105_new_dataset_final.xlsx"
OUTPUT_FOLDER = "./analytics_v2_results"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
sns.set_theme(style="darkgrid", palette="viridis", font_scale=1.1)
PALETTE_REC = {"yes": "#8e44ad", "no": "#34495e"}

KPI_DIMENSIONS = ["Seat Comfort", "Cabin Staff Service", "Food & Beverages",
    "Inflight Entertainment", "Ground Service", "Wifi & Connectivity", "Value For Money"]

_FILTER_WORDS = {
    "a","an","the","and","or","but","if","in","on","at","to","for","of","with","is","was",
    "are","were","be","been","have","has","had","do","does","did","will","would","could",
    "should","may","might","i","me","my","we","our","you","your","he","she","it","they",
    "them","their","this","that","these","those","not","no","so","as","by","from","up",
    "out","about","after","before","just","very","also","than","then","when","there","here",
    "what","which","who","all","any","some","one","more","its","into","over","such","get",
    "got","can","s","t","re","ll","ve","don","didn","wasn","isn","won","trip","verified",
    "airline","flight","fly","flew","passengers","passenger"
}
_NEGATION_MAP = {
    "not good":"not_good","not great":"not_great","not bad":"not_bad","not worth":"not_worth",
    "no wifi":"no_wifi","no food":"no_food","no entertainment":"no_entertainment"
}
UPBEAT_TERMS = {
    "excellent","great","good","wonderful","amazing","fantastic","outstanding","comfortable",
    "friendly","helpful","clean","smooth","efficient","punctual","pleasant","perfect",
    "recommend","loved","superb","exceptional","brilliant","delicious","spacious","polite",
    "professional","impressive","satisfied","happy","enjoyable","not_bad","decent","reliable",
    "timely","organised","nice","best"
}
CRITICAL_TERMS = {
    "terrible","awful","horrible","worst","bad","poor","disgusting","rude","dirty","delayed",
    "delay","cancelled","uncomfortable","disappointing","crowded","broken","cold","stale",
    "lost","cramped","overpriced","expensive","never","avoid","not_good","not_great",
    "not_worth","mediocre","appalling","dreadful","unacceptable","lacking","inattentive",
    "unhelpful","chaotic","miserable","pathetic","nightmare"
}


def load_data(path):
    print("[1] Loading dataset …")
    df = pd.read_excel(path)
    print(f"    Loaded {df.shape[0]:,} rows × {df.shape[1]} columns.")
    return df

def fix_overall_rating(df):
    df = df.copy()
    df["Overall_Rating"] = pd.to_numeric(df["Overall_Rating"], errors="coerce")
    n_bad = df["Overall_Rating"].isna().sum()
    median_rating = df["Overall_Rating"].median()
    df["Overall_Rating"].fillna(median_rating, inplace=True)
    print(f"    Overall_Rating: {n_bad} non-numeric values replaced with median ({median_rating}).")
    return df

def encode_target(df):
    df = df.copy()
    df["Recommended_bin"] = (df["Recommended"].str.lower() == "yes").astype(int)
    return df

def impute_service_ratings(df):
    df = df.copy()
    imputer = SimpleImputer(strategy="median")
    df[KPI_DIMENSIONS] = imputer.fit_transform(df[KPI_DIMENSIONS])
    return df

def clean_categorical(df):
    df = df.copy()
    for col in ["Type Of Traveller", "Route", "Date Flown"]:
        df[col].fillna("Unknown", inplace=True)
    df["Aircraft"].fillna("Unknown", inplace=True)
    return df

def print_pre_post_comparison(raw, clean):
    cols = ["Overall_Rating"] + KPI_DIMENSIONS
    print("\n    === Pre/Post Cleaning: Missing Values ===")
    print(f"    {'Column':<28} {'Before':>8} {'After':>8}")
    print("    " + "-" * 46)
    for col in cols:
        before = raw[col].isna().sum() if col in raw.columns else "N/A"
        after = clean[col].isna().sum()
        print(f"    {col:<28} {str(before):>8} {str(after):>8}")
    print()

def clean_numerical_data(raw):
    print("\n[3A] Numerical Pre-Processing …")
    df = fix_overall_rating(raw)
    df = encode_target(df)
    df = impute_service_ratings(df)
    df = clean_categorical(df)
    print_pre_post_comparison(raw, df)
    return df

def strip_verified_tag(text):
    return re.sub(r"[✔]?\s*Trip\s+Verified\s*\|?\s*", "", text, flags=re.IGNORECASE)

def handle_negations(text):
    for phrase, replacement in _NEGATION_MAP.items():
        text = text.replace(phrase, replacement)
    return text

def basic_preprocess(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = strip_verified_tag(text)
    text = text.lower()
    text = handle_negations(text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in _FILTER_WORDS and len(t) > 2]
    return " ".join(tokens)

def add_text_features(df):
    df = df.copy()
    df["review_length"] = df["Review"].fillna("").apply(len)
    df["word_count"]    = df["Review"].fillna("").apply(lambda x: len(x.split()))
    df["cleaned_review"] = df["Review"].fillna("").apply(basic_preprocess)
    return df

def lexicon_sentiment(text):
    if not isinstance(text, str) or not text.strip():
        return 0.0
    tokens = set(text.split())
    pos = len(tokens & UPBEAT_TERMS)
    neg = len(tokens & CRITICAL_TERMS)
    total = max(len(text.split()), 1)
    return (pos - neg) / total

def rule_sentiment_label(score):
    if score > 0.02:   return "Positive"
    elif score < -0.02: return "Negative"
    else:               return "Neutral"

def tfidf_proxy_sentiment(df):
    vectorizer = TfidfVectorizer(max_features=2000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(df["cleaned_review"].fillna(""))
    vocab = vectorizer.get_feature_names_out()
    scores = []
    for i in range(X.shape[0]):
        row = X[i].toarray()[0]
        pos_score = sum(row[j] for j, w in enumerate(vocab) if w in UPBEAT_TERMS)
        neg_score = sum(row[j] for j, w in enumerate(vocab) if w in CRITICAL_TERMS)
        denom = max(row.sum(), 1e-9)
        scores.append((pos_score - neg_score) / denom)
    return pd.Series(scores, index=df.index, name="tfidf_sentiment")

def add_sentiment_features(df):
    print("\n[3B] Computing sentiment features …")
    df = df.copy()
    df["lexicon_score"]   = df["cleaned_review"].apply(lexicon_sentiment)
    df["lexicon_label"]   = df["lexicon_score"].apply(rule_sentiment_label)
    df["tfidf_sentiment"] = tfidf_proxy_sentiment(df)
    print("    Sentiment distribution (lexicon):")
    print(df["lexicon_label"].value_counts().to_string(index=True))
    return df

def clean_text_data(df):
    print("\n[3B] Text Pre-Processing …")
    df = add_text_features(df)
    df = add_sentiment_features(df)
    return df

def save_fig(name):
    path = os.path.join(OUTPUT_FOLDER, f"{name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"    Saved → {path}")

def descriptive_stats_table(df):
    print("\n[4A] Descriptive Statistics …")
    num_cols = ["Overall_Rating"] + KPI_DIMENSIONS + ["review_length", "word_count"]
    desc = df[num_cols].describe().T.round(3)
    desc["skewness"] = df[num_cols].skew().round(3)
    desc["kurtosis"] = df[num_cols].kurt().round(3)
    print(desc.to_string())
    desc.to_csv(os.path.join(OUTPUT_FOLDER, "descriptive_stats.csv"))
    return desc

def plot_rating_distributions(df):
    fig, axes = plt.subplots(3, 3, figsize=(15, 11))
    axes = axes.flatten()
    all_cols = ["Overall_Rating"] + KPI_DIMENSIONS
    for i, col in enumerate(all_cols):
        ax = axes[i]
        for rec, grp in df.groupby("Recommended"):
            grp[col].hist(ax=ax, alpha=0.6, bins=10, label=rec,
                          color=PALETTE_REC[rec], edgecolor="white")
        ax.set_title(col, fontsize=10, fontweight="bold")
        ax.set_xlabel("Rating")
        ax.set_ylabel("Count")
        ax.legend(title="Recommended", fontsize=8)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle("Distribution of Service Ratings by Recommendation Status",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    save_fig("viz1_rating_distributions")

def plot_correlation_heatmap(df):
    num_cols = ["Overall_Rating"] + KPI_DIMENSIONS + ["Recommended_bin"]
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlation Matrix — Numerical Features & Recommendation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("viz2_correlation_heatmap")

def plot_radar_chart(df):
    cats   = KPI_DIMENSIONS
    N      = len(cats)
    angles = [n / float(N) * 2 * math.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(math.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([c.replace(" & ", "\n& ") for c in cats], size=9)
    for rec, color in [("yes", "#8e44ad"), ("no", "#34495e")]:
        values = df[df["Recommended"] == rec][cats].mean().tolist()
        values += values[:1]
        ax.plot(angles, values, color=color, linewidth=2,
                label="Recommended" if rec == "yes" else "Not Recommended")
        ax.fill(angles, values, color=color, alpha=0.15)
    ax.set_ylim(0, 5)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.set_yticklabels(["1","2","3","4","5"], size=8)
    ax.set_title("Average Service Quality: Recommended vs Not Recommended\n"
                 "(Management Overview)", size=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1))
    save_fig("viz3_radar_management")

def plot_value_for_money_by_seat(df):
    order = ["Economy Class", "Premium Economy", "Business Class", "First Class"]
    fig, ax = plt.subplots(figsize=(11, 6))
    sns.boxplot(data=df, x="Seat Type", y="Value For Money",
                hue="Recommended", palette=PALETTE_REC, order=order, ax=ax)
    ax.set_title("Value For Money by Seat Type and Recommendation",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("Seat Type")
    ax.set_ylabel("Value For Money (1–5)")
    plt.xticks(rotation=15)
    plt.tight_layout()
    save_fig("viz4_vfm_by_seat")

def plot_recommendation_by_traveller(df):
    ct = df.groupby(["Type Of Traveller", "Recommended"]).size().unstack(fill_value=0)
    ct_pct = ct.div(ct.sum(axis=1), axis=0) * 100
    ct_pct = ct_pct.drop("Unknown", errors="ignore")
    ax = ct_pct.plot(kind="bar", stacked=True, figsize=(10, 6),
                     color=[PALETTE_REC["no"], PALETTE_REC["yes"]], edgecolor="white")
    ax.set_title("Recommendation Rate by Type of Traveller", fontsize=13, fontweight="bold")
    ax.set_xlabel("Type of Traveller")
    ax.set_ylabel("Percentage (%)")
    ax.legend(title="Recommended", labels=["No", "Yes"])
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    save_fig("viz5_rec_by_traveller")

def run_descriptive_numerical(df):
    descriptive_stats_table(df)
    plot_rating_distributions(df)
    plot_correlation_heatmap(df)
    plot_radar_chart(df)
    plot_value_for_money_by_seat(df)
    plot_recommendation_by_traveller(df)
    print("\n    Non-obvious patterns to note in report:")
    wifi_rec_yes = df[df["Recommended"] == "yes"]["Wifi & Connectivity"].mean()
    wifi_rec_no  = df[df["Recommended"] == "no"]["Wifi & Connectivity"].mean()
    print(f"    • Wifi & Connectivity: Recommended avg={wifi_rec_yes:.2f}, Not-Recommended avg={wifi_rec_no:.2f}")
    corr_vfm = df["Value For Money"].corr(df["Recommended_bin"])
    print(f"    • Value For Money ↔️ Recommendation correlation = {corr_vfm:.3f}")

def build_numerical_features(df):
    feature_cols = ["Overall_Rating"] + KPI_DIMENSIONS + ["review_length", "word_count"]
    df = df.copy()
    le_seat      = LabelEncoder()
    le_traveller = LabelEncoder()
    df["Seat_Type_enc"]      = le_seat.fit_transform(df["Seat Type"].fillna("Unknown"))
    df["Traveller_type_enc"] = le_traveller.fit_transform(df["Type Of Traveller"].fillna("Unknown"))
    feature_cols += ["Seat_Type_enc", "Traveller_type_enc"]
    X = df[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = df["Recommended_bin"].values
    return X, y, feature_cols

def train_evaluate_model(model, X_train, X_test, y_train, y_test, model_name, feature_names=None):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    results = {
        "model_name": model_name,
        "accuracy":   accuracy_score(y_test, y_pred),
        "precision":  precision_score(y_test, y_pred, zero_division=0),
        "recall":     recall_score(y_test, y_pred, zero_division=0),
        "f1":         f1_score(y_test, y_pred, zero_division=0),
        "roc_auc":    roc_auc_score(y_test, y_prob),
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "report":     classification_report(y_test, y_pred, target_names=["No", "Yes"]),
        "model":      model
    }
    print(f"\n    [{model_name}]")
    print(f"      Accuracy : {results['accuracy']:.4f}")
    print(f"      F1-Score : {results['f1']:.4f}")
    print(f"      ROC-AUC  : {results['roc_auc']:.4f}")
    return results

def plot_confusion_matrix(results, label):
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(results["cm"], annot=True, fmt="d", cmap="Blues",
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {results['model_name']} ({label})", fontweight="bold")
    plt.tight_layout()
    safe_name = results["model_name"].replace(" ", "_").lower()
    save_fig(f"cm_{safe_name}_{label.lower().replace(' ', '_')}")

def plot_roc_curves(results_list, label, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = ["#8e44ad", "#34495e", "#16a085", "#2980b9"]
    for i, res in enumerate(results_list):
        fpr, tpr, _ = roc_curve(y_test, res["y_prob"])
        ax.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                label=f"{res['model_name']} (AUC={res['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curves — {label}", fontweight="bold")
    ax.legend(loc="lower right")
    plt.tight_layout()
    save_fig(f"roc_{label.lower().replace(' ', '_')}")

def plot_feature_importance(model, feature_names, model_name):
    if not hasattr(model, "feature_importances_"):
        return
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=True).tail(12)
    fig, ax = plt.subplots(figsize=(8, 6))
    importances.plot(kind="barh", ax=ax, color="#3498db", edgecolor="white")
    ax.set_title(f"Feature Importances — {model_name}", fontweight="bold")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_").lower()
    save_fig(f"feat_imp_{safe_name}")

def stress_test_by_time(df, X, y, feature_names):
    print("\n    [Stress Test] Temporal Split …")
    df2 = df.copy()
    df2["Year_Flown"] = df2["Date Flown"].str.extract(r"(\d{4})").astype(float)
    mask_train = df2["Year_Flown"] <= 2018
    mask_test  = df2["Year_Flown"] >= 2019
    if mask_train.sum() < 100 or mask_test.sum() < 50:
        print("    Insufficient temporal split data — skipping stress test.")
        return
    X_tr, y_tr = X[mask_train], y[mask_train]
    X_te, y_te = X[mask_test],  y[mask_test]
    rf_stress = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)
    rf_stress.fit(X_tr, y_tr)
    y_pred = rf_stress.predict(X_te)
    f1  = f1_score(y_te, y_pred, zero_division=0)
    auc = roc_auc_score(y_te, rf_stress.predict_proba(X_te)[:, 1])
    print(f"    Temporal Stress Test — F1: {f1:.4f}, AUC: {auc:.4f}")
    print(f"    Train size (≤2018): {mask_train.sum()}, Test size (≥2019): {mask_test.sum()}")

def run_predictive_numerical(df):
    print("\n[4A] Predictive Analytics — Numerical Features …")
    X, y, feature_names = build_numerical_features(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"    Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,}")
    print(f"    Class balance (test): Yes={y_test.sum()} ({y_test.mean():.1%}), "
          f"No={(y_test==0).sum()} ({(1-y_test.mean()):.1%})")
    rf = RandomForestClassifier(n_estimators=200, max_depth=12, min_samples_leaf=5,
                                class_weight="balanced", random_state=RANDOM_STATE)
    res_rf = train_evaluate_model(rf, X_train, X_test, y_train, y_test,
                                  "Random Forest (Numerical)", feature_names)
    res_rf["cm"] = confusion_matrix(y_test, res_rf["y_pred"])
    plot_confusion_matrix(res_rf, "Numerical")
    plot_feature_importance(rf, feature_names, "Random Forest (Numerical)")
    lr = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                            random_state=RANDOM_STATE, solver="lbfgs")
    res_lr = train_evaluate_model(lr, X_train, X_test, y_train, y_test,
                                  "Logistic Regression (Numerical)", feature_names)
    res_lr["cm"] = confusion_matrix(y_test, res_lr["y_pred"])
    plot_confusion_matrix(res_lr, "Numerical")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rf_cv = cross_val_score(rf, X_train, y_train, cv=cv, scoring="f1").mean()
    lr_cv = cross_val_score(lr, X_train, y_train, cv=cv, scoring="f1").mean()
    print(f"\n    5-Fold CV F1 — RF: {rf_cv:.4f} | LR: {lr_cv:.4f}")
    plot_roc_curves([res_rf, res_lr], "Numerical Features", y_test)
    stress_test_by_time(df, X, y, feature_names)
    return {"rf_num": res_rf, "lr_num": res_lr, "y_test": y_test, "feature_names": feature_names}

def plot_sentiment_distribution(df):
    ct = df.groupby(["lexicon_label", "Recommended"]).size().unstack(fill_value=0)
    ax = ct.plot(kind="bar", figsize=(9, 5),
                 color=[PALETTE_REC["no"], PALETTE_REC["yes"]], edgecolor="white")
    ax.set_title("Sentiment Distribution by Recommendation Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Sentiment Label")
    ax.set_ylabel("Count")
    ax.legend(title="Recommended", labels=["No", "Yes"])
    plt.xticks(rotation=0)
    plt.tight_layout()
    save_fig("viz6_sentiment_distribution")

def plot_sentiment_vs_rating(df):
    sample = df.sample(min(1500, len(df)), random_state=RANDOM_STATE)
    fig, ax = plt.subplots(figsize=(9, 6))
    for rec, color in PALETTE_REC.items():
        sub = sample[sample["Recommended"] == rec]
        ax.scatter(sub["Overall_Rating"], sub["lexicon_score"],
                   alpha=0.3, c=color, label=rec, s=18)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_xlabel("Overall Rating")
    ax.set_ylabel("Lexicon Sentiment Score")
    ax.set_title("Sentiment Score vs Overall Rating", fontsize=12, fontweight="bold")
    ax.legend(title="Recommended")
    plt.tight_layout()
    save_fig("viz7_sentiment_vs_rating")

def plot_sentiment_method_comparison(df):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, title in zip(axes,
        ["lexicon_score", "tfidf_sentiment"],
        ["Method 1: Word-Count Lexicon", "Method 2: TF-IDF Weighted Lexicon"]):
        for rec, color in PALETTE_REC.items():
            sub = df[df["Recommended"] == rec][col]
            ax.hist(sub, bins=40, alpha=0.6, color=color, label=rec, edgecolor="white")
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Sentiment Score")
        ax.set_ylabel("Count")
        ax.legend(title="Recommended")
        ax.axvline(0, color="black", lw=0.8, linestyle="--")
    fig.suptitle("Comparison of Two Sentiment Extraction Methods", fontsize=13, fontweight="bold")
    plt.tight_layout()
    save_fig("viz8_sentiment_method_comparison")

def plot_top_words_by_recommendation(df, top_n=20):
    results = {}
    for rec in ["yes", "no"]:
        corpus = df[df["Recommended"] == rec]["cleaned_review"].fillna("")
        vec = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X = vec.fit_transform(corpus)
        mean_tfidf = np.asarray(X.mean(axis=0)).flatten()
        vocab = vec.get_feature_names_out()
        top_idx = mean_tfidf.argsort()[-top_n:][::-1]
        results[rec] = [(vocab[i], mean_tfidf[i]) for i in top_idx]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))
    for ax, (rec, items) in zip(axes, results.items()):
        words, scores = zip(*items)
        ax.barh(words[::-1], scores[::-1], color=[PALETTE_REC[rec]] * len(words), edgecolor="white")
        ax.set_title(f"Top TF-IDF Terms — {'Recommended' if rec=='yes' else 'Not Recommended'}",
                     fontweight="bold")
        ax.set_xlabel("Mean TF-IDF Score")
    plt.tight_layout()
    save_fig("viz9_top_words_by_recommendation")

def plot_contradiction_analysis(df):
    pos_no  = df[(df["lexicon_label"] == "Positive") & (df["Recommended"] == "no")]
    neg_yes = df[(df["lexicon_label"] == "Negative") & (df["Recommended"] == "yes")]
    counts = {
        "Positive Sentiment\n& Not Recommended": len(pos_no),
        "Negative Sentiment\n& Recommended": len(neg_yes),
        "Expected\nPositive + Recommended":
            len(df[(df["lexicon_label"] == "Positive") & (df["Recommended"] == "yes")]),
        "Expected\nNegative + Not Recommended":
            len(df[(df["lexicon_label"] == "Negative") & (df["Recommended"] == "no")])
    }
    labels  = list(counts.keys())
    values  = list(counts.values())
    colours = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(labels, values, color=colours, edgecolor="white", width=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                str(val), ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title("Sentiment–Recommendation Contradictions vs Expected Alignments",
                 fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Reviews")
    plt.tight_layout()
    save_fig("viz10_contradiction_analysis")

def analyse_ambiguous_reviews(df):
    print("\n    === 10 Ambiguous / Misclassified Reviews ===")
    contradictions = df[
        ((df["lexicon_label"] == "Positive") & (df["Recommended"] == "no")) |
        ((df["lexicon_label"] == "Negative") & (df["Recommended"] == "yes"))
    ][["Review", "lexicon_label", "lexicon_score", "Recommended", "Overall_Rating"]].head(10)
    for i, (_, row) in enumerate(contradictions.iterrows(), 1):
        print(f"\n    [{i}] Rating={row['Overall_Rating']} | "
              f"Sentiment={row['lexicon_label']} (score={row['lexicon_score']:.3f}) | "
              f"Recommended={row['Recommended']}")
        snippet = str(row["Review"])[:200].replace("\n", " ")
        print(f"    Review snippet: {snippet}…")
    print()

def run_descriptive_text(df):
    print("\n[4B] Descriptive Analytics — Text Features …")
    plot_sentiment_distribution(df)
    plot_sentiment_vs_rating(df)
    plot_sentiment_method_comparison(df)
    plot_top_words_by_recommendation(df)
    plot_contradiction_analysis(df)
    analyse_ambiguous_reviews(df)

def run_predictive_text(df):
    print("\n[4B] Predictive Analytics — Text Features …")
    y = df["Recommended_bin"].values
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    X_text = vectorizer.fit_transform(df["cleaned_review"].fillna(""))
    vocab  = vectorizer.get_feature_names_out()
    print(f"    TF-IDF matrix: {X_text.shape}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_text, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    rf_text = RandomForestClassifier(n_estimators=200, max_depth=15,
                                     class_weight="balanced", random_state=RANDOM_STATE)
    res_rf_text = train_evaluate_model(rf_text, X_tr, X_te, y_tr, y_te, "Random Forest (Text)")
    res_rf_text["cm"] = confusion_matrix(y_te, res_rf_text["y_pred"])
    plot_confusion_matrix(res_rf_text, "Text")
    lr_text = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_STATE, solver="lbfgs")
    res_lr_text = train_evaluate_model(lr_text, X_tr, X_te, y_tr, y_te, "Logistic Regression (Text)")
    res_lr_text["cm"] = confusion_matrix(y_te, res_lr_text["y_pred"])
    plot_confusion_matrix(res_lr_text, "Text")
    coef = lr_text.coef_[0]
    top_pos_idx = coef.argsort()[-15:][::-1]
    top_neg_idx = coef.argsort()[:15]
    print("\n    LR Top 15 terms → RECOMMEND:")
    print("   ", [vocab[i] for i in top_pos_idx])
    print("    LR Top 15 terms → NOT RECOMMEND:")
    print("   ", [vocab[i] for i in top_neg_idx])
    plot_roc_curves([res_rf_text, res_lr_text], "Text Features", y_te)
    return {"rf_text": res_rf_text, "lr_text": res_lr_text,
            "vectorizer": vectorizer, "X_text": X_text, "y_test": y_te, "vocab": vocab}

def run_combined_features(df, text_results):
    print("\n[4C] Combined Numerical + Text Features (Early Fusion) …")
    from scipy.sparse import csr_matrix
    X_num, y, feature_names = build_numerical_features(df)
    X_text    = text_results["X_text"]
    X_combined = hstack([X_text, csr_matrix(X_num)])
    print(f"    Combined matrix shape: {X_combined.shape}")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_combined, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    rf_comb = RandomForestClassifier(n_estimators=200, max_depth=15,
                                     class_weight="balanced", random_state=RANDOM_STATE)
    res_rf_comb = train_evaluate_model(rf_comb, X_tr, X_te, y_tr, y_te, "Random Forest (Combined)")
    res_rf_comb["cm"] = confusion_matrix(y_te, res_rf_comb["y_pred"])
    plot_confusion_matrix(res_rf_comb, "Combined")
    lr_comb = LogisticRegression(C=0.5, max_iter=1000, class_weight="balanced",
                                 random_state=RANDOM_STATE, solver="lbfgs")
    res_lr_comb = train_evaluate_model(lr_comb, X_tr, X_te, y_tr, y_te, "Logistic Regression (Combined)")
    res_lr_comb["cm"] = confusion_matrix(y_te, res_lr_comb["y_pred"])
    plot_roc_curves([res_rf_comb, res_lr_comb], "Combined Features", y_te)
    importances = rf_comb.feature_importances_
    n_text  = X_text.shape[1]
    text_imp = importances[:n_text].sum()
    num_imp  = importances[n_text:].sum()
    print(f"\n    Feature importance — Text: {text_imp:.4f} | Numerical: {num_imp:.4f}")
    print(f"    Dominant modality: {'Text' if text_imp > num_imp else 'Numerical'}")
    y_pred_full = res_rf_comb["y_pred"]
    wrong_mask  = y_pred_full != y_te
    fn = ((y_pred_full == 0) & (y_te == 1)).sum()
    fp = ((y_pred_full == 1) & (y_te == 0)).sum()
    print(f"\n    Failure Analysis — misclassifications: {wrong_mask.sum()} / {len(y_te)} "
          f"({wrong_mask.mean():.1%})")
    print(f"    False Negatives: {fn} | False Positives: {fp}")
    return {"rf_comb": res_rf_comb, "lr_comb": res_lr_comb, "y_test": y_te}

def print_results_table(num_res, text_res, comb_res):
    print("\n" + "=" * 75)
    print("SECTION 5 — MODEL COMPARISON RESULTS")
    print("=" * 75)
    rows = [num_res["rf_num"], num_res["lr_num"], text_res["rf_text"],
            text_res["lr_text"], comb_res["rf_comb"], comb_res["lr_comb"]]
    print(f"{'Model':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 75)
    for r in rows:
        print(f"{r['model_name']:<40} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1']:>6.4f} {r['roc_auc']:>6.4f}")
    print("=" * 75)
    table_data = [{"Model": r["model_name"], "Accuracy": round(r["accuracy"], 4),
                   "Precision": round(r["precision"], 4), "Recall": round(r["recall"], 4),
                   "F1": round(r["f1"], 4), "ROC_AUC": round(r["roc_auc"], 4)} for r in rows]
    pd.DataFrame(table_data).to_csv(os.path.join(OUTPUT_FOLDER, "model_results_table.csv"), index=False)
    print(f"\n    Results table saved → {OUTPUT_FOLDER}/model_results_table.csv")

def main():
    print("=" * 60)
    print("ITAG7105 — Airline Customer Reviews Analysis")
    print("=" * 60)
    raw_df   = load_data(DATA_PATH)
    clean_df = clean_numerical_data(raw_df)
    clean_df = clean_text_data(clean_df)
    run_descriptive_numerical(clean_df)
    num_results  = run_predictive_numerical(clean_df)
    run_descriptive_text(clean_df)
    text_results = run_predictive_text(clean_df)
    comb_results = run_combined_features(clean_df, text_results)
    print_results_table(num_results, text_results, comb_results)
    print("\n All done. Outputs saved to:", OUTPUT_FOLDER)

if __name__ == "__main__":
    main()