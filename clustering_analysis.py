"""Fit gym member segments, check their holdout fit, and compare credit policies."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "outputs"
FEATURES = [
    "age", "visits_per_month", "avg_visit_minutes",
    "classes_per_month", "months_member", "monthly_extras_spend",
]
EXTRAS = "monthly_extras_spend"
SEED = 42
TEST_FRACTION = 0.30
K_VALUES = np.arange(1, 9)
N_STARTS = 50

CREDIT_COST = 10
SPEND_THRESHOLD = 20
EXTRA_SPEND_WITH_CREDIT = 25
EXTRAS_MARGIN = 0.60

# Names and descriptions follow the training centers in ascending extras spend.
SEGMENT_DESCRIPTIONS = {
    "New light users": "Short tenure, infrequent visits and little class attendance.",
    "Long-tenured regulars": "Longest tenure and oldest on average, with regular visits.",
    "Frequent long-visit members": "Most visits and longest sessions, with few classes.",
    "Class-focused spenders": "Most classes and the highest extras spending.",
}

NEW_MEMBER = {
    "age": 30, "visits_per_month": 4, "avg_visit_minutes": 42,
    "classes_per_month": 1, "months_member": 5, "monthly_extras_spend": 10,
}


def read_members():
    source = ROOT / "gym_members.csv"
    if not source.is_file():
        raise SystemExit("Place the course-provided gym_members.csv beside this script.")
    members = pd.read_csv(source)
    needed = {"member_id", *FEATURES}
    if not needed.issubset(members.columns):
        raise ValueError(f"Missing input columns: {sorted(needed - set(members.columns))}")
    if len(members) != 330 or members["member_id"].isna().any() or not members["member_id"].is_unique:
        raise ValueError("Expected 330 rows with distinct, nonempty member IDs.")
    members[FEATURES] = members[FEATURES].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(members[FEATURES].to_numpy()).all():
        raise ValueError("The clustering features must contain finite numbers.")
    return members


def elbow_k(spread):
    """Find the largest downward gap from the chord between the two endpoints."""
    scaled = (spread - spread[-1]) / (spread[0] - spread[-1])
    chord = np.linspace(1, 0, len(spread))
    return int(K_VALUES[1:-1][np.argmax((chord - scaled)[1:-1])])


def net_if_credited(spend):
    """Apply the finance assumptions to each member's prior extras spending."""
    extra_margin = round(EXTRA_SPEND_WITH_CREDIT * EXTRAS_MARGIN, 2)
    return np.where(spend < SPEND_THRESHOLD, extra_margin - CREDIT_COST, -CREDIT_COST)


def make_offer_rule(scaler, model, segment_names, credit_segments):
    def segment_and_offer(member_features):
        """Return (segment name, receives credit) for one member's feature mapping."""
        row = pd.DataFrame([member_features], columns=FEATURES)
        cluster = int(model.predict(scaler.transform(row))[0])
        return segment_names[cluster], cluster in credit_segments

    return segment_and_offer


def plot_elbow(curve, chosen_k, test_k, train_count, test_count):
    fig, ax = plt.subplots(figsize=(9, 4.8), constrained_layout=True)
    ax.plot(curve["k"], curve["training_sse"], "o-", color="#294b6e",
            linewidth=2, label=f"Training ({train_count} members)")
    ax.plot(curve["k"], curve["test_sse"], "s-", color="#0b7666",
            linewidth=2, label=f"Test ({test_count} members)")
    ax.axvline(chosen_k, color="#777777", linestyle="--", linewidth=1.2)
    highlighted = curve.loc[curve["k"].eq(chosen_k), "training_sse"].iloc[0]
    label = f"Both curves bend at k = {chosen_k}" if chosen_k == test_k else f"Training choice: k = {chosen_k}"
    ax.annotate(label, xy=(chosen_k, highlighted), xytext=(chosen_k + 0.4, highlighted + 170),
                fontsize=10, color="#333333", arrowprops={"arrowstyle": "-", "color": "#777777"})
    ax.set(xlabel="Number of segments (k)", ylabel="Sum of squared distances (scaled features)",
           title="Within-segment spread on training and test members", ylim=(0, None))
    ax.set_xticks(K_VALUES)
    ax.grid(axis="y", alpha=0.23)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    fig.savefig(OUTPUT / "elbow_train_test.png", dpi=220)
    plt.close(fig)


def money(value):
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.0f}"


def main():
    members = read_members()
    training, test = train_test_split(members, test_size=TEST_FRACTION, random_state=SEED)
    scaler = StandardScaler().fit(training[FEATURES])
    training_scaled = scaler.transform(training[FEATURES])

    with threadpool_limits(limits=1):
        models = {
            int(k): KMeans(n_clusters=int(k), n_init=N_STARTS, random_state=SEED).fit(training_scaled)
            for k in K_VALUES
        }
        train_spread = np.array([model.inertia_ for model in models.values()])
        chosen_k = elbow_k(train_spread)
        v1 = models[chosen_k]
        centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)
        ordered_clusters = centers.sort_values(EXTRAS).index
        display_names = list(SEGMENT_DESCRIPTIONS) if chosen_k == len(SEGMENT_DESCRIPTIONS) else [
            f"Segment {i + 1}" for i in range(chosen_k)
        ]
        segment_names = dict(zip(ordered_clusters, display_names, strict=True))

        # The segment policy is fixed using member-level net only on the training data.
        train_labels = v1.predict(training_scaled)
        credits_if_sent = net_if_credited(training[EXTRAS])
        training_net = (
            pd.Series(credits_if_sent).groupby(train_labels).sum().reindex(centers.index, fill_value=0)
        )
        credit_segments = set(training_net.index[training_net.gt(0)])

        # The test members are assigned to the existing centers, with no second fit.
        test_scaled = scaler.transform(test[FEATURES])
        test_spread = np.array([
            np.square(model.transform(test_scaled).min(axis=1)).sum() for model in models.values()
        ])
        test_k = elbow_k(test_spread)

        all_labels = v1.predict(scaler.transform(members[FEATURES]))
        offer_rule = make_offer_rule(scaler, v1, segment_names, credit_segments)
        new_segment, new_gets_credit = offer_rule(NEW_MEMBER)

    curve = pd.DataFrame({"k": K_VALUES, "training_sse": train_spread, "test_sse": test_spread})
    assignments = members[["member_id", EXTRAS]].copy()
    assignments.insert(1, "split", np.where(assignments.index.isin(training.index), "train", "test"))
    assignments["segment"] = pd.Series(all_labels, index=assignments.index).map(segment_names)
    assignments["baseline_credit"] = True
    assignments["v1_credit"] = np.isin(all_labels, list(credit_segments))
    assignments["baseline_monthly_net"] = net_if_credited(assignments[EXTRAS])
    assignments["v1_monthly_net"] = assignments["baseline_monthly_net"].where(assignments["v1_credit"], 0)

    profiles = centers.loc[ordered_clusters].copy()
    profiles.insert(0, "segment", profiles.index.map(segment_names))
    profiles["train_members"] = np.bincount(train_labels, minlength=chosen_k)[profiles.index]
    profiles["all_members"] = np.bincount(all_labels, minlength=chosen_k)[profiles.index]
    profiles["test_members"] = profiles["all_members"] - profiles["train_members"]
    profiles["description"] = profiles["segment"].map(SEGMENT_DESCRIPTIONS).fillna("See the centroid values.")
    profiles["training_net_if_credited"] = training_net.reindex(profiles.index)
    training_low_spenders = pd.Series(training[EXTRAS].lt(SPEND_THRESHOLD).to_numpy()).groupby(
        train_labels
    ).sum()
    profiles["train_low_spenders"] = training_low_spenders.reindex(profiles.index, fill_value=0)
    profiles["v1_credit"] = profiles.index.isin(credit_segments)
    profiles["all_low_spenders"] = assignments.loc[assignments[EXTRAS].lt(SPEND_THRESHOLD)].groupby(
        "segment"
    ).size().reindex(profiles["segment"], fill_value=0).to_numpy()

    OUTPUT.mkdir(exist_ok=True)
    curve.to_csv(OUTPUT / "elbow_values.csv", index=False)
    profiles.to_csv(OUTPUT / "cluster_profiles.csv", index=False)
    assignments.to_csv(OUTPUT / "member_assignments.csv", index=False)
    plot_elbow(curve, chosen_k, test_k, len(training), len(test))

    baseline_net = assignments["baseline_monthly_net"].sum()
    v1_net = assignments["v1_monthly_net"].sum()
    baseline_test_sse = test_spread[0]
    selected_test_sse = test_spread[chosen_k - 1]
    test_reduction = 1 - selected_test_sse / baseline_test_sse
    eligible = assignments["v1_credit"]
    low_spend = assignments[EXTRAS].lt(SPEND_THRESHOLD)

    lines = [
        "# Gym member clustering results",
        "",
        "## Data and method",
        "",
        f"- Course file: `gym_members.csv`, {len(members)} members and {len(members.columns)} columns. The brief describes member averages over the last three months.",
        f"- Mean extras spend across the dataset: {money(members[EXTRAS].mean())} per month.",
        f"- A seeded random 70/30 split (seed {SEED}) produces {len(training)} training members and {len(test)} test members.",
        f"- K-means uses {', '.join(FEATURES)}. Member ID serves as a record key.",
        f"- StandardScaler centers and scales each input using its training mean and standard deviation. Each k-means fit uses {N_STARTS} initializations and training members only.",
        "- The elbow is the interior k with the largest gap below the straight line joining k = 1 and k = 8 after scaling the spread to those endpoints. The test curve uses the same training centers and scaler.",
        "",
        "## Fit across candidate values of k",
        "",
        "| k | Training spread (SSE) | Test spread (SSE) |",
        "|---:|---:|---:|",
    ]
    for row in curve.itertuples(index=False):
        lines.append(f"| {row.k} | {row.training_sse:,.2f} | {row.test_sse:,.2f} |")
    lines += [
        "",
        f"The training elbow is k = {chosen_k}. The test elbow is k = {test_k}.",
        f"Test spread falls from {baseline_test_sse:,.2f} at k = 1 to {selected_test_sse:,.2f} at k = {chosen_k}, a {test_reduction:.1%} reduction.",
        "",
        "## Member segments",
        "",
        f"The centers below are the fitted training centroids converted to the original units. Counts cover all {len(members)} members assigned using the fitted model.",
        "",
        "| Segment | Members | Age | Visits/month | Minutes/visit | Classes/month | Months as member | Extras/month | Credit? |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for profile in profiles.itertuples(index=False):
        lines.append(
            f"| {profile.segment} | {profile.all_members} | {profile.age:.1f} | "
            f"{profile.visits_per_month:.1f} | {profile.avg_visit_minutes:.1f} | "
            f"{profile.classes_per_month:.1f} | {profile.months_member:.1f} | "
            f"${profile.monthly_extras_spend:.2f} | {'Yes' if profile.v1_credit else 'No'} |"
        )
    lines.append("")
    for profile in profiles.itertuples(index=False):
        lines.append(f"- **{profile.segment}.** {profile.description}")
    lines += [
        "",
        "## Offer decision and monthly net",
        "",
        f"- A credited member who normally spends under {money(SPEND_THRESHOLD)} on extras contributes {money(EXTRA_SPEND_WITH_CREDIT * EXTRAS_MARGIN - CREDIT_COST)} in monthly net. A credited member who spends at least {money(SPEND_THRESHOLD)} contributes {money(-CREDIT_COST)}. An uncredited member contributes $0.",
        "- The model sends the credit to segments whose members produced a positive combined net on training data under these assumptions.",
        f"- A segment needs a share above {CREDIT_COST / (EXTRA_SPEND_WITH_CREDIT * EXTRAS_MARGIN):.1%} below the spend threshold for positive net. The table below shows the training evidence used for each decision.",
        "",
        "| Segment | Training members under threshold | Total training members | Training net if all get credit |",
        "|---|---:|---:|---:|",
        *[
            f"| {profile.segment} | {profile.train_low_spenders} | {profile.train_members} | {money(profile.training_net_if_credited)} |"
            for profile in profiles.itertuples(index=False)
        ],
        "",
        f"- New member (age {NEW_MEMBER['age']}, {NEW_MEMBER['visits_per_month']} visits/month, {NEW_MEMBER['avg_visit_minutes']} minutes/visit, {NEW_MEMBER['classes_per_month']} class/month, {NEW_MEMBER['months_member']} months as a member, {money(NEW_MEMBER[EXTRAS])} in extras): **{new_segment}, {'send the credit' if new_gets_credit else 'do not send the credit'}**.",
        "",
        "| Policy | Recipients under the spend threshold | Recipients at or over the threshold | Total recipients | Monthly net |",
        "|---|---:|---:|---:|---:|",
        f"| Baseline, credit to all | {int(low_spend.sum())} | {int((~low_spend).sum())} | {len(members)} | {money(baseline_net)} |",
        f"| v1, segment credit | {int((eligible & low_spend).sum())} | {int((eligible & ~low_spend).sum())} | {int(eligible.sum())} | {money(v1_net)} |",
        "",
        f"Monthly net difference (v1 minus baseline): **{money(v1_net - baseline_net)}**. These estimates apply the brief's assumed spending response to individual members.",
        "",
    ]
    report = "\n".join(lines)
    (ROOT / "analysis_results.md").write_text(report, encoding="utf-8")
    print(report)
    print("Generated outputs/elbow_train_test.png")


if __name__ == "__main__":
    main()
