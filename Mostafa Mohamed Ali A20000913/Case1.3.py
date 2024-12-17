import pandas as pd
import numpy as np
from scipy.stats import pearsonr

# ========================= Load Modified Dataset =========================
df = pd.read_csv('modified_ratings.csv')
df = df.set_index('User')  # Set 'User' as index for easier manipulation

# ========================= Set Active Users =========================
selected_users = ['User_1', 'User_26', 'User_50']
target_items = df.columns[1:3]  # First two items as target items
print("Selected Active Users:", selected_users)
print("Target Items:", target_items)

# ========================= 1.3.1: Compute PCC Similarity =========================
def compute_pcc_similarity(df, active_user):
    """Compute PCC similarity between active user and all other users."""
    similarities = {}
    active_user_ratings = df.loc[active_user]

    for user in df.index:
        if user != active_user:
            common_items = df.loc[[active_user, user]].dropna(axis=1)
            if len(common_items.columns) > 1:  # Ensure at least 2 common items
                pcc, _ = pearsonr(common_items.loc[active_user], common_items.loc[user])
                similarities[user] = pcc
    return pd.Series(similarities).dropna().sort_values(ascending=False)

pcc_similarities = {user: compute_pcc_similarity(df, user) for user in selected_users}

# ========================= 1.3.2: Top 20% Closest Users =========================
def select_top_20_percent(similarity_scores):
    """Select the top 20% closest users."""
    n_top_users = int(np.ceil(0.2 * len(similarity_scores)))
    return similarity_scores.head(n_top_users)

top_20_users = {user: select_top_20_percent(pcc_similarities[user]) for user in selected_users}

# ========================= 1.3.3: Predict Ratings =========================
def predict_ratings(active_user, top_users, df, similarity_scores, threshold=3.0):
    """Predict ratings for target items."""
    predictions = {}
    for item in target_items:
        if pd.isna(df.loc[active_user, item]):
            numerator, denominator = 0, 0
            for neighbor in top_users.index:
                neighbor_rating = df.loc[neighbor, item]
                if not pd.isna(neighbor_rating):
                    sim = similarity_scores[neighbor]
                    numerator += sim * neighbor_rating
                    denominator += abs(sim)
            if denominator > 0:
                predictions[item] = numerator / denominator
    return {item: "Like" if pred >= threshold else "Dislike" for item, pred in predictions.items()}

predictions_1_3_3 = {user: predict_ratings(user, top_20_users[user], df, pcc_similarities[user]) for user in selected_users}

# ========================= 1.3.4: Discount Factor and Discounted Similarity =========================
def compute_discounted_similarity(similarity_scores, beta=0.5):
    """Apply discounting to similarity scores."""
    return similarity_scores.apply(lambda x: x * beta if x < beta else x)

discounted_similarities = {user: compute_discounted_similarity(pcc_similarities[user]) for user in selected_users}

# ========================= 1.3.5: Top 20% Users with Discounted Similarity =========================
top_20_users_discounted = {user: select_top_20_percent(discounted_similarities[user]) for user in selected_users}

# ========================= 1.3.6: Predict Ratings with Discounted Similarity =========================
predictions_1_3_6 = {user: predict_ratings(user, top_20_users_discounted[user], df, discounted_similarities[user]) for user in selected_users}

# ========================= Save Results =========================
with open('case1_3_results.txt', 'w', encoding='utf-8') as file:
    file.write("1.3.1: PCC Similarity Results for Active Users\n")
    for user, sims in pcc_similarities.items():
        file.write(f"\n{user}:\n{sims}\n")

    file.write("\n1.3.2: Top 20% Closest Users\n")
    for user, top_users in top_20_users.items():
        file.write(f"\n{user}:\n{top_users}\n")

    file.write("\n1.3.3: Predicted Ratings (Like/Dislike) for Target Items\n")
    for user, preds in predictions_1_3_3.items():
        file.write(f"\n{user}:\n{preds}\n")

    file.write("\n1.3.5: Top 20% Users with Discounted Similarity\n")
    for user, top_users in top_20_users_discounted.items():
        file.write(f"\n{user}:\n{top_users}\n")

    file.write("\n1.3.6: Predicted Ratings with Discounted Similarity for Target Items\n")
    for user, preds in predictions_1_3_6.items():
        file.write(f"\n{user}:\n{preds}\n")

print("Results have been saved to 'case1_3_results.txt'.")
