import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# ========================= Load Modified Dataset =========================
df = pd.read_csv('modified_ratings.csv')
df = df.set_index('User')  # Set User as index for easy manipulation

# ========================= Fixed Target Items and Selected Users =========================
target_items = df.columns[:3]  # Automatically selects the first two items
selected_users = ['User_1', 'User_26', 'User_50']  # Users with missing ratings
print("Target Items:", target_items)
print("Selected Users:", selected_users)

# ========================= 2.3.1: PCC Similarity for Target Items =========================
def compute_pcc_similarity(df, target_item):
    """Compute PCC for the target item with all other items."""
    similarities = {}
    for item in df.columns:
        if item != target_item:
            common_ratings = df[[target_item, item]].dropna()
            if len(common_ratings) > 1:  # Compute only if at least 2 common ratings exist
                correlation = common_ratings[target_item].corr(common_ratings[item])
                similarities[item] = correlation if not np.isnan(correlation) else 0
            else:
                similarities[item] = 0  # No sufficient data, assign 0 similarity
    return pd.Series(similarities).sort_values(ascending=False)

# Calculate PCC similarities for target items
pcc_similarities = {item: compute_pcc_similarity(df, item) for item in target_items}

# ========================= 2.3.2: Top 20% Closest Items =========================
def select_top_20_percent(similarity_scores):
    """Select the top 20% closest items."""
    n_top_items = max(1, int(np.ceil(0.2 * len(similarity_scores))))  # Ensure at least 1 item
    return similarity_scores.head(n_top_items)

top_20_items = {item: select_top_20_percent(pcc_similarities[item]) for item in target_items}

# ========================= 2.3.3: Predict Ratings for Missing Values =========================
def predict_ratings(target_item, top_items, df, threshold=3.0):
    """Predict missing ratings for the target item for selected users."""
    predictions = {}
    for user in selected_users:
        if pd.isna(df.loc[user, target_item]):  # Predict only if the value is missing
            numerator, denominator = 0, 0
            for similar_item, similarity in top_items.items():
                if not pd.isna(df.loc[user, similar_item]):  # Include only valid ratings
                    numerator += similarity * df.loc[user, similar_item]
                    denominator += abs(similarity)
            if denominator > 0:
                prediction = numerator / denominator
                predictions[user] = "Like" if prediction >= threshold else "Dislike"
    return predictions

predictions_2_3_3 = {item: predict_ratings(item, top_20_items[item], df) for item in target_items}

# ========================= 2.3.4: Discount Factor and Discounted Similarity =========================
def compute_discounted_similarity(similarity_scores, beta=0.5):
    """Apply discount factor to similarity scores."""
    return similarity_scores.apply(lambda x: x * beta if x < beta else x)

discounted_similarities = {item: compute_discounted_similarity(pcc_similarities[item]) for item in target_items}

# ========================= 2.3.5: Top 20% Items with Discounted Similarity =========================
top_20_items_discounted = {item: select_top_20_percent(discounted_similarities[item]) for item in target_items}

# ========================= 2.3.6: Predict Ratings with Discounted Similarity =========================
predictions_2_3_6 = {item: predict_ratings(item, top_20_items_discounted[item], df) for item in target_items}

# ========================= Save Results =========================
with open('case2_3_results.txt', 'w', encoding='utf-8') as file:
    # 2.3.1
    file.write("2.3.1: PCC Similarity for Target Items\n")
    for item, sims in pcc_similarities.items():
        file.write(f"\n{item}:\n{sims}\n")

    # 2.3.2
    file.write("\n2.3.2: Top 20% Closest Items\n")
    for item, top_items in top_20_items.items():
        file.write(f"\n{item}:\n{top_items}\n")

    # 2.3.3
    file.write("\n2.3.3: Predicted Ratings for Missing Values\n")
    for item, preds in predictions_2_3_3.items():
        file.write(f"\n{item}:\n{preds}\n")

    # 2.3.4
    file.write("\n2.3.4: Discounted Similarity for Target Items\n")
    for item, sims in discounted_similarities.items():
        file.write(f"\n{item}:\n{sims}\n")

    # 2.3.5
    file.write("\n2.3.5: Top 20% Items with Discounted Similarity\n")
    for item, top_items in top_20_items_discounted.items():
        file.write(f"\n{item}:\n{top_items}\n")

    # 2.3.6
    file.write("\n2.3.6: Predicted Ratings with Discounted Similarity\n")
    for item, preds in predictions_2_3_6.items():
        file.write(f"\n{item}:\n{preds}\n")

print("Results have been saved to 'case2_3_results.txt'.")
