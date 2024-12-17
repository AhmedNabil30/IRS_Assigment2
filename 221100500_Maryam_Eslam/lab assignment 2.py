import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Dell\\Downloads\\Balanced_CF_Dataset.csv")

data = data[data['rating'].notna() & (data['rating'] != 0)]

print(f"Rating Range: {data['rating'].min()} to {data['rating'].max()}")

user_count = data['userId'].nunique()
item_count = data['productId'].nunique()
print(f"Total Number of Users: {user_count}\nTotal Number of Items: {item_count}")

product_ratings = data['productId'].value_counts()
print("Number of Ratings per Product:\n", product_ratings.head())

active_users_list = ['U35', 'U7', 'U10']
print(f"Active Users: {', '.join(active_users_list)}")

target_items_list = data['productId'].unique()[:2]
print(f"Target Items: {target_items_list[0]} and {target_items_list[1]}")

def compute_co_ratings(user, data):
    user_items = set(data.loc[data['userId'] == user, 'productId'])
    total_users = data[data['productId'].isin(user_items)]['userId'].nunique()
    return total_users, len(user_items)

for user in active_users_list:
    print(f"Co-ratings for {user}: {compute_co_ratings(user, data)}")

def find_common_users(data, active_users):
    common_users_list = []
    for user in active_users:
        user_items = set(data.loc[data['userId'] == user, 'productId'])
        for other_user in data['userId'].unique():
            if user != other_user:
                other_items = set(data.loc[data['userId'] == other_user, 'productId'])
                common_count = len(user_items & other_items)
                common_users_list.append([common_count, other_user])
    return sorted(common_users_list, key=lambda x: x[0], reverse=True)

common_users_sorted = find_common_users(data, active_users_list)
print("Top Common Users (Descending Order):", common_users_sorted[:5])

plt.figure(figsize=(10, 6))
plt.plot(product_ratings.values, marker='o')
plt.title("Quantity of Ratings for Every Item")
plt.xlabel("Items")
plt.ylabel("Number of Ratings")
plt.grid()
plt.show()

def calculate_threshold(user, data, threshold_pct):
    user_items = set(data.loc[data['userId'] == user, 'productId'])
    threshold_users = [other_user for other_user in data['userId'].unique() if user != other_user and len(user_items & set(data.loc[data['userId'] == other_user, 'productId'])) / len(user_items) >= threshold_pct]
    return len(threshold_users)

for user in active_users_list:
    print(f"Threshold β for {user}: {calculate_threshold(user, data, 0.3)}")

results_summary = {
    "Total Users": user_count,
    "Total Items": item_count,
    "Ratings Per Product": product_ratings,
    "Active Users": active_users_list,
    "Target Items": list(target_items_list),
    "Co-Ratings": {user: compute_co_ratings(user, data) for user in active_users_list},
    "Thresholds": {user: calculate_threshold(user, data, 0.3) for user in active_users_list}
}


df_cleaned = df.groupby(['userId', 'productId'], as_index=False)['rating'].mean()
print("Dataset Overview (Cleaned):")
display(df_cleaned.head())
print(f"Rating Range: {df_cleaned['rating'].min()} to {df_cleaned['rating'].max()}")
total_users = df_cleaned['userId'].nunique()
total_items = df_cleaned['productId'].nunique()
print(f"Total Number of Users: {total_users}")
print(f"Total Number of Items: {total_items}")

user_item_matrix = df_cleaned.pivot(index='userId', columns='productId', values='rating').fillna(0)
cosine_sim_matrix = cosine_similarity(user_item_matrix)
cosine_sim_df = pd.DataFrame(cosine_sim_matrix, index=user_item_matrix.index, columns=user_item_matrix.index)
print("Cosine Similarity Matrix:")
display(cosine_sim_df.head())

def get_top_closest_users(active_user, similarity_matrix, top_percent=0.2):
    user_similarities = similarity_matrix[active_user].drop(active_user)
    top_n = int(len(user_similarities) * top_percent)
    return user_similarities.nlargest(top_n)

active_users = ['U35', 'U7', 'U10']
print(f"Active Users: {active_users}")

top_users = {}
for user in active_users:
    top_users[user] = get_top_closest_users(user, cosine_sim_df)
    print(f"Top Closest Users for {user}:")
    display(top_users[user])

def predict_ratings(active_user, top_users, user_item_matrix):
    unseen_items = user_item_matrix.loc[active_user][user_item_matrix.loc[active_user] == 0].index
    predictions = {}
    for item in unseen_items:
        numerator = 0
        denominator = 0
        for user, sim in top_users.items():
            rating = user_item_matrix.loc[user, item]
            numerator += sim * rating
            denominator += abs(sim)
        predictions[item] = numerator / denominator if denominator != 0 else 0
    return predictions

predictions = {}
for user in active_users:
    predictions[user] = predict_ratings(user, top_users[user], user_item_matrix)
    print(f"Predicted Ratings for {user}:")
    display(predictions[user])

def compute_discounted_similarity(active_user, similarity_matrix, beta):
    user_similarities = similarity_matrix[active_user].drop(active_user)
    df = user_similarities.apply(lambda x: x / (1 + beta))
    return df

threshold_beta = {user: 2 for user in active_users}

discounted_sim = {}
for user in active_users:
    discounted_sim[user] = compute_discounted_similarity(user, cosine_sim_df, threshold_beta[user])
    print(f"Discounted Similarity for {user}:")
    display(discounted_sim[user])

discounted_top_users = {}
for user in active_users:
    discounted_top_users[user] = discounted_sim[user].nlargest(int(len(discounted_sim[user]) * 0.2))
    print(f"Top Discounted Users for {user}:")
    display(discounted_top_users[user])

discounted_predictions = {}
for user in active_users:
    discounted_predictions[user] = predict_ratings(user, discounted_top_users[user], user_item_matrix)
    print(f"Predicted Ratings with Discounted Similarity for {user}:")
    display(discounted_predictions[user])

print("Comparison of Top Closest Users (Original vs Discounted):")
for user in active_users:
    original = set(top_users[user].index)
    discounted = set(discounted_top_users[user].index)
    overlap = original & discounted
    print(f"User {user} - Overlap: {len(overlap)} / {len(original)}")

print("Comparison of Predictions (Original vs Discounted):")
for user in active_users:
    original_preds = predictions[user]
    discounted_preds = discounted_predictions[user]
    print(f"Predictions for {user}:")
    for item in original_preds.keys():
        print(f"Item {item}: Original={original_preds[item]:.2f}, Discounted={discounted_preds[item]:.2f}")
