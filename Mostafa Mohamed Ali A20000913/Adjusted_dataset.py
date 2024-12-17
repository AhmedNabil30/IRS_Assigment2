import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ========================= Part 1 & 2: Load Dataset and Count Users/Items =========================
# Load the dataset
df = pd.read_csv('pivoted_user_item_ratings.csv')

# Rename the first column to 'User' for clarity
df.rename(columns={'Unnamed: 0': 'User'}, inplace=True)

# ========================= Step to Compute Common Users and Select Diverse Users =========================
def compute_common_users(df):
    """Compute No_common_users for all users."""
    common_users = {}
    for user in df['User']:
        active_user_ratings = df.loc[df['User'] == user].drop('User', axis=1).iloc[0]
        co_rated_users = 0
        for i, row in df.iterrows():
            if row['User'] == user:
                continue
            common_items = row.drop('User').notna() & active_user_ratings.notna()
            if common_items.sum() > 0:
                co_rated_users += 1
        common_users[user] = co_rated_users
    return common_users

# Compute common users for all users
common_users = compute_common_users(df)

# Select 3 users with diverse No_common_users
sorted_common_users = sorted(common_users.items(), key=lambda x: x[1])  # Sort by No_common_users
selected_users = [sorted_common_users[0][0], sorted_common_users[len(sorted_common_users)//2][0], sorted_common_users[-1][0]]  # Pick min, mid, max

# ========================= Writing to TXT File =========================
with open('results.txt', 'w', encoding='utf-8') as file:
    file.write("Part 1 & 2: Load Dataset and Count Users/Items\n")
    # 1. Total number of users
    tnu = df['User'].nunique()
    file.write(f"Total number of users: {tnu}\n")

    # 2. Total number of items
    tni = len(df.columns) - 1  # Exclude 'User' column
    file.write(f"Total number of items: {tni}\n\n")

    # Selected Active Users
    file.write("Selected Active Users with Different No_common_users:\n")
    for user, common_count in sorted_common_users[:3]:
        file.write(f"{user}: {common_count} common users\n")
    file.write("\n")

    # ========================= Part 6: Active Users with Missing Ratings =========================
    # Introduce missing ratings for the 3 fixed active users
    for i, user in enumerate(selected_users):
        num_missing_ratings = 2 + i * 1  # 2, 3, and 5 missing ratings for U1, U2, U3
        missing_items = df.columns[1:1+num_missing_ratings]  # Use specific columns to introduce NaNs
        df.loc[df['User'] == user, missing_items] = np.nan

    file.write("Updated Active Users with Missing Ratings:\n")
    file.write(f"{df.loc[df['User'].isin(selected_users)]}\n\n")

    # ========================= Part 7: Target Items with Missing Ratings =========================
    fixed_target_books = df.columns[1:3]  # Use first two books as fixed target items
    for i, book in enumerate(fixed_target_books):
        missing_percentage = 0.04 if i == 0 else 0.10  # 4% for the first item, 10% for the second item
        total_ratings = df[book].count()
        num_missing_ratings = int(np.ceil(missing_percentage * total_ratings))
        missing_indices = df.index[:num_missing_ratings]  # Fixed indices to ensure repeatability
        df.loc[missing_indices, book] = np.nan

    file.write("Missing Ratings for Target Books:\n")
    file.write(f"{df[fixed_target_books].isna().sum()}\n\n")

    # ========================= Part 8: Count Co-rated Users and Items =========================
    def count_co_rated(df, active_user):
        active_user_ratings = df.loc[df['User'] == active_user].drop('User', axis=1).iloc[0]
        co_rated_users = 0
        co_rated_items = []

        for i, row in df.iterrows():
            if row['User'] == active_user:
                continue
            common_items = row.drop('User').notna() & active_user_ratings.notna()
            num_common = common_items.sum()
            if num_common > 0:
                co_rated_users += 1
                co_rated_items.append(num_common)

        return co_rated_users, co_rated_items

    file.write("Part 8: Count Co-rated Users and Items\n")
    co_rated_data = []
    for active_user in selected_users:
        co_rated_users, co_rated_items = count_co_rated(df, active_user)
        total_co_rated_items = sum(co_rated_items)
        file.write(f"Active User: {active_user}\n")
        file.write(f"No_common_users: {co_rated_users}\n")
        file.write(f"No_coRated_items: {total_co_rated_items}\n\n")
        co_rated_data.append((co_rated_users, total_co_rated_items))

    # ========================= Part 9: Create 2-D Array of Co-Rated Users and Items =========================
    co_rated_array = np.array(sorted(co_rated_data, key=lambda x: -x[0]))
    file.write("Part 9: 2-D Array of Co-rated Users and Items:\n")
    file.write(f"{co_rated_array}\n\n")

    # ========================= Part 10: Curve for Quantity of Ratings =========================
    item_ratings_count = df.drop('User', axis=1).count()

    plt.figure(figsize=(12, 6))
    plt.plot(item_ratings_count.index, item_ratings_count.values, marker='o')
    plt.xticks(rotation=90)
    plt.title("Part 10: Quantity of Ratings for Each Item")
    plt.xlabel("Items (Books)")
    plt.ylabel("Number of Ratings")
    plt.grid()
    plt.savefig('item_ratings_curve.png')
    plt.close()
    file.write("Curve saved as 'item_ratings_curve.png'.\n\n")

# ========================= Save Modified Dataset =========================
df.to_csv('modified_ratings.csv', index=True)
print("The modified dataset has been saved to 'modified_ratings.csv'.")
print("Results have been saved to 'results.txt' and the curve to 'item_ratings_curve.png'.")
