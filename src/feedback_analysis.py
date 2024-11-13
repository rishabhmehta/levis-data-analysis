import pandas as pd
from keybert import KeyBERT
from sentence_transformers import SentenceTransformer, util


# data = pd.read_csv("feedback.csv")
# kw_model = KeyBERT()
#
# data['VoteChoice'] = data['VoteChoice'].fillna('').astype(str)
#
# data['Keywords'] = data['VoteChoice'].apply(lambda x: kw_model.extract_keywords(x, top_n=3))
#
# data[['VoteChoice', 'Keywords']].to_csv("feedback_with_keywords.csv", index=False)
#
# # Group feedback keywords by Region and Location
# # Initialize an empty dictionary to store aggregated keywords
# region_location_keywords = {}
#
# # Iterate through each row to aggregate keywords by Region and Location
# for index, row in data.iterrows():
#     region = row['Region']
#     location = row['Location']
#     keywords = [kw[0] for kw in row['Keywords']]  # Extract keyword text only
#
#     # Initialize region-location in the dictionary if not present
#     if (region, location) not in region_location_keywords:
#         region_location_keywords[(region, location)] = []
#
#     # Add keywords to the list for the specific region-location pair
#     region_location_keywords[(region, location)].extend(keywords)
#
# # Convert the dictionary to a DataFrame for easier analysis
# region_location_df = pd.DataFrame([
#     {"Region": region, "Location": location, "Keywords": list(set(keywords))}
#     for (region, location), keywords in region_location_keywords.items()
# ])
#
# # Save the results to a new CSV
# region_location_df.to_csv("keywords_by_region_location.csv", index=False)
#
# print("Output saved to keywords_by_region_location.csv")


# Load the original data and the keywords file
feedback_data = pd.read_csv("feedback.csv")
keywords_data = pd.read_csv("feedback_with_keywords.csv")

# Merge the two datasets on the 'VoteChoice' column (or another unique identifier if available)
merged_data = pd.merge(feedback_data, keywords_data, on="VoteChoice")

# Group feedback keywords by Region and Location
region_location_keywords = {}

# Iterate through each row to aggregate keywords by Region and Location
for index, row in merged_data.iterrows():
    region = row['Region']
    location = row['Location']
    keywords = eval(row['Keywords'])  # Convert the string representation of the list back to a list

    # Initialize region-location in the dictionary if not present
    if (region, location) not in region_location_keywords:
        region_location_keywords[(region, location)] = []

    # Add keywords to the list for the specific region-location pair
    region_location_keywords[(region, location)].extend([kw[0] for kw in keywords])

# Convert the dictionary to a DataFrame for easier analysis
region_location_df = pd.DataFrame([
    {"Region": region, "Location": location, "Keywords": list(set(keywords))}
    for (region, location), keywords in region_location_keywords.items()
])

# Save the results to a new CSV
region_location_df.to_csv("keywords_by_region_location.csv", index=False)

print("Output saved to keywords_by_region_location.csv")
