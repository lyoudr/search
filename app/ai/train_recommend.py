import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader 


# Suppose we have these interactions
# interactions = [
#     [0, 0, 5.0],  # user 0 rated item 0 with 5
#     [0, 1, 3.0],
#     [1, 0, 4.0],
#     [1, 2, 1.0],
#     [2, 1, 4.0],
#     [2, 2, 5.0],
# ]

# class RatingDataset(Dataset):
#     def __init__(self, data):
#         self.data = torch.tensor(data, dtype=torch.float32)

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         user, item, rating = self.data[idx]
#         return int(user), int(item), rating

# dataset = RatingDataset(interactions)
# loader = DataLoader(dataset, batch_size=2, shuffle=True)


# # ######## 3. Define the Model ##########
# class MatrixFactorization(nn.Module):
#     def __init__(self, num_users, num_items, latent_dim):
#         super().__init__()
#         self.user_emb = nn.Embedding(num_users, latent_dim)
#         self.item_emb = nn.Embedding(num_items, latent_dim)

#     def forward(self, user, item):
#         u = self.user_emb(user)   # shape: [batch_size, latent_dim]
#         v = self.item_emb(item)   # shape: [batch_size, latent_dim]
#         return (u * v).sum(dim=1) # dot product for predicted rating
    

# # ######## 4. Training the Model ##########
# model = MatrixFactorization(num_users=3, num_items=3, latent_dim=8)
# optimizer = optim.Adam(model.parameters(), lr=0.01)
# loss_fn = nn.MSELoss()

# for epoch in range(20):
#     total_loss = 0.0
#     for user, item, rating in loader:
#         pred = model(user, item)
#         loss = loss_fn(pred, rating)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         total_loss += loss.item()
#     print(f"Epoch {epoch+1}: Loss = {total_loss:.4f}")

# # ######## 5. Making Predictions ##########
# user_id = torch.tensor([0])  # Example user
# item_id = torch.tensor([1])
# pred_rating = model(user_id, item_id).item()
# print(f"Predicted rating for user 0 on item 2: {pred_rating:.2f}")


#### Content Based Filtering Example ####
# Given item features and user interaction labels (likes/dislikes or ratings), train a model to predict user preference.
import numpy as np 

# Suppose 5 items with 3 features each
item_features = torch.tensor([
    [1.0, 0.0, 0.2],   # Item 0
    [0.9, 0.1, 0.1],   # Item 1
    [0.1, 1.0, 0.2],   # Item 2
    [0.0, 1.0, 0.3],   # Item 3
    [0.2, 0.1, 0.9],   # Item 4
], dtype=torch.float32)

# User feedback: (item_id, label)
# 1 = liked, 0 = not liked
user_interactions = [
    (0, 1),
    (1, 1),
    (2, 0),
    (3, 0),
    (4, 1),
]


class UserPreferenceDataset(Dataset):
    def __init__(self, item_features, interactions):
        self.x = torch.stack([item_features[i] for i, _ in interactions])
        self.y = torch.tensor([label for _, label in interactions], dtype=torch.float32)

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
dataset = UserPreferenceDataset(item_features, user_interactions)
loader = DataLoader(dataset, batch_size=2, shuffle=True)


class ContentBasedModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x).squeeze(-1) # ✅ Only squeeze the last dimension


model = ContentBasedModel(input_dim=3)
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()  # Binary cross-entropy for binary like/dislike

for epoch in range(20):
    total_loss = 0
    for x_batch, y_batch in loader:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")
# Making Recommendations
model.eval()
with torch.no_grade():
    scores = model(item_features)
    ranked_items = torch.argsort(scores, descending=True)

print("Recommended item order:")
for idx in ranked_items:
    print(f"Item {idx.item()}, Score: {scores[idx].item():.4f}")