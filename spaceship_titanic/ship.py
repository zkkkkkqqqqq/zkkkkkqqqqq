from pathlib import Path
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split

data_dir = Path(__file__).resolve().parent
feature_columns = ["HomePlanet", "CryoSleep", "Destination", "Age",
                    "VIP", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]

def make_features(data):
    features = data[feature_columns].copy()
    features["HomePlanet"] = features["HomePlanet"].map({"Europa": 0, "Earth": 1, "Mars": 2}).fillna(-1)
    features["CryoSleep"] = features["CryoSleep"].map({False: 0, True: 1}).fillna(-1)
    features["Destination"] = features["Destination"].map({"TRAPPIST-1e": 0, "PSO J318.5-22": 1, "55 Cancri e": 2}).fillna(-1)
    features["VIP"] = features["VIP"].map({False: 0, True: 1}).fillna(-1)
    features[["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]] = features[["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]].fillna(-1.0)
    cabin = data["Cabin"].str.split("/", expand=True)
    deck_map = {deck: i for i, deck in enumerate("ABCDEFGT")}
    features["CabinDeck"] = cabin[0].map(deck_map).fillna(-1)
    features["CabinNumber"] = pd.to_numeric(cabin[1], errors="coerce").fillna(-1)
    features["CabinSide"] = cabin[2].map({"P": 0, "S": 1}).fillna(-1)
    return features.to_numpy(dtype=np.float32)

class Trainset(Dataset):
    def __init__(self, features, lables):
        super().__init__()
        self.x = torch.from_numpy(features)
        self.y = torch.tensor(lables, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

data = pd.read_csv(data_dir/'train.csv')
test_inputs = pd.read_csv(data_dir/'test.csv')

train_rows, valid_rows = train_test_split(data, test_size=0.2,
                                          random_state=42, stratify=data["Transported"])
train_x = make_features(train_rows)
valid_x = make_features(valid_rows)
test_x = make_features(test_inputs)

scale = np.maximum(np.abs(train_x).max(axis=0), 1.0)
train_set = Trainset(train_x/scale, train_rows[["Transported"]].to_numpy())
valid_set = Trainset(valid_x/scale, train_rows[["Transported"]].to_numpy())
test_set = torch.from_numpy(test_x/scale)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(13, 9)
        self.linear2 = torch.nn.Linear(9, 6)
        self.linear3 = torch.nn.Linear(6, 3)
        self.linear4 = torch.nn.Linear(3, 1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        x = self.activate(self.linear3(x))
        return self.linear4(x)

model = Model()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(200):
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        scores = model(inputs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    average_loss = total_loss / len(train_set)
    print(f"第 {epoch + 1}/200 轮，平均损失 {average_loss:.4f}")

model.eval()
correct = 0
predictions = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        predicted = model(inputs) >= 0
        correct += (predicted == labels).sum().item()

    for inputs in test_loader:
        predicted = (model(inputs) >= 0)
        predictions.extend(predicted.squeeze(1).tolist())

print(f"验证集准确率 {correct / len(valid_set):.2%}")

submission = pd.DataFrame({
    "PassengerId": test_inputs["PassengerId"],
    "Transported": predictions,
})
print(submission.head())
print(f"共生成 {len(submission)} 条预测")

submission.to_csv(data_dir / "submission.csv", index=False)
