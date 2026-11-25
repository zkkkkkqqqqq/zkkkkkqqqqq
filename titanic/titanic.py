from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

data_dir = Path(__file__).resolve().parent
feature_columns = [
    "Pclass", "Sex", "Age", "SibSp", "Parch",
    "Ticket", "Fare", "Cabin", "Embarked",
]


def make_features(data, ticket_counts):
    features = data[feature_columns].copy()
    features["Sex"] = features["Sex"].map({"male": 0, "female": 1}).fillna(-1)
    features["Embarked"] = features["Embarked"].map(
        {"S": 0, "C": 1, "Q": 2}
    ).fillna(-1)
    features["Ticket"] = features["Ticket"].map(ticket_counts).fillna(1)
    features["Cabin"] = features["Cabin"].notna().astype(np.float32)
    features[["Age", "Fare"]] = features[["Age", "Fare"]].fillna(-1.0)
    return features.to_numpy(dtype=np.float32)


class Trainset(Dataset):
    def __init__(self, features, labels):
        self.x = torch.from_numpy(features)
        self.y = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


data = pd.read_csv(data_dir / "train.csv")
test_inputs = pd.read_csv(data_dir / "test.csv")

train_rows, valid_rows = train_test_split(
    data, test_size=0.2, random_state=42, stratify=data["Survived"]
)

# 用训练部分统计票号次数，三份数据共用这张统计表。
ticket_counts = train_rows["Ticket"].value_counts()
train_x = make_features(train_rows, ticket_counts)
valid_x = make_features(valid_rows, ticket_counts)
test_x = make_features(test_inputs, ticket_counts)

# 每列除以训练部分该列的最大绝对值，缩小不同特征的数值差距。
scale = np.maximum(np.abs(train_x).max(axis=0), 1.0)
train_set = Trainset(train_x / scale, train_rows[["Survived"]].to_numpy())
valid_set = Trainset(valid_x / scale, valid_rows[["Survived"]].to_numpy())
test_set = torch.from_numpy(test_x / scale)

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=32, shuffle=False)
test_loader = DataLoader(test_set, batch_size=32, shuffle=False)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(9, 6)
        self.linear2 = torch.nn.Linear(6, 3)
        self.linear3 = torch.nn.Linear(3, 1)
        self.activate = torch.nn.ReLU()

    def forward(self, x):
        x = self.activate(self.linear1(x))
        x = self.activate(self.linear2(x))
        return self.linear3(x)


torch.manual_seed(42)
model = Model()
criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

model.train()
for epoch in range(20):
    total_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        scores = model(inputs)
        loss = criterion(scores, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)

    average_loss = total_loss / len(train_set)
    print(f"第 {epoch + 1}/20 轮，平均损失 {average_loss:.4f}")

model.eval()
correct = 0
predictions = []

with torch.no_grad():
    for inputs, labels in valid_loader:
        predicted = model(inputs) >= 0
        correct += (predicted == labels).sum().item()

    for inputs in test_loader:
        predicted = (model(inputs) >= 0).long()
        predictions.extend(predicted.squeeze(1).tolist())

print(f"验证集准确率 {correct / len(valid_set):.2%}")

submission = pd.DataFrame({
    "PassengerId": test_inputs["PassengerId"],
    "Survived": predictions,
})
print(submission.head())
print(f"共生成 {len(submission)} 条预测")

submission.to_csv(data_dir / "submission.csv", index=False)
