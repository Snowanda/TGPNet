import os
from models.tgpnet import TGPNet
from trainer.tree_trainer import TreeTrainer

tree_folder = "data/datasetFull"
model = TGPNet()
trainer = TreeTrainer(model, tree_folder)
filenames = sorted([f for f in os.listdir(tree_folder) if f.endswith('.skel')])
#trainer.train(filenames, epochs=600)

trainer.prepare_curriculum(filenames)
trainer.train(epochs=600, curriculum_epochs=10)


"""import torch
from torch.utils.data import DataLoader
from data.tree_dataset import TreeDataset
from models.tgpnet import TGPNet
import torch.nn as nn

# Training pipeline
def train_model(dataset_path, epochs=600, batch_size=4, initial_lr=1e-3):
    dataset = TreeDataset(dataset_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=lambda x: x)

    model = TGPNet().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr, betas=(0.9, 0.999))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
    loss_fn = nn.L1Loss()

    for epoch in range(epochs):
        total_loss = 0
        model.train()
        for batch in dataloader:
            optimizer.zero_grad()
            loss = 0
            for local_input, X, A, current_idx, gt_z in batch:
                local_input = local_input.cuda()
                X = X.cuda()
                A = A.cuda()
                gt_z = gt_z.cuda()
                z_pred = model(local_input, X, A, current_idx)
                loss += loss_fn(z_pred, gt_z)
            loss /= len(batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}")

if __name__ == "__main__":
    train_model("data/dataset", epochs=600, batch_size=4)"""