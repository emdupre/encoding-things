from collections import defaultdict
from time import perf_counter

import torch
import torch.nn as nn
from himalaya.scoring import correlation_score
from sklearn.metrics import r2_score
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    LeaveOneGroupOut,
)
from torch.utils.data import DataLoader, Dataset

"""
Adapted from :
https://michael-franke.github.io/npNLG/04-ANNs/04d-MLP-pytorch.html
"""


# Creating our dataset class
class Dataset(Dataset):
    # Constructor
    def __init__(self, X_matrix, y_matrix):
        self.X = X_matrix
        self.y = y_matrix

    # Getting the data
    def __getitem__(self, index):
        return self.X[index], self.y[index]

    # Getting length of the data
    def __len__(self):
        return len(self.X)


class MLPRegressor(nn.Module):
    """
    Multi-layer perceptron for non-linear regression.
    """

    def __init__(self, n_input=1, n_hidden=256, n_output=1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features=n_input, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_hidden),
            nn.ReLU(),
            nn.Linear(in_features=n_hidden, out_features=n_output),
        )

    def fit(self, data_loader, n_train_steps=1000, learning_rate=1e-4):
        # Define the loss function and optimizer
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        # Run the training loop
        for epoch in range(0, n_train_steps):
            # Set initial loss value
            current_loss = 0.0

            # Iterate over the training data
            for X, y in data_loader:
                # Zero the gradients
                optimizer.zero_grad()
                # Perform forward pass
                outputs = self(torch.as_tensor(X, dtype=torch.float32)).squeeze()
                # Compute loss
                loss = loss_function(outputs, torch.as_tensor(y, dtype=torch.float32))
                # Perform backward pass
                loss.backward()
                # Perform optimization
                optimizer.step()
                # Print statistics
                current_loss += loss.item()

            # if (epoch + 1) % 2500 == 0:
            print("Loss after epoch %5d: %.3f" % (epoch + 1, current_loss))
            current_loss = 0.0

        # Process is complete.
        print("Training process has finished.")

    def forward(self, x):
        return self.layers(x)


def mlp_pytorch(
    X_matrix,
    y_matrix,
    groups=None,
    scoring=r2_score,
    cv_strategy="image",
    batch_size=64,
):
    """
    Parameters
    ----------
    X_matrix : np.arr
        Training data for stimulus embeddings.
        Expected shape (n_samples, n_features)
    y_matrix : np.arr
        Training data for brain responses
        Expected shape (n_samples, n_features, n_repeats)
    groups : np.arr
        Group labels for outer_cv, should correspond to image
        identity or image category.
        Expected shape (n_samples, )
    scoring : Callable
        Scoring function for estimator predictions.
    cv_strategy : str
    """
    scores = defaultdict()
    train_indices, test_indices = [], []
    test_scores = []
    fit_times = []

    if groups is None:
        outer_cv = KFold(shuffle=True, random_state=0)
    else:
        if cv_strategy in ["category", "image"]:
            outer_cv = GroupKFold(shuffle=True, random_state=0)
        elif cv_strategy == "multilabel":
            outer_cv = LeaveOneGroupOut()

    mlp = MLPRegressor(
        n_input=X_matrix.shape[1], n_output=y_matrix.shape[1], n_hidden=256
    )

    for train_index, test_index in outer_cv.split(X_matrix, y_matrix, groups):
        start_time = perf_counter()

        train_indices.append(train_index)
        test_indices.append(test_index)

        # Creating DataLoader object
        train_dataset = Dataset(X_matrix[train_index], y_matrix[train_index])
        data_loader = DataLoader(
            dataset=train_dataset, batch_size=batch_size, shuffle=True
        )

        mlp.fit(data_loader)

        y_pred = (
            mlp.forward(torch.as_tensor(X_matrix[test_index], dtype=torch.float32))
            .detach()
            .numpy()
        )

        if scoring is correlation_score:
            test_scores.append(correlation_score(y_matrix[test_index], y_pred))
        else:
            test_scores.append(
                r2_score(y_matrix[test_index], y_pred, multioutput="raw_values")
            )

        fit_time = perf_counter() - start_time
        fit_times.append(fit_time)

    scores["fit_time"] = fit_times
    scores["alphas"] = None
    scores["per_target_test_scores"] = test_scores
    scores["indices"] = {"train": train_indices, "test": test_indices}

    return scores
