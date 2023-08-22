import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size=17, hidden_size=50, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def forward(self, x, hidden=None):
        
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out, hidden

def train_model(data, model, epochs=100, lr=0.001, clip_norm=5):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # A simple learning rate scheduler can help in some cases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=False)
    
    with tqdm(range(epochs), total=epochs, desc="Training model", position=0, leave=True) as pbar:
        for epoch in pbar:
            total_loss = 0.0


            for sequence in data:
                inputs = sequence[:, :-1, :]
                targets = sequence[:, 1:, :]
                
                optimizer.zero_grad()
                outputs, _  = model(inputs)

                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / (len(data) - 1)
            
            old_lr = optimizer.param_groups[0]['lr']
            scheduler.step(avg_loss)
            if optimizer.param_groups[0]['lr'] < old_lr:
                tqdm.write(f"Epoch {epoch}: Reducing learning rate to {optimizer.param_groups[0]['lr']}")

            if epoch % 10 == 0:
                tqdm.write(f"Epoch {epoch}, Avg Loss: {avg_loss}")

    return model


def predict_path(model, start_sequence, end_sequence, num_points=30):
    start_matrix = start_sequence[0, -1, :-1]
    predicted_path = [start_matrix]

    current_input = start_sequence

    for i in range(num_points):
        next_point, _ = model(current_input)

        # Extract the matrix from the last frame of the sequence.
        next_matrix = next_point[0, -1, :-1]

        predicted_path.append(next_matrix)

        # Update the frame index for the last frame in the sequence
        next_point[0, -1, -1] = current_input[0, -1, -1] + 1

        current_input = next_point

    formatted_path = []

    for point in predicted_path:
        # Reshape the tensor into a 4x4 matrix
        reshaped_matrix = point.view(4, 4).tolist()
        formatted_path.append(reshaped_matrix)

    return formatted_path

