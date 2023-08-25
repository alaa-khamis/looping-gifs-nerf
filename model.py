import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm

class LSTMModel(nn.Module):
    def __init__(self, input_size=16, hidden_size=100, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        x = x.unsqueeze(1)  # Adds a sequence dimension
        out, _ = self.lstm(x, (h0, c0))
        out = self.linear(out[:, -1, :])
        return out

def train_model(data, model, epochs=100, lr=0.001, clip_norm=5):
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # A simple learning rate scheduler can help in some cases
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=False)
    
    with tqdm(range(epochs), total=epochs, desc="Training model", position=0, leave=True) as pbar:
        for epoch in pbar:
            total_loss = 0.0

            for i in range(len(data) - 1):
                inputs = data[i].unsqueeze(0)
                targets = data[i + 1].unsqueeze(0)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)

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

def predict_path(model, start_point, end_point, num_points=30):
    predicted_path = [start_point]
    current_input = start_point.unsqueeze(0)

    for _ in range(num_points):
        
        next_point = model(current_input)
        predicted_path.append(next_point.squeeze(0))
        
        # Break the loop if the next_point is very close to the end_point
        if torch.all(torch.abs(next_point - end_point) < 0.5):
            break
            
        current_input = next_point

    # Convert tensors to nested lists and ensure the format
    formatted_path = []
    for point in predicted_path:
        # Reshape the tensor into a 3x4 matrix.
        reshaped_matrix = point.view(4, 4).tolist()
        
        formatted_path.append(reshaped_matrix)

    return formatted_path

# Generate set of linearly interpolated points
def interpolate(start, end, num_points):
    alphas = torch.linspace(0, 1, num_points).unsqueeze(-1).cuda()
    interpolated_points = (1 - alphas) * start + alphas * end
    return interpolated_points

# Generate path using the model based on the liniearly interpolated points
def refine_path(model, start_point, end_point, num_points=30):
    interpolated_points = interpolate(start_point, end_point, num_points)
    
    refined_path = [start_point]
    
    for i in range(1, len(interpolated_points)):
        current_input = interpolated_points[i-1].unsqueeze(0)
        target_point = interpolated_points[i].unsqueeze(0)
        
        refined_point = model(current_input)
        
        # We can use a weighted average of the refined_point and target_point
        # This step will help to keep the refined path closer to the original interpolation.
        refined_point = 0.85 * refined_point + 0.15 * target_point

        refined_path.append(refined_point.squeeze(0))

    formatted_path = []
    for point in refined_path:
        reshaped_matrix = point.view(4, 4).tolist()
        formatted_path.append(reshaped_matrix)

    return formatted_path