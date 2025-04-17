import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from config.settings import TransformerSettings
from models.transformer.model import TransformerModel
from tqdm import tqdm

class TransformerTrainer:
    def __init__(self, experiment_id):
        self.config = TransformerSettings()
        self.experiment_id = experiment_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
    def train(self, X_train, y_train, X_val, y_val, num_classes, class_weights):
        # Convert numpy arrays to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train) if num_classes == 1 else torch.LongTensor(y_train)
        X_val = torch.FloatTensor(X_val)
        y_val = torch.FloatTensor(y_val) if num_classes == 1 else torch.LongTensor(y_val)
        
        # Create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=self.config.BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.config.BATCH_SIZE)
        
        # Initialize model
        model = TransformerModel(X_train.shape[1:], num_classes).to(self.device)
        print(f"\nModel architecture:\n{model}\n")
        
        # Loss function
        if num_classes == 1:
            criterion = nn.BCELoss()
        else:
            if class_weights is not None:
                class_weights = torch.FloatTensor(list(class_weights.values())).to(self.device)
            criterion = nn.CrossEntropyLoss(weight=class_weights)
            
        # Optimizer
        optimizer = optim.Adam(model.parameters(), lr=self.config.LEARNING_RATE)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=self.config.PATIENCE//2)
        
        # Training history
        history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_state = None
        
        print("\nStarting training...")
        print(f"Training on {len(train_loader.dataset)} samples")
        print(f"Validating on {len(val_loader.dataset)} samples")
        
        # Training loop
        for epoch in range(self.config.EPOCHS):
            # Training phase
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            train_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Train]')
            for batch_X, batch_y in train_bar:
                batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs = model(batch_X)
                
                if num_classes == 1:
                    loss = criterion(outputs.squeeze(), batch_y)
                    predictions = (outputs.squeeze() > 0.5).float()
                else:
                    loss = criterion(outputs, batch_y)
                    predictions = outputs.argmax(dim=1)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                train_correct += (predictions == batch_y).sum().item()
                train_total += batch_y.size(0)
                
                # Update progress bar
                train_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{train_correct/train_total:.4f}'
                })
            
            # Validation phase
            model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                val_bar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{self.config.EPOCHS} [Val]')
                for batch_X, batch_y in val_bar:
                    batch_X, batch_y = batch_X.to(self.device), batch_y.to(self.device)
                    outputs = model(batch_X)
                    
                    if num_classes == 1:
                        loss = criterion(outputs.squeeze(), batch_y)
                        predictions = (outputs.squeeze() > 0.5).float()
                    else:
                        loss = criterion(outputs, batch_y)
                        predictions = outputs.argmax(dim=1)
                    
                    val_loss += loss.item()
                    val_correct += (predictions == batch_y).sum().item()
                    val_total += batch_y.size(0)
                    
                    # Update progress bar
                    val_bar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{val_correct/val_total:.4f}'
                    })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss / len(train_loader)
            epoch_val_loss = val_loss / len(val_loader)
            epoch_train_acc = train_correct / train_total
            epoch_val_acc = val_correct / val_total
            
            # Update history
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)
            history['train_acc'].append(epoch_train_acc)
            history['val_acc'].append(epoch_val_acc)
            
            # Print epoch summary
            print(f'\nEpoch {epoch+1}/{self.config.EPOCHS}:')
            print(f'Train Loss: {epoch_train_loss:.4f} - Train Acc: {epoch_train_acc:.4f}')
            print(f'Val Loss: {epoch_val_loss:.4f} - Val Acc: {epoch_val_acc:.4f}')
            
            # Learning rate scheduling
            scheduler.step(epoch_val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate: {current_lr:.6f}')
            
            # Early stopping
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                best_model_state = model.state_dict()
                print('New best model saved!')
            else:
                patience_counter += 1
                if patience_counter >= self.config.PATIENCE:
                    print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
        
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
            print('\nRestored best model from checkpoint')
        
        return model, history 