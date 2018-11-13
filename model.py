import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""
    dqn_type = "vanilla"

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        # Convolutional neural network to produce Q-values
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        # Fully-connected layer
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    
class DuellingQNetwork(nn.Module):
    """Actor (policy) model.
    Since the values of most states don't vary across actions, 
    duelling Q networks estimate state values and capture the 
    difference actions make in each state, the 'advantage'.
    """
    dqn_type = "duelling"
    
    
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        
        # Advantage for each action
        self.fc2_advantage = nn.Linear(fc1_units, fc2_units)
        # State value for the action function
        self.fc2_value = nn.Linear(fc1_units, fc2_units)
        # Advantage, state values branch into own fully-connected layers
        self.fc3_advantage = nn.Linear(fc2_units, action_size)
        self.fc3_value = nn.Linear(fc2_units, 1)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        x = F.relu(self.fc1(state))
        advantage = F.relu(self.fc2_advantage(x))
        value = F.relu(self.fc2_value(x))
        advantage = self.fc3_adv(advantage)
        value = self.fc3_value(value)
        # Obtain final Q values by combining state and advantage values
        result = value + advantage - advantage.mean()
        return result