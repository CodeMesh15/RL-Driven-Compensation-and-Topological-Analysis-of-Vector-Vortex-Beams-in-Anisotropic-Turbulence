import gymnasium as gym
from gymnasium import spaces
import numpy as np
import turbulence_engine
from skimage.metrics import structural_similarity as ssim

class TurbulenceEnv(gym.Env):
    def __init__(self, baseline_matrix=None):
        super(TurbulenceEnv, self).__init__()
        
        # If no lab data is provided, create a dummy baseline (the 'Golden Standard')
        if baseline_matrix is None:
            self.baseline = self._generate_dummy_vortex()
        else:
            self.baseline = baseline_matrix

        # AI Action: 14 Zernike coefficients
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(14,), dtype=np.float32)
        
        # AI Observation: The 2D matrix (S1 Stokes parameter)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(1, 256, 256), dtype=np.float32)

    def _generate_dummy_vortex(self):
        """Generates a dummy 256x256 vortex-like baseline."""
        x = np.linspace(-1, 1, 256)
        y = np.linspace(-1, 1, 256)
        xx, yy = np.meshgrid(x, y)
        r = np.sqrt(xx**2 + yy**2)
        # Creating a simple 'Fried Egg' intensity profile
        beam = np.exp(-r**2) * (r > 0.1) 
        return beam.astype(np.float64)

    def step(self, action):
        # Apply the AI's predicted phase compensation
        # In a real setup, we subtract the action from the turbulence
        corrected = turbulence_engine.apply_turbulence(self.current_state, action.tolist())
        
        # Calculate Reward based on similarity to the Cold Baseline
        current_ssim = ssim(self.baseline, corrected, data_range=1.0)
        reward = current_ssim - self.last_ssim
        self.last_ssim = current_ssim
        self.current_state = corrected
        
        # Terminate if we reach 98% similarity
        terminated = bool(current_ssim > 0.98)
        truncated = False
        
        return self._get_obs(corrected), reward, terminated, truncated, {"ssim": current_ssim}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Apply random 'Heat' (0.1 to 0.5 intensity) to the baseline to start
        random_heat = np.random.uniform(-0.5, 0.5, 14).tolist()
        self.current_state = turbulence_engine.apply_turbulence(self.baseline, random_heat)
        self.last_ssim = ssim(self.baseline, self.current_state, data_range=1.0)
        return self._get_obs(self.current_state), {}

    def _get_obs(self, matrix):
        return np.expand_dims(matrix, axis=0).astype(np.float32)
