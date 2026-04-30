from stable_baselines3 import PPO
from env import TurbulenceEnv
import os

env = TurbulenceEnv()

# We add policy_kwargs to tell SB3: "Do not treat this as a JPEG, treat it as pure math"
model = PPO(
    "CnnPolicy", 
    env, 
    verbose=1, 
    device="cpu", 
    policy_kwargs=dict(normalize_images=False)
) 

print("Starting Training on Codespace CPU...")
model.learn(total_timesteps=10000)

model.save("ppo_turbulence_compensator")
print("Training complete. Model saved!")