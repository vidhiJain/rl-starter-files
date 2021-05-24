import shap

import argparse
import time
import numpy 
import torch
import torch.nn as nn
from tqdm import tqdm 
import gym_minigrid
import matplotlib.pyplot as plt
import utils



# Parse arguments

parser = argparse.ArgumentParser()
parser.add_argument("--env", required=True,
                    help="name of the environment to be run (REQUIRED)")
parser.add_argument("--model", required=True,
                    help="name of the trained model (REQUIRED)")
parser.add_argument("--seed", type=int, default=0,
                    help="random seed (default: 0)")
parser.add_argument("--shift", type=int, default=0,
                    help="number of times the environment is reset at the beginning (default: 0)")
parser.add_argument("--argmax", action="store_true", default=False,
                    help="select the action with highest probability (default: False)")
parser.add_argument("--pause", type=float, default=0.1,
                    help="pause duration between two consequent actions of the agent (default: 0.1)")
parser.add_argument("--gif", type=str, default=None,
                    help="store output as gif with the given filename")
parser.add_argument("--episodes", type=int, default=15,
                    help="number of episodes to visualize")
parser.add_argument("--memory", action="store_true", default=False,
                    help="add a LSTM to the model")
parser.add_argument("--text", action="store_true", default=False,
                    help="add a GRU to the model")
parser.add_argument("--savedir", default='/data/vidhi/shap',
                    help="path to the savedir for image plots")
parser.add_argument("--tag", type=str, default='new',
                    help="tag the image saved")
args = parser.parse_args()
print(args)

# Set seed for all randomness sources
utils.seed(args.seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

# Load environment
env = utils.make_env(args.env, args.seed)

for _ in range(args.shift):
    env.reset()
print("Environment loaded\n")

# Load agent
model_dir = utils.get_model_dir(args.model)
agent = utils.Agent(env.observation_space, env.action_space, model_dir,
                    device=device, argmax=args.argmax, use_memory=args.memory, use_text=args.text)
print("Agent loaded\n")

#simple image scaling to (nR x nC) size
def scale(im, nR, nC):
  nR0 = len(im)     # source number of rows 
  nC0 = len(im[0])  # source number of columns 
  return numpy.array([[ im[int(nR0 * r / nR)][int(nC0 * c / nC)]  
             for c in range(nC)] for r in range(nR)])

# def get_shap_reward(shap_values):
#     # return [numpy.sum(numpy.abs(shap_values[i])) for i in range(5)]
#     return [numpy.std(shap_values[i]) for i in range(5)]


class CriticModel(nn.Module):
    def __init__(self, agent):
        super(CriticModel, self).__init__()
        self.image_conv = agent.acmodel.image_conv
        self.critic = agent.acmodel.critic

    def forward(self, x):
        x = self.image_conv(x)
        x = x.reshape(x.shape[0], -1)
        x = self.critic(x)
        return x #.cpu().detach().numpy()


model = CriticModel(agent)
model = model.to(device)
# agent.acmodel.image_conv()
# reshape
# agent.acmodel.actor()

background = []
record_action_probs = []
max_steps = 100
# Run agent and collect (state, action) pairs
for episode in tqdm(range(args.episodes)):
    obs = env.reset()
    count = 0 
    while count < max_steps:
        img = env.render('rgb_array')
        # if args.gif:
        #     frames.append(img)

        image_input = numpy.expand_dims(obs['image'], axis=0)
        background.append(obs['image'])
        image_input = torch.tensor(image_input, device=device, dtype=torch.float).transpose(1, 3).transpose(2, 3)
        # image_input = image_input.to(device)
        action_logits = model(image_input)
        # print(action_logits)
        record_action_probs.append(action_logits.cpu().detach().numpy().copy())
        # print(record_action_probs)
        # action = numpy.argmax(action_logits, 1)
        _, action = torch.max(action_logits,1)
        obs, reward, done, _ = env.step(action[0])
        
        # agent.analyze_feedback(reward, done)
        if done:
            break

print('experience length: ', len(background))
data = torch.tensor(background, device=device, dtype=torch.float).transpose(1, 3).transpose(2, 3)
e = shap.DeepExplainer(model, torch.zeros((10, 3, 7, 7), device=device)) #data[numpy.random.choice(len(background), int(0.8*len(background)), replace=False)])
# e = shap.DeepExplainer(model, torch.zeros((10, 3,7,7), device=device))
# shap_values = e.shap_values(data[2000:2001])
# action_names = ['turn left', 'turn right', 'move forward', 'pickup', 'drop', 'toggle', 'done']
# dummy_names = ['']*7
# data[2000]
# shap.image_plot(shap_values, data[2000].cpu().detach().numpy())
# indices = numpy.random.choice(len(background), 5, replace=False)
indices = numpy.arange(5)
selected = data[indices]
action_probs = numpy.array(record_action_probs)[indices]

obss = selected.cpu().detach().numpy()
# record_dict = {action_names[j] : action_probs[i][j] for j in range(7)}

obsss = numpy.transpose(obss, (0,2,3,1))
images = []
for i, obs in enumerate(obsss):
    plt.clf()
    img = env.get_obs_render(obs)
    images.append(img)
    plt.imshow(img)
    # plt.title(record_dict)
    # plt.savefig(f'test_img_{i}')

shap_values = e.shap_values(selected)
# L1 = get_shap_reward(shap_values)
# print(L1)
shap_val = numpy.transpose(shap_values, (0,3,2,1))
shap_val = numpy.array([scale(shap_val[j], 112, 112) for j in range(5)])
# shap_val = numpy.array(shap_val)


shap.image_plot(shap_val, numpy.array(images)/255.) #,labels=[action_names]+[dummy_names]*4)
filename = f'{args.savedir}/shap_plot_critic-{args.env}_model-{args.model}_{args.tag}'
print(filename)
plt.savefig(filename +'.png') # f'{args.savedir}/shap_plot_critic_env-{args.env}_model-{args.model}')

x = action_probs
print(x)
# print(numpy.exp(x)/numpy.exp(x).sum())
# img = shap_values[0][0]
# img = numpy.transpose(img, (1, 2, 0))

# shap.image_plot(shap_val, obsss.transpose(0,2,1,3), labels=[action_names]*5)

# 


print('done')
