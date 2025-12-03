import numpy as np
import torch
import argparse

#from utility import grab
import LFT_Phi4
from obs import Obs
from analisi import gamma_analysis_work_rep

# Parser
parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='scalingData', help='Output file of results')
parser.add_argument('--L1', type=int, default=8, help='length of the lattice side L1')
parser.add_argument('--L2', type=int, default=8, help='length of the lattice side L2')
parser.add_argument('--T', type=int, default=8, help='length of the lattice side T in the time direction')
parser.add_argument('--nmeas', type=int, default=100, help='number of measurements')
parser.add_argument('--steps', type=int, default=1000, help='out of equilibrium steps')
parser.add_argument('--thermal', type=int, default=10000, help='thermalisation steps')
parser.add_argument('--BatchSize', type=int, default=20, help='number of lattices simulated')
parser.add_argument('--kappa', type=float, default=0.18670475, help='number of measurements')
parser.add_argument('--lam', type=float, default=0.1, help='number of measurements')
parser.add_argument('--HotStart', type=bool, default=False, help='whether the system begins cold (randomly chosen 1 or -1, equal for all lattice sites) or hot (random numbers)')
args = parser.parse_args()

def grab(var):
  if torch.is_tensor(var):
    return var.detach().cpu().numpy()
  else:
    return var

def hc_start(lattice_shape,scaling_noise = None):
  '''
  Hot start: gaussian:
  Cold start: all +1 or -1 at fixed replica
  '''
  if scaling_noise:
    return torch.randn(lattice_shape)*scaling_noise
  else:
    bs,L1,L2,T = lattice_shape
    signs = torch.randint(0, 2, (bs, 1, 1, 1)) * 2 - 1
    # torch.randint(0,2) -> 0 o 1
    # *2 -1 -> mappa 0->-1, 1->1

    # 2. espandi lungo le altre dimensioni
    x = signs.expand(bs, L1, L2, T).clone()
    return x

def gamma_analysis_single_measure(W_list):
  replicas_name = [f'ensemble1|r0{i+1}' for i in range(len(W_list))]

  # <W>
  W_Obs = Obs([W_list], [replicas_name])
  W_Obs.gamma_method()

  # Delta F
  w0=np.mean(W_list)
  obs1 = Obs([np.exp(-W_list + w0)], [replicas_name])
  DeltaF_Obs=-np.log(obs1)+w0
  DeltaF_Obs.gamma_method()

  # ESS
  obs2= Obs([np.exp(2*(-W_list+w0))], [replicas_name])
  ESS_Obs = obs1**2/obs2
  ESS_Obs.gamma_method()

  return W_Obs, DeltaF_Obs, ESS_Obs

# --- Main code --- #

k = args.kappa
l = args.lam
p = 0 # we begin with the periodic conditions

if torch.cuda.is_available():
  device = 'cuda'
else:
  device = 'cpu'
print("device: ", device)

BatchSize = args.BatchSize
LatticeShape = [args.L1, args.L2, args.T]
th = LFT_Phi4.Phi4Protocol(k, l, p)
sim = LFT_Phi4.MC_Heatbath(LatticeShape, BatchSize, th, device = device)

phi = hc_start([BatchSize, sim.shape[0], sim.shape[1], sim.shape[2]], args.ColdStart)
#phi = torch.randn(BatchSize, sim.shape[0], sim.shape[1], sim.shape[2])
W_list_0 = LFT_Phi4.get_work(sim, args.steps, phi, thermal_steps=args.thermal, nmeas = args.nmeas)

work_np = grab(W_list_0) # Converte in numpy se è torch

if(args.nmeas == 1):
  gamma = gamma_analysis_single_measure(work_np)
else:
  gamma = gamma_analysis_work_rep(work_np)

# costruisco stringhe dei B_i con separatore "|"
matrix_part = " | ".join(" ".join(str(x) for x in B) for B in work_np)

# prendo i sei valori da gamma
gamma_str = str(gamma[0].value) + " " + str(gamma[0].dvalue) + " " 
gamma_str += str(gamma[1].value) + " " + str(gamma[1].dvalue) + " " 
gamma_str += str(gamma[2].value) + " " + str(gamma[2].dvalue)

# linea completa
line = matrix_part + " || " + gamma_str

# scrittura su file
kappa_str = str(k).split(".")[-1]
lambda_str = str(l).split(".")[-1]
filename = f"{args.output}_{args.L1}-{args.L2}-{args.T}_{kappa_str}_{lambda_str}_{BatchSize}_{args.nmeas}_{args.steps}.txt"
with open(f"luscher_data/{filename}", "a") as f:
  f.write(line + "\n")

print(f"File {args.output} salvato")