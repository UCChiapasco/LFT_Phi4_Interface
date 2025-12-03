import numpy as np
#import copy
import torch
#from tqdm import tqdm
#import pyerrors as pe

if torch.cuda.is_available():
    torch_device = 'cuda'
    float_dtype = np.float32 # single
    torch.set_default_tensor_type(torch.cuda.FloatTensor)
else:
    torch_device = 'cpu'
    float_dtype = np.float64 # double
    torch.set_default_tensor_type(torch.DoubleTensor)

def grab(var):
  if torch.is_tensor(var):
    return var.detach().cpu().numpy()
  else:
    return var


# Class definition

class MC_Heatbath():
  """class for the heatbath upgrade on a dim+1 cubic lattice"""
  def __init__(self, latt_shape, bs, action_th, dim=3, nsteps=1, DEBUG=0, device = 'cuda'):
    self.shape = list(latt_shape) # Note that updating the shape does not update the masks, which generates errors. If necessary, update the masks manually in the main
    self.bs = bs # batch size
    if(dim == 2):
      self.even_mask = mask2d(self.shape,0)
      self.odd_mask = mask2d(self.shape,1)
    if(dim == 3):
      self.even_mask = mask3d(self.shape,0)
      self.odd_mask = mask3d(self.shape,1)
    self.update = self.local_heatbath
    self.action_th = action_th #defines the theory we study (we'll use the phi^4 class defined above)
    self.action_density = action_th.action_density
    self.get_action = action_th.get_action
    self.heatbathSd = 1.0 / np.sqrt(2.0 * self.action_th.phi2coeff)
    self.heatbathSd2 = 1.0 / (2.0 * self.action_th.phi2coeff)
    self.dim = dim
    self.dims = range(dim)
    self.nsteps = nsteps
    self.DEBUG = bool(DEBUG)
    if(DEBUG):
      print("DEBUG session: MC_Heatbath class. Value of k: ", self.action_th.k)
    self.device = device

  def heatbath(self, x_active, x_frozen, norm_rn, unif_rn):
    #nn_sum = torch.zeros(x_active.shape)
    #x_active_new = torch.zeros(x_active.shape)
    # the first component is different from the others since it represents
    # the batch size and not a physical dimension. This is taken into
    # account in the action theory class
    nn_sum = self.action_th.local_kineticterm(x_frozen)
    #if(self.DEBUG):
    #  print("nn_sum: ", nn_sum)
    x_active_new = norm_rn - self.heatbathSd2*nn_sum
    #print("x_new: ", x_active_new) #-----------------<
    acc = torch.exp(-self.action_th.phi4term(x_active_new))
    #print("acc: ", acc.mean())
    mask = (acc > unif_rn).float()
    #print("mask: ", mask.shape)
    x_active = x_active_new*mask + x_active*(1 - mask)
    #print("x_active: ", x_active.shape)
    return x_active

  def local_heatbath(self, x):
    x = torch.as_tensor(x)
    S_old = self.get_action(x)
    shape = tuple(x.shape)
    dist_norm = torch.distributions.normal.Normal(torch.zeros(shape), self.heatbathSd)
    dist_unif = torch.distributions.uniform.Uniform(torch.zeros(shape), torch.ones(shape))
    #if x.shape != self.shape:
    #  print("Error occurred in local_heatbath(), input shape is different from lattice shape")
    #  return 0, 0

    for _ in range(self.nsteps):

      if self.device == 'cuda':
        norm_rn = torch.cuda.FloatTensor(x.shape).normal_(0.0, self.heatbathSd)
        unif_rn = torch.cuda.FloatTensor(x.shape).uniform_(0.0, 1.0)
      else:
        norm_rn = dist_norm.sample()
        unif_rn = dist_unif.sample()

      #print("x shape: ", x.shape)
      #print("mask shape: ", self.odd_mask.shape)
      x_odd = x * self.odd_mask # each element of the batch size row is a lattice
      x_even = x * self.even_mask
      #print("S shape: ", x_even.shape, " self dims: ", self.dims)
      # --------------------------------/!\--------------------------------
      x_even = self.heatbath(x_even, x_odd, norm_rn * self.even_mask, unif_rn * self.even_mask) # potrebbe essere necessario segnalare un [:] su norm_rn, che ha una dimensione in più rispetto alla maschera
      x_odd = self.heatbath(x_odd, x_even, norm_rn * self.odd_mask, unif_rn * self.odd_mask)

      x = x_even + x_odd

    S = self.get_action(x)
    dQ = S - S_old

    return x, dQ

  def forward(self, x):
    x, dQ = self.update(x)
    return x, dQ
  
  def get_mag(self, x):
    return x.mean().item()
  
  def get_abs_mag(self, phi):
    return torch.abs(phi).mean().item()


def mask2d(shape, parity):
  """makes a checkerboard mask (array/matrix of 0 and 1 of dimension 2)"""
  a = torch.ones(shape) - parity
  a[::2, ::2] = parity
  a[1::2, 1::2] = parity
  return a

def mask3d(shape, parity):
  """makes a checkerboard mask (array/matrix of 0 and 1 of dimension 3)"""
  dim = len(shape)
  #print("mask of dim: ", dim)
  a = torch.ones(tuple(shape)) - parity
  a[::2, ::2, ::2] = parity
  a[1::2, 1::2, ::2] = parity
  a[1::2, ::2, 1::2] = parity
  a[::2, 1::2, 1::2] = parity    
  return a

class Phi4Protocol:
  """class containing elements of the phi4 theory"""
  def __init__(self, k, l, p, dim = 3, antiper_dir = 2):
    # initialize the parameters of phi^4 theory
    self.k = k
    self.l = l
    self.p = p #protocol between [0,1]
    self.dim = dim
    self.dims = range(self.dim)
    self.phi2coeff = (1.0 - 2.0 * self.l)
    self.info = '_ScalarPhi4_kappa = ' + str(k) + '_lambda = ' + str(l) # returns info on self as a string
    self.antiper_dir = antiper_dir # note that in order to change this direction you also have to change the definition of border_mask3()

  def action_density(self, cfgs):
    """takes configuration and returns the action density"""
    return self.phi2term(cfgs) + self.phi4term(cfgs) + self.kineticterm(cfgs)
    # note that one is subtracted to give antiperiodic conditions, and the other to cancel with periodic ones

  def phi2term(self, cfgs):
    return self.phi2coeff * cfgs * cfgs
  
  def phi4term(self, cfgs):
    return self.l * cfgs**4

  def kineticterm(self, cfgs):
    static_cfgs = -2 * self.k * cfgs
    temp = 0
    cfgs_shape = cfgs.shape
    border_mask = self.border_mask3(tuple(cfgs_shape))
    cfgs_inverse_border = cfgs - 2*cfgs*border_mask
    for i in range(self.dim):
      if(i == self.antiper_dir):
        temp += torch.roll(cfgs_inverse_border,-1,i+1)*static_cfgs
      else:
        temp += torch.roll(cfgs,-1,i+1)*static_cfgs
    return temp

  def border_mask3(self, shape):
    # dim = 3, direction = 3
    a = torch.zeros(shape)
    a[::,::,::,0] = 1 * self.p
    #remember that the first direction is vertical in the matrices, the second is vertical along them and the third is the horizontal along them
    return a
  
  def nearest_neighbours_sum(self, cfgs):
    #lattice_shape = cfgs[0].shape
    temp = torch.zeros(tuple(cfgs.shape)) # this must have the same shape as cfgs
    border_mask = self.border_mask3(tuple(cfgs.shape)) #* self.p
    for i in self.dims:
      temp += torch.roll(cfgs, 1, i+1) + torch.roll(cfgs, -1, i+1)
    temp -= torch.roll(2*border_mask*cfgs, -1, self.antiper_dir+1)
    temp -= torch.roll(torch.roll(border_mask, -1, self.antiper_dir+1)*2*cfgs, 1, self.antiper_dir+1)
    return temp

  def antiper_nearest_neighbours_sum(self, cfgs):
    #lattice_shape = cfgs[0].shape
    temp = torch.zeros(tuple(cfgs.shape)) # this must have the same shape as cfgs
    for i in self.dims:
      temp += torch.roll(cfgs, 1, i+1) + torch.roll(cfgs, -1, i+1)
    temp -= torch.roll(2*self.border_mask3(tuple(cfgs.shape))*cfgs,-1,self.antiper_dir+1)
    temp -= torch.roll(torch.roll(self.border_mask3(tuple(cfgs.shape)),-1,self.antiper_dir+1)*2*cfgs,1,self.antiper_dir+1)
    return temp

  #def get_hamiltonian(self, chi, cfgs):
  #  return 0.5 * torch.sum(chi**2) + self.get_action(cfgs)

  #def get_drift(self, phi): # not checked in the updating of the class
  #  return (2 * self.k * (torch.roll(phi, 1, 0) + torch.roll(phi, -1, 0) + torch.roll(phi, 1, 1) + torch.roll(phi, -1, 1)) + 2 * phi * (2 * self.l * (1 - phi**2) - 1))

  def get_action(self, phi):
    """returns the action (not action density) of the system in configuration phi"""
    #print("DEBUG: get_action is using k = ", self.k)
    return torch.sum(self.action_density(phi), dim = tuple([a+1 for a in self.dims]))
  
  def local_kineticterm(self, cfgs):
    return -2.0 * self.k * self.nearest_neighbours_sum(cfgs)
  
  def roll_with_bc():
    return 0
  
def get_ess_works(sim, layer_list, phi, W_list_fn = [], thermal_steps = 5000):
  
  for layers in layer_list:
    #if len(protocols)==0:
    #  protocols = np.linspace(0,1,layers)
    protocol_list = np.linspace(0,1,layers)

    #thermalization with periodic conditions
    sim.action_th.p = 0
    for _ in range(thermal_steps):
    #for _ in tqdm(range(1000)):
      phi, dQ = sim.update(phi)

    #non-equilibrium evolution
    Q = torch.zeros(sim.bs)
    S_0 = sim.action_th.get_action(phi)

    for i in range(layers):
    #for i in tqdm(range(layers), desc = f"{k:.2f} layers: "):
      phi, dQ = sim.update(phi)
      Q += dQ
      sim.action_th.p = protocol_list[i]

    S_f = sim.action_th.get_action(phi)
    W = S_f - S_0 - Q
    #print("average work for ", layers, " layers: ", torch.mean(W))
    W_list_fn.append(W)
  return W_list_fn

def get_work(sim, layers, phi0, thermal_steps = 10000, nmeas = 10000, sweep = 1):
  """
  compute the work along multiple trajectories and saves the data in a list W_list of length nmeas
  """
  protocol_list = np.linspace(0,1,layers)
  W_list = torch.zeros(nmeas, sim.bs)
  #thermalization with periodic conditions
  sim.action_th.p = 0
  for _ in range(thermal_steps):
  #for _ in tqdm(range(1000)):
    phi0, _ = sim.update(phi0)

  #we build nmeas starting points for the trajectories
  for measure in range(nmeas):
    # sweep of the system on periodic couplings to reduce autocorrelation
    sim.action_th.p = 0
    for _ in range(sweep):
      phi0, _ = sim.update(phi0)
    # assign phi0 -> phi, the latter evolves through Jarzynski, the former is saved for the next cycle
    phi = phi0.clone()

    #non-equilibrium evolution
    Q = torch.zeros(sim.bs)
    S_0 = sim.get_action(phi)

    for i in range(layers):
    #for i in tqdm(range(layers), desc = f"{k:.2f} layers: "):
      sim.action_th.p = protocol_list[i]
      phi, dQ = sim.update(phi)
      Q += dQ

    S_f = sim.get_action(phi)
    W = S_f - S_0 - Q
    W_list[measure] = W
    #print("average work for ", layers, " layers: ", torch.mean(W))
  return W_list

def thermalise(phi, sim, steps = 5000):
  sim.action_th.p = 0
  for _ in range(steps):
  #for _ in tqdm(range(1000)):
    phi, _ = sim.update(phi)
  return phi

def get_work_with_checkpoints(sim, layers, phi0, thermal_steps = 10000, nmeas = 10000, sweep = 1, save_data = ""):
  """
  compute the work along multiple trajectories and saves the data in a list W_list of length nmeas
  """
  protocol_list = np.linspace(0,1,layers)
  W_list = torch.zeros(nmeas, sim.bs)
  #thermalization with periodic conditions
  sim.action_th.p = 0
  for _ in range(thermal_steps):
  #for _ in tqdm(range(1000)):
    phi0, _ = sim.update(phi0)

  #we build nmeas starting points for the trajectories
  for measure in range(nmeas):
    # sweep of the system on periodic couplings to reduce autocorrelation
    sim.action_th.p = 0
    for _ in range(sweep):
      phi0, _ = sim.update(phi0)
    # assign phi0 -> phi, the latter evolves through Jarzynski, the former is saved for the next cycle
    phi = phi0.clone()

    #non-equilibrium evolution
    Q = torch.zeros(sim.bs)
    S_0 = sim.get_action(phi)

    for i in range(layers):
    #for i in tqdm(range(layers), desc = f"{k:.2f} layers: "):
      sim.action_th.p = protocol_list[i]
      phi, dQ = sim.update(phi)
      Q += dQ

    S_f = sim.get_action(phi)
    W = S_f - S_0 - Q
    W_list[measure] = W
    if(save_data != "" and save_data.split(".")[-1] == "txt"):
      work_np = grab(W_list)
      row_W = " ".join(str(x) for x in work_np)
      with open(save_data, "a") as f:
        f.write(row_W + " | ")

    #print("average work for ", layers, " layers: ", torch.mean(W))
  return W_list
