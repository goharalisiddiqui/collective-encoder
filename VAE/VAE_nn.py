##################################
# Define Networks
################################## 

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from dataclasses import dataclass
from functools import reduce

from typing import List

torch.manual_seed(5) ## don't know the utility of this yet



device = "cpu"



def tuple_product(t):
	return reduce(lambda x, y: x * y, t, 1)










class ReshapeView(nn.Module): # Module to create a layer in NN that only changes the shape of the input
	def __init__(self, shape):
		super(ReshapeView, self).__init__()
		self._shape = shape

	def forward(self, x):
		return x.view(-1, *self._shape)
	
	def extra_repr(self) -> str:
		return f'reshape to [B, {self._shape}]'























class BasicVAE(nn.Module):
	def __init__(self,
				 feature_dims : int = 10, latent_dim : int = 64,
				 encoding_act=None, decoding_act=None, final_act=None):
		super().__init__()
		self._latent_dim = latent_dim

		if isinstance(feature_dims, tuple): # if feature_dims is given in tuples?
			tot_features = tuple_product(feature_dims)
			im_features = tot_features//2 # im_features = Intermediate features?
			# print(tot_features)
			self.encoding_layer_0 = torch.nn.Sequential(
				torch.nn.Flatten(),
				torch.nn.Linear(tot_features,
								im_features)
			)
			
			self.decoding_layer_mu = torch.nn.Sequential(
				torch.nn.Linear(im_features, tot_features),
				ReshapeView(feature_dims)
			)
			self.decoding_layer_logvar = torch.nn.Sequential(
				torch.nn.Linear(im_features, tot_features),
				ReshapeView(feature_dims)
			)
		else:
			tot_features = feature_dims
			im_features = tot_features//2
			self.encoding_layer_0 = nn.Linear(feature_dims, im_features)
			self.decoding_layer_mu = nn.Linear(im_features, feature_dims)
			self.decoding_layer_logvar = nn.Linear(im_features, feature_dims)
		
		

		self.encoding_mu = nn.Linear(im_features, latent_dim)
		self.encoding_logvar = nn.Linear(im_features, latent_dim)

		self.decoding_layer_0 = nn.Linear(latent_dim, im_features)

		if encoding_act is not None:
			self.encoding_act = encoding_act
		else:
			self.encoding_act = F.leaky_relu

		if decoding_act is not None:
			self.decoding_act = decoding_act
		else:
			self.decoding_act = F.leaky_relu

		print("Encoding:\n", end=" ")
		print(tot_features, " --> ", im_features, end=" ")
		print("(leaky relu)")
		print(im_features, " --> ", latent_dim, end=" ")
		print("(mu for latent space)")
		print( "  ", " \--> ", latent_dim, end=" ")
		print("(logvar for latent space)\n\n")

		print("Decoding:\n", end=" ")
		print(latent_dim, " --> ", im_features, end=" ")
		print("(leaky relu)")
		print(im_features, " --> ", latent_dim, end=" ")
		print("(mu for input space)")
		print( "  ", " \--> ", latent_dim, end=" ")
		print("(logvar for input space)\n\n")


	def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
		self.normIn = True
		self.Mean = Mean
		self.Range = Range

	def normalize(self, x):
		batch_size = x.size(0)
		x_size = x.size(1)

		Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
		Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

		return x.sub(Mean).div(Range)
	
	def denormalize(self, x):
		batch_size = x.size(0)
		x_size = x.size(1)

		Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
		Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

		return x.mul(Range).add(Mean)

	def encode(self, x):
		m, l = self.encode_(x)
		if self.training:
			return self.reparametrize(m, l)
		else:
			return m
		
	def encode_(self, x):
		x = self.encoding_act(self.encoding_layer_0(x))
		mu = self.encoding_mu(x)
		logvar = self.encoding_logvar(x)
		return mu, logvar

	def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
		std = torch.exp(0.5*logvar) ### Why 0.5? maybe std = standard deviation
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, x):
		x = self.decoding_act(self.decoding_layer_0(x))
		mu = self.decoding_layer_mu(x)
		logvar = self.decoding_layer_logvar(x)
		return mu, logvar

	def forward(self, x):
		mu_latent, logvar_latent = self.encode_(x)
		z = self.reparametrize(mu_latent, logvar_latent)
		mu_x, logvar_x = self.decode(z)
		if self.training:
			x_out = self.reparametrize(mu_x, logvar_x)
		else:
			x_out = mu_x
		return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
					   "mu_x" : mu_x, "logvar_x" : logvar_x}





















































class BasicVAR(nn.Module):
	def __init__(self, arch : list):
		super().__init__()
		self._latent_dim = arch[-1]
		self._feature_dim = arch[0]
		self.normIn = False
		self._sample_size = 1000



		#### Setting up the layer variables prior ####
		self.mean_z = torch.zeros(self._latent_dim)
		self.std_z = torch.ones(self._latent_dim)
		self.q_z = torch.distributions.normal.Normal(loc=self.mean_z, scale=self.std_z)

		#### Setting up the layers of the netwrok ####
		print(f"[Initiating a BasicVAR Network with architecture: {arch}]")
		print("")
		assert len(arch) > 3
		print("========= NN =========")
		encoder_layers = []
		for i in range(len(arch) - 2):
			print(arch[i], " --> ", arch[i + 1], end=" ")
			encoder_layers.append(nn.Linear(arch[i], arch[i + 1]))
			encoder_layers.append(nn.ReLU(True))
			print("(relu)")
			encoder_layers.append(nn.BatchNorm1d(arch[i + 1]))
			print("(batch_normalization layer)")

		self.encoder_hidden = nn.Sequential(*encoder_layers)

		self.encoder_mu = nn.Linear(arch[-2], arch[-1])
		print(arch[-2], " --> ", arch[-1], end=" ")
		print("(mu for latent space)")
  
		self.encoder_logvar = nn.Linear(arch[-2], arch[-1])
		print( "  ", " \--> ", arch[-1], end=" ")
		print("(logvar for latent space)\n\n")
  

		print("(Reparameterization Sampler)\n\n")
  
		decoder_layers = []

		a = len(arch) - 1
		for i in range(len(arch) - 2):
			print(arch[a - i], " --> ", arch[a - i - 1], end=" ")
			decoder_layers.append(nn.Linear(arch[a- i], arch[a - i - 1]))
			decoder_layers.append(nn.ReLU(True))
			print("(relu)")
			decoder_layers.append(nn.BatchNorm1d(arch[a - i - 1]))
			print("(batch_normalization layer)")
   
		self.decoder_hidden = nn.Sequential(*decoder_layers)

		self.decoder_mu = nn.Linear(arch[1], arch[0])
		print(arch[1], " --> ", arch[0], end=" ")
		print("(mu for feature space)")
  
		self.decoder_logvar = nn.Linear(arch[1], arch[0])
		print( "  ", " \--> ", arch[0], end=" ")
		print("(logvar for feature space)\n\n")
		print("======================")
  
  


	def set_sample_size(self, sample_size):
		self._sample_size = sample_size

	def set_norm(self, Mean: torch.Tensor, Range: torch.Tensor):
		self.normIn = True
		self.Mean = Mean
		self.Range = Range

	def normalize(self, x):
		batch_size = x.size(0)
		x_size = x.size(1)

		Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
		Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

		return x.sub(Mean).div(Range)
	
	def denormalize(self, x):
		batch_size = x.size(0)
		x_size = x.size(1)

		Mean = self.Mean.unsqueeze(0).expand(batch_size, x_size)
		Range = self.Range.unsqueeze(0).expand(batch_size, x_size)

		return x.mul(Range).add(Mean)

	def encode(self, x):
		m, l = self.encode_(x)
		if self.training:
			return self.reparametrize(m, l)
		else:
			return m
		
	def encode_(self, x):
		if self.normIn:
			x = self.normalize(x)
		x = self.encoder_hidden(x)
		mu = self.encoder_mu(x)
		logvar = self.encoder_logvar(x)
		return mu, logvar

	def reparametrize(self, mu, logvar): # Drawing a random sample from the normal distribution mu, logvar
		std = torch.exp(0.5*logvar) ### Why 0.5? maybe std = standard deviation
		eps = torch.randn_like(std)
		return mu + eps*std

	def decode(self, x):
		x = x.to(device)
		x = self.decoder_hidden(x)
		mu = self.decoder_mu(x)
		logvar = self.decoder_logvar(x)
		return mu, logvar

	def forward(self, x):
		mu_latent, logvar_latent = self.encode_(x) # p(z|x)
		if self.training:
			z = self.reparametrize(mu_latent, logvar_latent)
		else:
			z = mu_latent

		mu_x, logvar_x = self.decode(z) # q(x|z)
		if self.training:
			x_out = self.reparametrize(mu_x, logvar_x)
		else:
			x_out = mu_x

		return x_out, {"mu_latent" : mu_latent, "logvar_latent" : logvar_latent,
					   "mu_x" : mu_x, "logvar_x" : logvar_x}

	def generate_frames(self, n_samples_z : int = 10, n_samples_x : int = 100):
		samples_z = torch.randn((n_samples_z, self._latent_dim))
		mu_x, logvar_x = self.decode(samples_z)
		# print(f"mu_x shape.before={mu_x.shape}")
		mu_x = mu_x.repeat(n_samples_x, 1)
		logvar_x = logvar_x.repeat(n_samples_x, 1)
		# print(f"mu_x shape.after={mu_x.shape}")
		x_out = self.reparametrize(mu_x, logvar_x)
		return x_out





























#################################                       ####################################
################################# Probability functions ####################################
#################################                       ####################################

	def q_x_given_z(self, samples_z): ## q(x|z)
		mu, logvar  = self.decode(samples_z.view(-1, self._latent_dim))
		qxgz = torch.distributions.normal.Normal(loc=mu, scale=logvar.exp().sqrt())
		return qxgz

	def E_log_q_x_given_z(self, samples_x, samples_z): ## Expecation value of log q(x|z) calculated at samples_x and samples_z
		qxgz = self.q_x_given_z(samples_z)
		E_log_qxgz = qxgz.log_prob(samples_x.view(-1, self._feature_dim)).sum(dim=1)
		return E_log_qxgz

	def r_z_given_x(self, samples_x): ## r(z|x)
		mu, logvar  = self.encode(samples_x.view(-1, self._feature_dim))
		rzgx = torch.distributions.normal.Normal(loc=mu, scale=logvar.exp().sqrt())
		return rzgx

	def E_log_r_z_given_x(self, samples_x, samples_z): ## Expecation value of log r(z|x) calculated at samples_x and samples_z
		rzgx = self.r_z_given_x(samples_x)
		E_log_rzgx = rzgx.log_prob(samples_z.view(-1, self._latent_dim)).sum(dim=1)
		return E_log_rzgx
	
	def q_z(self): ## Prior q(z)
		return self.q_z

	def E_log_q_z(self, samples_z): ## Expecation value of log q(z) prior calculated on samples_z
		qz = self.q_z()
		E_log_q_z = qz.log_prob(samples_z.view(-1, self._latent_dim)).sum(dim=1)
		return E_log_q_z

	def log_p_x(self, samples_x, n_samples_z : int = None):
		if n_samples_z == None:
			n_samples_z - self._sample_size
		n_samples_x = samples_x.shape[0]
		samples_z = self.q_z.sample((n_samples_z,))
		qxgz = self.q_x_given_z(samples_z)
		log_qxgz = torch.zeros((n_samples_x, n_samples_z))
		for i in range(n_samples_x):
			log_qxgz[i, :] = qxgz.log_prob(samples_x[i, :]).sum(dim=1)
		
		log_qxgz_max, _ = log_qxgz.max(dim=1, keepdim=True)
		xma = log_qxgz - log_qxgz_max
		expxma = torch.exp(xma)
		sumexpxma = expxma.sum(dim=1, keepdim=True)
		logsumtemp = torch.log(sumexpxma)
		log_qx = log_qxgz_max + logsumtemp
		log_qx.add_(-np.log(n_samples_z))


#############################################################################################
#############################################################################################
#############################################################################################

