###############################################################
# ECE517: Reinforcement Learning in Artificial Intelligence   #
# Project 1: Markov Decision Process and Dynamic Programming  #
#                                                             #
# Writen By:                                                  #
# Ronald Randolph and John Geissberger Jr.                    #
###############################################################

import sys
import numpy as np


#argument check
if(len(sys.argv) != 9):
	print("usage: python3 iceCream.py M p_{min} p_{max} p_{same} t_{drive} t_{walk} t_{wait} r_{no}")
	sys.exit(1)

#initialization function
def initialize():
	
	M = int(sys.argv[1])
	p_min = float(sys.argv[2])
	p_max = float(sys.argv[3])

	#initialize state array
	S = np.arange(1,(2*M)+1,1)

	#initialize value function
	V = np.ones((2*M)+1)
	V[2*M] = 0
	
	#initialize random policy
	Pi = np.zeros(len(S),dtype=int)
	
	#initialize probability array
	p = np.linspace(p_min, p_max, num=M)
	
	#return inialized data structures
	return S,V,Pi,p


#policy iteration function
#this function evaluates the policy 
def policy_iteration(S,V,Pi,p):

	delta = 9.99
	theta = 0.1
	gamma = 0.50

	#continue until convergence
	while(delta > theta):
		delta = 0.00

		for s in range(len(S)):
			
			#store current value for state s
			v = V[s]

			#calculate new value for V[s]
			V[s] = action_val(Pi[s],s,p,V,gamma)
			
			#calculate delta
			delta = max(delta, abs(v-V[s]))


	#begin policy improvement
	policy_improvement(S,Pi,p,V,gamma)

	return
	

#calculates policy from value function
def policy_improvement(S,Pi,p,V,gamma):

	stable = True

	#sweep through policy for each state
	for s in range(len(S)):
		a_vals = list()
		old_action = Pi[s]

		#grab values or all possible actions at s
		for a in range(3):
			a_vals.append(action_val(a,s,p,V,gamma))

		#set policy to best action
		Pi[s] = np.argmax(a_vals)

		#check stability of policy
		if(old_action != Pi[s]):
			stable = False
	
	#if unstable -> re-evaluate
	if(not stable):
		policy_iteration(S,V,Pi,p)
	
	#if stable -> print & return
	else:
		print("Optimal Value Function")
		print(V)

		print("\nOptimal Policy")
		print(Pi)
		return

#solves using value iteration
def value_iteration(S,V,Pi,p):
	
	delta = 2.0
	gamma = 0.50
	theta = 0.1
	
	#continue until convergence
	while(delta > theta):
		delta = 0.00

		#loop for each state in S
		for s in range(len(S)):
			a_vals = list()	
			#store current value for state s
			v = V[s]

			#grab values or all possible actions at s
			for a in range(3):
				a_vals.append(action_val(a,s,p,V,gamma))

			#store highest value for V[s]
			V[s] = max(a_vals)
			
			#calculate delta
			delta = max(delta, abs(v-V[s]))

	#begin policy improvement
	output_policy(S,V,Pi,p,gamma)
	return

def output_policy(S,V,Pi,p,gamma):
	
	#sweep through policy for each state
	for s in range(len(S)):
		a_vals = list()

		#get action values for each a in A(s)
		for a in range(3):
			a_vals.append(action_val(a,s,p,V,gamma))

		#greedily update policy by value function 
		Pi[s] = np.argmax(a_vals)
	
	#print and return
	print("Optimal Value Function")
	print(V)
	
	print("\nOptimal Policy")
	print(Pi)
	return Pi

#returns summation of (eq) for all a
def action_val(a,s,p,V,gamma):

	M = int(sys.argv[1])
	p_same = float(sys.argv[4])
	t_drive = (-1 * int(sys.argv[5]))
	t_walk = (-1 * int(sys.argv[6]))
	t_wait = (-1 * int(sys.argv[7]))
	r_No = (-1 * int(sys.argv[8]))
	r_Yes = (int(sys.argv[8])+50)

	TS = (2*M)
	ret = 0.0000

	#action = drive
	if(a == 0):
		
		#drove past last spot -> no ice cream
		if(s == (M-1) or s == ((M*2)-1)):
			ret += (1 * (r_No + gamma * V[TS]))
		
		else:
			#adjust index for taken spots
			if(s >= M):
				i = (s-M)
			else:
				i = s

			#next spot is taken or next spot is open 
			ret += (p[i+1] * (t_drive + gamma * V[i+1]))
			ret += ((1-p[i+1]) * (t_drive + gamma * V[i+1+M]))

	#action = wait
	elif(a == 1):

		#wait while spot is free
		if(s >= M):
			i = (s-M)
			
			#spot stays open or spot gets taken
			ret += (p_same * (t_wait + gamma * V[i+M]))
			ret += ((1-p_same) * (t_wait + gamma * V[i]))

		#wait while spot is taken
		else:
			i = s
		
			#spot stays taken or spot becomes open
			ret += (p_same * (t_wait + gamma * V[i]))
			ret += ((1-p_same) * (t_wait + gamma * V[i+M]))


	#action = park
	elif(a == 2):

		#spot is taken
		if(s < M):

			#crash -> no ice cream :(
			ret += (1 * (r_No + gamma * V[TS]))

		#spot is free
		else:
			
			#park -> walk to ice cream :)
			dist = M - (s-M+1)
			ret += (1 * ((r_Yes + (t_walk * dist)) + gamma * V[TS]))

	return ret

#driver code
def main():
		
	#a - action			   | Pi - Policy
	#s - current state	   | gamma - Discount Fct.
	#p - transition matrix | V - value function
	
	#def_process: 1 for value iteration, 0 for policy iteration
	def_process = 1

	#initialization
	S,V,Pi,p = initialize()
	
	if(def_process == 1):
		#DP - Value Iteration
		value_iteration(S,V,Pi,p)
	
	else:
		#DP - Policy Iteration
		policy_iteration(S,V,Pi,p)


if __name__ == "__main__":
	main()


