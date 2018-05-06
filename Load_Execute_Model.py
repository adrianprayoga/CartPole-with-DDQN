import gym
import numpy as np
from keras.models import model_from_json
import sys, getopt
import warnings
from tqdm import tqdm

if not sys.warnoptions:
    warnings.simplefilter("ignore")

def read_arg(argv):
	try:
		opts, args = getopt.getopt(argv, 'N:f:p:r:')
		options = dict(opts)

		problem = options.get('-p','CartPole-v0')
		filename = options.get('-f','CartPole_FullDDQN')
		num_run = options.get('-N',100)
		render = options.get('-r',0)

		if(type(num_run) is int()):
			print(type(num_run))
			print("!! Incorrect input type for -N (Number of runs)")
			sys.exit(2)
		if(type(render) is int()):
			print("!! Incorrect input type for -r (Number of render)")
			sys.exit(2)

	except getopt.GetoptError:
		print('Error in reading argument(s)')
		sys.exit(2)

	return int(num_run), filename, problem, int(render)

if __name__ == "__main__":
	num_run, filename, problem, render = read_arg(sys.argv[1:])
	
	# load json and create model
	json_file = open(filename+'.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	# load weights into new model
	loaded_model.load_weights(filename+".h5")
	print("Sucessfully Loaded model from disk")

	try:
		env = gym.make(problem)
	except:
		print("!! Incorrect Environment Input Provided. Please check gym documentation.")
		sys.exit(2)
	
	state_count = env.observation_space.shape[0]
	hist_r = []

	for i in tqdm(range(num_run)):
	    total_r = 0
	    s = env.reset()
	    while True:
	        if (i<render): env.render()
	        act = loaded_model.predict(np.reshape(s, [-1, state_count]))
	        act = np.argmax(act[0])
	        state, reward, done, _ = env.step(act)
	        s=state
	        total_r += reward

	        if done:
	            if (i<render): env.close()
	            break
	            

	    hist_r.append(total_r)

	print("Execution completed. Out of", num_run, "runs, your model average is", \
		np.average(hist_r), "and", np.average((np.array(hist_r)==200))*100, "% of runs achieved maximum rewards.")
    