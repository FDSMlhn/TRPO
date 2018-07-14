import 



parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')




if len(sys.argv) > 1:
    task = sys.argv[1]
else:
    task = "RepeatCopy-v0"

env = envs.make(task)
env.monitor.start(training_dir)

env = SpaceConversionEnv(env, Box, Discrete)

agent = TRPOAgent(env)
agent.learn()
env.monitor.close()
gym.upload(training_dir,
           algorithm_id='trpo_ff')
