from cv2 import VideoWriter, VideoWriter_fourcc
from datetime import datetime
from matplotlib import pyplot as plt
from os import makedirs
import torch

def log(topic, message):
	message = f'[{topic} {datetime.now().strftime("%F %T.%f")}] {message}'
	print(message)
	makedirs('logs', exist_ok=True)
	with open(f'logs/{topic}.txt', 'a', encoding='utf-8') as f:
		f.write(message + '\n')

def checkpoint(agent, name):
	makedirs('checkpoints', exist_ok=True)
	torch.save(agent.state_dict(), f'checkpoints/{name}.pt')
	log('train', f'Checkpointed agent {name}')

def plot(data, title, avg_each=100):
	plt.figure(figsize=(16, 6))
	plt.xlabel('Episode')
	plt.ylabel(title)
	flatten_data = [e for d in data for e in d]
	flatten_data = flatten_data[:avg_each * (len(flatten_data) // avg_each)]
	plt.plot(range(0, len(flatten_data), avg_each), torch.tensor(flatten_data).view(-1, avg_each).mean(1))
	total = 0
	for d in data:
		total += len(d)
		plt.axvline(x=total, color='k', linestyle='--')
	makedirs('plots', exist_ok=True)
	plt.savefig(f'plots/{title}.png')
	plt.close()

def record_video(boards, name):
	palette = torch.tensor([[0, 0, 0], [255, 255, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0], [255, 0, 0]], dtype=torch.uint8)
	video = palette[boards]
	scaling_factor = video.shape[-2] * 40
	video = torch.nn.functional.interpolate(
		video.permute(0, 3, 1, 2).float(),
		size=(scaling_factor, scaling_factor),
		mode='nearest'
	).byte().permute(0, 2, 3, 1).numpy()
	fourcc = VideoWriter_fourcc(*'mp4v')
	makedirs('videos', exist_ok=True)
	out = VideoWriter(f'videos/{name}.mp4', fourcc, 10, (scaling_factor, scaling_factor))
	for frame in video: out.write(frame)
	out.release()