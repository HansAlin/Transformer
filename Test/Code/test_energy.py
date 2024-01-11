import subprocess

import subprocess

def get_gpu_info():
    try:
        _output_to_list = lambda x: x.split('\n')[:-1]

        command = "nvidia-smi --query-gpu=index,name,utilization.gpu,memory.total,memory.used,memory.free,power.draw --format=csv"
        gpu_info = subprocess.check_output(command.split(), universal_newlines=True)
        gpu_info = _output_to_list(gpu_info)

        # the first line is the header
        gpu_info = gpu_info[1:]

        gpu_info = [x.split(', ') for x in gpu_info]
        gpu_info = [[int(x[0]), x[1], float(x[2].split(' ')[0]), x[3], x[4], x[5], x[6]] for x in gpu_info]

        return gpu_info
    except Exception as e:
        print(f"Error: {e}")
        return None


def get_energy(device_number):
    gpu_info = get_gpu_info()
    if gpu_info is None:
        return None
    try:
        energy = gpu_info[device_number][-1]
        energy = float(energy.split(' ')[0])
        return energy
    except Exception as e:
        print(f"Error: {e}")
        return None


config = {'current_epoch':0}
config['power'] = 2
power = [3,1,1,1,1,1]

for e in power:
    config['current_epoch'] += 1
    config['power'] = ((config['current_epoch'])*config['power'] + e)/(config['current_epoch'] + 1)

power.append(2)
mean = sum(power)/len(power)
print(config['power'])
print(mean)