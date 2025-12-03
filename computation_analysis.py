import math
import fire
import json
import numpy as np

from retrieval import launch
from utils import get_default_config
def get_computational_statistics(**kwargs):
    cfg = get_default_config("config.yaml")
    kwargs['task'] = kwargs.get('task', cfg['GENERAL']['TASK']).lower()
    kwargs['dataset'] = kwargs.get('dataset', cfg['GENERAL']['DATASET'])
    kwargs['split'] = kwargs.get('split', cfg['GENERAL']['SPLIT']).lower()
    kwargs['extractor'] = kwargs.get('extractor', cfg['GENERAL']['EXTRACTOR'])
    kwargs['use_mllm'] = kwargs.get('use_mllm', cfg['GENERAL']['USE_MLLM']).lower()
    kwargs['mllm'] = kwargs.get('mllm', cfg['GENERAL']['MLLM']).lower()
    kwargs['image_gen_model'] = kwargs.get('image_gen_model', cfg['GENERAL']['IMAGE_GEN_MODEL']).lower()
    res={
        'seed': [],
        'avg_map': [],
        'inference_time_per_sample': [],
        'total_pipeline_time': []
    }
    for seed in [0, 10, 42]:
        kwargs['seed'] = seed
        avg_map, inference_time_per_sample, total_pipeline_time = launch(**kwargs)
        res['seed'].append(seed)
        res['avg_map'].append(avg_map)
        res['inference_time_per_sample'].append(inference_time_per_sample)
        res['total_pipeline_time'].append(total_pipeline_time/220)
    if kwargs['task'] == 'txt2img':
        filename = f"computation_analysis_{kwargs['task']}_{kwargs['mllm']}.json"
    elif kwargs['task'] == 'img2img':
        if kwargs['use_mllm'] == 'no':
            filename = f"computation_analysis_{kwargs['task']}_no_mllm_{kwargs['image_gen_model']}.json"
        else:
            filename = f"computation_analysis_{kwargs['task']}_{kwargs['mllm']}_{kwargs['image_gen_model']}.json"
    json.dump(res, open(filename, "w"))

    print("="*50)
    print(f"Computational Statistics for {kwargs['task'].lower()} with {kwargs['use_mllm'].lower()} {kwargs['mllm'].lower()} and {kwargs['image_gen_model'].lower()}:")
    print(f"Seed\tAvgMAP\tInference-Time\tPipeline-Time")
    for i in range(len(res['seed'])):
        print(f"{res['seed'][i]}\t{res['avg_map'][i]:.2f}\t{res['inference_time_per_sample'][i]:.2f}s\t{res['total_pipeline_time'][i]:.2f}s")
    print(f"Overall Results:\t{np.mean(res['avg_map']):.2f}\t{np.mean(res['inference_time_per_sample']):.2f}s\t{np.mean(res['total_pipeline_time']):.2f}s")
    print(f"Standard Deviation:\t{np.std(res['avg_map']):.2f}\t{np.std(res['inference_time_per_sample']):.2f}s\t{np.std(res['total_pipeline_time']):.2f}s")
    print("="*50)
    print(res)

if __name__ == "__main__":
    fire.Fire(get_computational_statistics)