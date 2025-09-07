import json
import fire
import os

from transformers import set_seed

from main import main
from utils import get_default_config
from figures import plot_ablation_metrics

def ablate_llm(**kwargs):
    cfg = get_default_config("config.yaml")
    set_seed(cfg['GENERAL']['SEED'])
    if kwargs.get('dataset') == 'circo' and kwargs.get('split') == 'val':
        res_file ='./ablation_llm_results.json'
        if not os.path.exists(res_file):
            res = {
                'res_temperature_keys': [x/10.0 for x in range(1, 10)],
                'res_temperature_vals': [],
                'res_top-p_keys': [x/10.0 for x in range(1, 10)],
                'res_top-p_vals': [],
                'res_top-k_keys': [10, 30, 50, 100, 150],
                'res_top-k_vals': []
            }
            for temp in res['res_temperature_keys']:
                kwargs['temperature'] = temp
                kwargs['top_p'] = 0.9
                kwargs['llm_top_k'] = 50
                res['res_temperature_vals'].append(main(cfg, **kwargs))
            for top_p in res['res_top-p_keys']:
                kwargs['temperature'] = 0.1
                kwargs['top_p'] = top_p
                kwargs['llm_top_k'] = 50
                res['res_top-p_vals'].append(main(cfg, **kwargs))
            for top_k in res['res_top-k_keys']:
                kwargs['temperature'] = 0.1
                kwargs['top_p'] = 0.9
                kwargs['llm_top_k'] = top_k
                res['res_top-k_vals'].append(main(cfg, **kwargs))
            json.dump(res, open(f"./ablation_llm_results.json", "w"))
            plot_ablation_metrics(res, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP']*3, file_path="llm_ablation.pdf")
        else:
            res = json.load(open(res_file, "r"))
            plot_ablation_metrics(res, xlabels=['Temperature', 'Top-p', 'Top-k'], ylabels=['mAP']*3, file_path="llm_ablation.pdf")

def ablate_df(**kwargs):
    cfg = get_default_config("config.yaml")
    set_seed(cfg['GENERAL']['SEED'])
    if kwargs.get('dataset') == 'circo' and kwargs.get('split') == 'val':
        res_file ='./ablation_df_results.json'
        if not os.path.exists(res_file):
            res = {
                'res_guidance_scale_keys': [7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
                'res_guidance_scale_vals': [],
                'res_num_infer_steps_keys': [10, 20, 30, 50, 100],
                'res_num_infer_steps_vals': [],
                'res_image_guidance_scale_keys': [1.0, 1.3, 1.7, 2.0, 3.0, 4.0, 5.0, 10.0],
                'res_image_guidance_scale_vals': []
            }
            for guidance_scale in res['res_guidance_scale_keys']:
                kwargs['guidance_scale'] = guidance_scale
                kwargs['n_infer_steps'] = 30
                kwargs['image_guidance_scale'] = 3.0
                res['res_guidance_scale_vals'].append(main(cfg, **kwargs))
            for num_infer_steps in res['res_num_infer_steps_keys']:
                kwargs['guidance_scale'] = 7.5
                kwargs['n_infer_steps'] = num_infer_steps
                kwargs['image_guidance_scale'] = 3.0
                res['res_num_infer_steps_vals'].append(main(cfg, **kwargs))
            for image_guidance_scale in res['res_image_guidance_scale_keys']:
                kwargs['guidance_scale'] = 7.5
                kwargs['n_infer_steps'] = 30
                kwargs['image_guidance_scale'] = image_guidance_scale
                res['res_image_guidance_scale_vals'].append(main(cfg, **kwargs))
            json.dump(res, open(f"./ablation_df_results.json", "w"))
            plot_ablation_metrics(res, xlabels=['Guidance Scale', 'Num Infer Steps', 'Image Guidance Scale'], ylabels=['mAP']*3, file_path="df_ablation.pdf")
        else:
            res = json.load(open(res_file, "r"))
            plot_ablation_metrics(res, xlabels=['Guidance Scale', 'Num Infer Steps', 'Image Guidance Scale'], ylabels=['mAP']*3, file_path="df_ablation.pdf")

if __name__ == '__main__':
    fire.Fire(ablate_df)
