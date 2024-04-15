# DeepSafeMPC
This is the code for experiments of [DeepSafeMPC: Deep Learning-Based Model Predictive Control for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.06397).

## Pre-requisites

To use SafePO-Baselines, you need to install environments. Please refer to [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for more details on installation. Details regarding the installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym).

## Setup

- Python 3.8
- [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)
- PyTorch 1.13


## Navigation

* `alg/` - Implementation of LIO and PG/AC baselines
* `env/` - Implementation of the Escape Room game and wrappers around the SSD environment.
* `results/` - Results of training will be stored in subfolders here. Each independent training run will create a subfolder that contains the final Tensorflow model, and reward log files. For example, 5 parallel independent training runs would create `results/cleanup/10x10_lio_0`,...,`results/cleanup/10x10_lio_4` (depending on configurable strings in config files).
* `utils/` - Utility methods


## Examples

### Train LIO on Escape Room

* Set config values in `alg/config_room_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio er`. Default settings conduct 5 parallel runs with different seeds.
* For a single run, execute `$ python train_lio.py er`.

### Train LIO on Cleanup

* Set config values in `alg/config_ssd_lio.py`
* `cd` into the `alg` folder
* Execute training script `$ python train_multiprocess.py lio ssd`.
* For a single run, execute `$ python train_ssd.py`.

## Citation

<pre>
@article{yang2020learning,
  title={Learning to incentivize other learning agents},
  author={Yang, Jiachen and Li, Ang and Farajtabar, Mehrdad and Sunehag, Peter and Hughes, Edward and Zha, Hongyuan},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  pages={15208--15219},
  year={2020}
}
</pre>

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
