# DeepSafeMPC
This is the code for experiments of [DeepSafeMPC: Deep Learning-Based Model Predictive Control for Safe Multi-Agent Reinforcement Learning](https://arxiv.org/abs/2403.06397).

## Setup

- Python 3.8
- [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium)
- [MAMuJoCo](https://github.com/schroederdewitt/multiagent_mujoco)
- PyTorch 1.13


## Navigation

* `logs/` - Experimental results and stored models
* `safepo/common/` - Config of experiments.
* `safepo/multi_agent/` - For the MAPPO and predictor's training, please use 'mampc.py'; while 'test_mpc.py' contains the MPC code.
* `safepo/utils/` - Utility methods.


## Examples

### Train Agents on 2*4 Ants

```bash
cd safepo/multi_agent
python mampc.py --task Safety2x4AntVelocity-v0 --seed 0
```

### Test Agents on 2*4 Ants

```bash
cd safepo/multi_agent
python test_mpc.py
```

## Citation
If you find this paper helpful, please cite it:

<pre>
@misc{wang2024deepsafempc,
      title={DeepSafeMPC: Deep Learning-Based Model Predictive Control for Safe Multi-Agent Reinforcement Learning}, 
      author={Xuefeng Wang and Henglin Pu and Hyung Jun Kim and Husheng Li},
      year={2024},
      eprint={2403.06397},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
</pre>

## License

See [LICENSE](LICENSE).

SPDX-License-Identifier: MIT
