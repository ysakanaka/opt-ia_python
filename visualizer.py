import sys
import subprocess

sys.path.append('/Users/ysakanaka/Program/Research/coco/code-experiments/build/python')
#sys.path.append('/Users/ysakanaka/Program/Research/coco/code-experiments
# /build/python')

res = subprocess.run(["python",
                      "/Users/ysakanaka/Program/Research/coco/code"
                      "-experiments/build/python/opt_ia_python_experiment.py"
                      "", "bbob", "10"],
                     stdout=subprocess.PIPE)
sys.stdout.buffer.write(res.stdout)