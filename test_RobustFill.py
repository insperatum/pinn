import string
import random
import time
import torch
from RobustFill import RobustFill

print("Making net...")
net = RobustFill(input_vocabularies=[string.ascii_uppercase,
                                     string.ascii_uppercase + string.ascii_lowercase],
                 target_vocabulary=string.ascii_lowercase)

if torch.cuda.is_available():
    print("CUDAfying net...")
    net.cuda()
else:
    print("Not using CUDA")

nBatch=500
nSupport=3
n_iterations = 500

def getInstance():
    target = random.sample(string.ascii_lowercase, random.randint(1,3))
    inputs =  [(x, x+target) for x in (random.sample(string.ascii_uppercase, random.randint(1,3)) for _ in range(nSupport))]
    return inputs, target

print("Training:")
start=time.time()
for i in range(n_iterations):
    instances = [getInstance() for _ in range(nBatch)]
    inputs = [_inputs for (_inputs, _target) in instances]
    targets = [_target for (_inputs, _target) in instances]
    score = net.optimiser_step(inputs, targets)
    if i%10==0: print("Iteration %d/%d" % (i, n_iterations), "Score %3.3f" % score, "(%3.3f seconds per iteration)" % ((time.time()-start)/(i+1)))

instances = [getInstance() for _ in range(5)]
inputs = [_inputs for (_inputs, _target) in instances]

for (input, program) in zip(inputs, net.sample(inputs)):
    print("Inputs:", ", ".join("".join(in1) + "->" + "".join(in2) for in1, in2 in input), "\tProgram:", "".join(program))