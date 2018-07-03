from __future__ import print_function
import random
import time
import torch
from robustfill import RobustFill

modes = ['single', 'double']

vocab="ABCDEF"

for mode in modes:
    print("-"*20, "\nmode:%s"%mode)
    print("Making net...")
    net = RobustFill(
            input_vocabularies=
                [vocab] if mode=="single" else
                [vocab, vocab],
            target_vocabulary=vocab)
    
    if torch.cuda.is_available():
        print("CUDAfying net...")
        net.cuda()
    else:
        print("Not using CUDA")

    nBatch=50
    nSupport=2
    max_n_iterations=1000

    def getInstance():
        target = random.sample(vocab, random.randint(1,2))
        if mode=="single":
            inputs = [target * random.randint(1,2) for _ in range(nSupport)]
        else:
            inputs =  [(x, x+target) for x in (random.sample(vocab, random.randint(1,2)) for _ in range(nSupport))]
        return inputs, target

    def makePredictions(vocab_filter=None):
        instances = [getInstance() for _ in range(5)]
        if vocab_filter is not None: vocab_filter = [vocab_filter]*5
        inputs = [_inputs for (_inputs, _target) in instances]
        for (i, (input, program)) in enumerate(zip(inputs, net.sample(inputs, vocab_filter=vocab_filter))):
            if mode=="single":
                print("Inputs:", ", ".join("".join(inp) for inp in input), "\tProgram:", "".join(program))
            else:
                print("Inputs:", ", ".join("".join(in1) + "->" + "".join(in2) for in1, in2 in input), "\tProgram:", "".join(program))
            if vocab_filter is not None: assert(all(x in vocab_filter[i] for x in program))
        print()

    print("Training:")
    start=time.time()
    for i in range(max_n_iterations):
        instances = [getInstance() for _ in range(nBatch)]
        inputs = [_inputs for (_inputs, _target) in instances]
        targets = [_target for (_inputs, _target) in instances]
        score = net.optimiser_step(inputs, targets)
        if i%10==0: print("Iteration %d/%d" % (i, max_n_iterations), "Score %3.3f" % score, "(%3.3f seconds per iteration)" % ((time.time()-start)/(i+1)))
        if score>-0.2: break

    print("Predictions on " + vocab + ":")
    makePredictions()
    print("Predictions on ABC:")
    makePredictions(set("ABC"))
