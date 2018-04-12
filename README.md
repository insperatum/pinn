# Program Induction Neural Networks
For training RobustFill-like networks (https://arxiv.org/pdf/1703.07469.pdf)

Example: 3-shot learning f:A->B from a support set X = [(a1,b1), (a2, b2), (a3, b3)]

where ai, bi, f are sequences with vocabularies of v_a, v_b, v_f

```
from pinn import RobustFill
net = RobustFill(input_vocabularies=[v_a, v_b], target_vocabulary=v_f)
batch_inputs = [X1, X2, X3, ...]
batch_target = [f1, f2, f3, ...]
score = net.optimiser_step(batch_inputs, batch_targets)
```

# Todo:
- [ ] Double check if correct: attend during P->FC rather than during softmax->P?
- [X] Output attending to input
- [X] Target attending to output
- [X] Allow both input->target and (input,output)->target modes
- [ ] Pytorch JIT
- [ ] BiLSTM
- [ ] Multiple attention and different attention functions
- [ ] Reinforce
- [ ] Beam search
- [ ] Give n_examples as input to FC