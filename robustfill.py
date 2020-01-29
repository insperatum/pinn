from __future__ import print_function

import math

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
import torch.nn.functional as F

import copy

def choose(matrix, idxs):
    if type(idxs) is Variable: idxs = idxs.data
    assert(matrix.ndimension()==2)
    unrolled_idxs = idxs + torch.arange(0, matrix.size(0)).type_as(idxs)*matrix.size(1)
    return matrix.view(matrix.nelement())[unrolled_idxs]

class RobustFill(nn.Module):
    def __init__(self, input_vocabularies, target_vocabulary, hidden_size=512, embedding_size=128, cell_type="LSTM", max_length=25, condition_linear=False):
        """
        :param: input_vocabularies: List containing a vocabulary list for each input. E.g. if learning a function f:A->B from (a,b) pairs, input_vocabularies has length 2
        :param: target_vocabulary: Vocabulary list for output
        """
        super(RobustFill, self).__init__()
        self.n_encoders = len(input_vocabularies)

        self.t = Parameter(torch.ones(1)) #template

        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.input_vocabularies = input_vocabularies
        self.target_vocabulary = target_vocabulary
        self._refreshVocabularyIndex()
        self.v_inputs = [len(x) for x in input_vocabularies] # Number of tokens in input vocabularies
        self.v_target = len(target_vocabulary) # Number of tokens in target vocabulary

        self.no_inputs = len(self.input_vocabularies)==0
        self.max_length = max_length

        self.cell_type=cell_type
        if cell_type=='GRU':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size))
            self.encoder_cells = nn.ModuleList(
                [nn.GRUCell(input_size=self.v_inputs[0]+1, hidden_size=self.hidden_size, bias=True)] + 
                [nn.GRUCell(input_size=self.v_inputs[i]+1+self.hidden_size, hidden_size=self.hidden_size, bias=True) for i in range(1, self.n_encoders)]
            )
            self.decoder_cell = nn.GRUCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)
        if cell_type=='LSTM':
            self.encoder_init_h = Parameter(torch.rand(1, self.hidden_size)) #Also used for decoder if self.no_inputs=True
            self.encoder_init_cs = nn.ParameterList(
                [Parameter(torch.rand(1, self.hidden_size)) for i in range(len(self.v_inputs))]
            )
            self.encoder_cells = nn.ModuleList()
            for i in range(self.n_encoders):
                input_size = self.v_inputs[i] + 1 + (self.hidden_size if i>0 else 0)
                self.encoder_cells.append(nn.LSTMCell(input_size=input_size, hidden_size=self.hidden_size, bias=True))
            self.decoder_cell = nn.LSTMCell(input_size=self.v_target+1, hidden_size=self.hidden_size, bias=True)
            self.decoder_init_c = Parameter(torch.rand(1, self.hidden_size))
        
        self.W = nn.Linear(self.hidden_size if self.no_inputs else 2*self.hidden_size, self.embedding_size)
        self.V = nn.Linear(self.embedding_size, self.v_target+1)
        
        self.condition_linear = condition_linear
        if self.condition_linear:
            self.C = nn.Linear(self.hidden_size, self.v_target+1)
            self.C.weight.data.normal_(0, 0.1)
            self.C.bias.data.zero_()

        self.V.weight.data.normal_(0, 0.1)
        self.V.bias.data.zero_()
        self.V.bias.data[-1] = math.log(len(self.V.bias)-1) #Initialize so that length of output strings ~ Geo(p=0.5)

        self.As = nn.ModuleList([nn.Bilinear(self.hidden_size, self.hidden_size, 1, bias=False) for i in range(self.n_encoders)])

    def with_target_vocabulary(self, target_vocabulary):
        """
        Returns a new network which modifies this one by changing the target vocabulary
        """
        if target_vocabulary == self.target_vocabulary:
            return self

        V_weight = []
        V_bias = []
        decoder_ih = []

        for i in range(len(target_vocabulary)):
            if target_vocabulary[i] in self.target_vocabulary:
                j = self.target_vocabulary.index(target_vocabulary[i])
                V_weight.append(self.V.weight.data[j:j+1])
                V_bias.append(self.V.bias.data[j:j+1])
                decoder_ih.append(self.decoder_cell.weight_ih.data[:,j:j+1])
            else:
                V_weight.append(self._zeros(1, self.V.weight.size(1)))
                V_bias.append(self._ones(1) * -10)
                decoder_ih.append(self._zeros(self.decoder_cell.weight_ih.data.size(0), 1))

        V_weight.append(self.V.weight.data[-1:])
        V_bias.append(self.V.bias.data[-1:])
        decoder_ih.append(self.decoder_cell.weight_ih.data[:,-1:])

        self.target_vocabulary = target_vocabulary
        self.v_target = len(target_vocabulary)

        self.V.weight.data = torch.cat(V_weight, dim=0)
        self.V.bias.data = torch.cat(V_bias, dim=0)
        self.V.out_features = self.V.bias.data.size(0)

        self.decoder_cell.weight_ih.data = torch.cat(decoder_ih, dim=1)
        self.decoder_cell.input_size = self.decoder_cell.weight_ih.data.size(1)

        self._clear_optimiser()
        self._refreshVocabularyIndex()
        return copy.deepcopy(self)

    def optimiser_step(self, batch_inputs, batch_target, vocab_filter=None):
        """
        Perform a single step of SGD
        """
        if not hasattr(self, 'opt'): self._get_optimiser()
        self.opt.zero_grad()
        score = self.score(batch_inputs, batch_target, autograd=True, vocab_filter=vocab_filter).mean()
        (-score).backward()
        self.opt.step()
                
        return score.data.item()

    def score(self, batch_inputs, batch_target, autograd=True, vocab_filter=None, init_h=None, get_embeddings=False):
        inputs = self._inputsToTensors(batch_inputs)
        target = self._targetToTensor(batch_target)
        _, score, embeddings = self._run(inputs, target=target, mode="score", vocab_filter=vocab_filter, init_h=init_h)
        if not autograd: score=score.data

        if get_embeddings: return score, list(map(torch.stack, embeddings))
        else: return score

    def sample(self, batch_inputs=None, n_samples=None, vocab_filter=None, init_h=None, get_embeddings=False):
        assert batch_inputs is not None or n_samples is not None
        inputs = self._inputsToTensors(batch_inputs)
        target, score, embeddings = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter, init_h=init_h)
        target = self._tensorToOutput(target)

        if get_embeddings: return target, list(map(torch.stack, embeddings))
        else: return target

    def sampleAndScore(self, batch_inputs=None, n_samples=None, nRepeats=None, vocab_filter=None, init_h=None, autograd=True, get_embeddings=False):
        assert batch_inputs is not None or n_samples is not None
        inputs = self._inputsToTensors(batch_inputs)
        if nRepeats is None:
            target, score, embeddings = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter, init_h=init_h)
            target = self._tensorToOutput(target)
            if not autograd: score=score.data

            if get_embeddings: return target, score, list(map(torch.stack, embeddings))
            else: return (target, score)
        else:
            target = []
            score = []
            for i in range(nRepeats):
                t, s, embeddings = self._run(inputs, mode="sample", n_samples=n_samples, vocab_filter=vocab_filter, init_h=init_h)
                t = self._tensorToOutput(t)
                target.extend(t)
                if not autograd: s=s.data
                score.extend(list(s))

            if get_embeddings: return target, score, list(map(torch.stack, embeddings))
            else: return (target, score)
                                
    def _refreshVocabularyIndex(self):
        self.input_vocabularies_index = [
            {self.input_vocabularies[i][j]: j for j in range(len(self.input_vocabularies[i]))}
            for i in range(len(self.input_vocabularies))
        ]
        self.target_vocabulary_index = {self.target_vocabulary[j]: j for j in range(len(self.target_vocabulary))}
        
    def __getstate__(self):
        if hasattr(self, 'opt'):
            return dict([(k,v) for k,v in self.__dict__.items() if k is not 'opt'] + 
                        [('optstate', self.opt.state_dict())])
        else: return self.__dict__
    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if hasattr(self, 'optstate'): self._fix_optstate()

    def _ones(self, *args, **kwargs):
        return self.t.new_ones(*args, **kwargs)

    def _zeros(self, *args, **kwargs):
        return self.t.new_zeros(*args, **kwargs)

    def _clear_optimiser(self):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): del self.optstate

    def _get_optimiser(self):
        self.opt = torch.optim.Adam(self.parameters(), lr=0.001)
        if hasattr(self, 'optstate'): self.opt.load_state_dict(self.optstate)

    def _fix_optstate(self): #make sure that we don't have optstate on as tensor but params as cuda tensor, or vice versa
        is_cuda = next(self.parameters()).is_cuda
        for state in self.optstate['state'].values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda() if is_cuda else v.cpu()

    def cuda(self, *args, **kwargs):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): self._fix_optstate()
        super(RobustFill, self).cuda(*args, **kwargs)

    def cpu(self, *args, **kwargs):
        if hasattr(self, 'opt'): del self.opt
        if hasattr(self, 'optstate'): self._fix_optstate()
        super(RobustFill, self).cpu(*args, **kwargs)

    def _encoder_get_init(self, encoder_idx, h=None, batch_size=None):
        if h is None: h = self.encoder_init_h.repeat(batch_size, 1)
        if self.cell_type=="GRU": return h
        if self.cell_type=="LSTM": return (h, self.encoder_init_cs[encoder_idx].repeat(batch_size, 1))

    def _decoder_get_init(self, h=None, batch_size=None):
        if h is None:
            assert self.no_inputs
            h = self.encoder_init_h.repeat(batch_size, 1)

        if self.cell_type=="GRU": return h
        if self.cell_type=="LSTM": return (h, self.decoder_init_c.repeat(h.size(0), 1))

    def _cell_get_h(self, cell_state):
        if self.cell_type=="GRU": return cell_state
        if self.cell_type=="LSTM": return cell_state[0]

    def _run(self, inputs, target=None, mode="sample", n_samples=None, vocab_filter=None, init_h=None):
        """
        :param mode: "score" or "sample"
        :param list[list[LongTensor]] inputs: n_encoders * n_examples * (max length * batch_size)
        :param list[LongTensor] target: max length * batch_size
        :param vocab_filter: batch_size * ... (set of possible outputs)
        Returns output and score
        """
        assert((mode=="score" and target is not None) or mode=="sample")

        if vocab_filter is not None:
            vocab_mask = self.t.new([[v in V for v in self.target_vocabulary] + [True] for V in vocab_filter]).byte() #True for STOP

        if self.no_inputs:
            batch_size = target.size(1) if mode=="score" else n_samples
        else:
            batch_size = inputs[0][0].size(1)
            n_examples = len(inputs[0])
            max_length_inputs = [[inputs[i][j].size(0) for j in range(n_examples)] for i in range(self.n_encoders)]
            inputs_scatter = [
                [   Variable(self._zeros(max_length_inputs[i][j], batch_size, self.v_inputs[i]+1).scatter_(2, inputs[i][j][:, :, None], 1))
                    for j in range(n_examples)
                ] for i in range(self.n_encoders)
            ]  # n_encoders * n_examples * (max_length_input * batch_size * v_input+1)

        max_length_target = target.size(0) if target is not None else self.max_length
        score = Variable(self._zeros(batch_size))
        if target is not None: target_scatter = Variable(self._zeros(max_length_target, batch_size, self.v_target+1).scatter_(2, target[:, :, None], 1)) # max_length_target * batch_size * v_target+1

        H = [] # n_encoders * n_examples * (max_length_input * batch_size * h_encoder_size)
        embeddings = [] # n_encoders * (h for example at INPUT_EOS)
        attention_mask = [] # n_encoders * (0 until (and including) INPUT_EOS, then -inf)
        def attend(i, j, h):
            """
            'general' attention from https://arxiv.org/pdf/1508.04025.pdf
            :param i: which encoder is doing the attending (or self.n_encoders for the decoder)
            :param j: Index of example
            :param h: batch_size * hidden_size
            """
            assert(i != 0)
            scores = self.As[i-1](
                H[i-1][j].view(max_length_inputs[i-1][j] * batch_size, self.hidden_size),
                h.view(batch_size, self.hidden_size).repeat(max_length_inputs[i-1][j], 1)
            ).view(max_length_inputs[i-1][j], batch_size) + attention_mask[i-1][j]
            c = (F.softmax(scores[:, :, None], dim=0) * H[i-1][j]).sum(0)
            return c


        # -------------- Encoders -------------
        for i in range(len(self.input_vocabularies)):
            H.append([])
            embeddings.append([])
            attention_mask.append([])

            for j in range(n_examples):
                active = self._ones(max_length_inputs[i][j], batch_size).byte()
                state = self._encoder_get_init(i, batch_size=batch_size, h=embeddings[i-1][j] if i>0 else init_h)
                hs = []
                h = self._cell_get_h(state)
                for k in range(max_length_inputs[i][j]):
                    if i==0:
                        state = self.encoder_cells[i](inputs_scatter[i][j][k, :, :], state)
                    else:
                        state = self.encoder_cells[i](torch.cat([inputs_scatter[i][j][k, :, :], attend(i, j, h)], 1), state)
                    if k+1 < max_length_inputs[i][j]: active[k+1, :] = active[k, :] * (inputs[i][j][k, :] != self.v_inputs[i]).byte()
                    h = self._cell_get_h(state) 
                    hs.append(h[None, :, :])
                H[i].append(torch.cat(hs, 0))
                embedding_idx = active.sum(0).long() - 1
                embedding = H[i][j].gather(0, Variable(embedding_idx[None, :, None].repeat(1, 1, self.hidden_size)))[0]
                embeddings[i].append(embedding)
                attention_mask[i].append(Variable(active.float().log()))


        # ------------------ Decoder -----------------
        # Multi-example pooling: Figure 3, https://arxiv.org/pdf/1703.07469.pdf
        target = target if mode=="score" else self._zeros(max_length_target, batch_size).long()
        if self.no_inputs: decoder_states = [self._decoder_get_init(init_h, batch_size=batch_size)]
        else: decoder_states = [self._decoder_get_init(embeddings[self.n_encoders-1][j]) for j in range(n_examples)] #P
        active = self._ones(batch_size).byte()
        for k in range(max_length_target):
            FC = []
            for j in range(1 if self.no_inputs else n_examples):
                h = self._cell_get_h(decoder_states[j])
                p_aug = h if self.no_inputs else torch.cat([h, attend(self.n_encoders, j, h)], 1)
                FC.append(F.tanh(self.W(p_aug)[None, :, :]))
            m = torch.max(torch.cat(FC, 0), 0)[0] # batch_size * embedding_size
            v = self.V(m)
            if self.condition_linear: v = v + self.C(init_h)
            if vocab_filter is not None: v = v.masked_fill(1-vocab_mask, float('-inf'))
            logsoftmax = F.log_softmax(v, dim=1)
            if mode=="sample": target[k, :] = torch.multinomial(logsoftmax.data.exp(), 1)[:, 0]
            score = score + choose(logsoftmax, target[k, :]) * Variable(active.float())
            active *= (target[k, :] != self.v_target).byte()
            for j in range(1 if self.no_inputs else n_examples):
                if mode=="score":
                    target_char_scatter = target_scatter[k, :, :]
                elif mode=="sample":
                    target_char_scatter = Variable(self._zeros(batch_size, self.v_target+1).scatter_(1, target[k, :, None], 1))
                decoder_states[j] = self.decoder_cell(target_char_scatter, decoder_states[j]) 
        return target, score, embeddings

    def _inputsToTensors(self, inputsss):
        """
        :param inputs: size = nBatch * nExamples * nEncoders (or nBatch*nExamples is n_encoders=1)
        Returns nEncoders * nExamples tensors of size nBatch * max_len
        """
        if self.n_encoders == 0: return []
        tensors = []
        for i in range(self.n_encoders):
            tensors.append([])
            for j in range(len(inputsss[0])):
                if self.n_encoders == 1:
                    inputs = [x[j] for x in inputsss]
                else: inputs = [x[j][i] for x in inputsss]

                maxlen = max(len(s) for s in inputs)
                t = self._ones(maxlen+1, len(inputs)).long()*self.v_inputs[i]
                for k in range(len(inputs)):
                    s = inputs[k]
                    if len(s)>0: t[:len(s), k] = torch.LongTensor([self.input_vocabularies_index[i][x] for x in s])
                tensors[i].append(t)
        return tensors

    def _targetToTensor(self, targets):
        """
        :param targets: 
        """
        maxlen = max(len(s) for s in targets)
        t = self._ones(maxlen+1, len(targets)).long()*self.v_target
        for i in range(len(targets)):
            s = targets[i]
            if len(s)>0: t[:len(s), i] = torch.LongTensor([self.target_vocabulary_index[x] for x in s])
        return t

    def _tensorToOutput(self, tensor):
        """
        :param tensor:
        """
        out = []
        for i in range(tensor.size(1)):
            l = tensor[:,i].tolist()
            if l[0]==self.v_target:
                out.append(tuple())
            elif self.v_target in l:
                final = tensor[:,i].tolist().index(self.v_target)
                out.append(tuple(self.target_vocabulary[x] for x in tensor[:final, i]))
            else:
                out.append(tuple(self.target_vocabulary[x] for x in tensor[:, i]))
        return out    
