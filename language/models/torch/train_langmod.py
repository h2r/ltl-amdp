import time
import math
import torch
import torch.nn as nn
from torch.autograd import Variable
from lang import Corpus

# check that this works with new LTL format!
def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if torch.cuda.is_available():
        data = data.cuda()
    return data


def repackage_hidden(h):
    """Wraps hidden states in new Variables, to detach them from their history."""
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def get_batch(source, i, bptt, evaluation=False):
    seq_len = min(bptt, len(source) - 1 - i)
    data = Variable(source[i:i+seq_len], volatile=evaluation)
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target


def langmod_train(data, model, batch_size, bptt, epoch, log_interval=200, lr=0.01):
    corpus = Corpus(data)

    model.train()
    train_data = batchify(corpus.train, batch_size)
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    last_total_loss = None
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output[0], targets)
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            if last_total_loss:
                if abs(last_total_loss - cur_loss) <= 0.001:
                    lr /= 4.0
                    print('Dropping learning rate from {0} to {1}'.format(lr * 4.0, lr))
                last_total_loss = cur_loss
            else:
                last_total_loss = cur_loss
            total_loss = 0
            start_time = time.time()

def langmod_train2(train_data, model, batch_size, bptt, epoch, log_interval=200, lr=0.01):
    model.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    start_time = time.time()
    hidden = model.init_hidden(batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output[0], targets)
        loss.backward()

        # torch.nn.utils.clip_grad_norm(model.parameters(), clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // bptt, lr,
                elapsed * 1000 / log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()


if __name__ == '__main__':
    langmod_train()
