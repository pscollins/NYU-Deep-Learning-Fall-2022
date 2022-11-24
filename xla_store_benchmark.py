import timeit

import torch
import torch_xla
import torch_xla.core.xla_model as xm
import torch_xla.debug.metrics as met


DEST_SHAPE = (100000,)
THRESHOLD = .5

ITERS = 1000

def test_store_getitem(device):
    dest = torch.rand(DEST_SHAPE, device=device)
    mask = torch.rand(DEST_SHAPE, device=device) > THRESHOLD

    dest[mask] = 1.0
    if device != 'cpu':
        xm.mark_step()

def test_store_where(device):
    dest = torch.rand(DEST_SHAPE, device=device)
    mask = torch.rand(DEST_SHAPE, device=device) > THRESHOLD

    dest = torch.where(mask, 1.0, dest)
    if device != 'cpu':
        xm.mark_step()


def main():
    xla_device = xm.xla_device()
    cpu_device = 'cpu'

    scope = dict(globals(), **locals())
    print('CPU: getitem')
    print(timeit.timeit('test_store_getitem(cpu_device)', number=ITERS, globals=scope))
    print('CPU: where')
    print(timeit.timeit('test_store_where(cpu_device)', number=ITERS, globals=scope))

    print('XLA/CPU: getitem')
    print(timeit.timeit('test_store_getitem(xla_device)', number=ITERS, globals=scope))
    print(met.metrics_report())
    print('XLA/CPU: where')
    print(timeit.timeit('test_store_where(xla_device)', number=ITERS, globals=scope))
    print(met.metrics_report())



if __name__ == '__main__':
    main()
