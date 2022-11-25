

def on_device(tmp_device):
    original_device = 'xla'
    def move(el, device):
        try:
            return el.to(device)
        except AttributeError(e):
            print(f'Failed to move {el} to {device}: {e}, skipping')
            return el

    def call_on_device(f, device, *args, **kwargs):
        args = [
            move(arg, device) for arg in args
        ]
        kwargs = {
            k: move(v, device) for k, v in kwargs.items()
        }
        return f(*args, **kwargs)

    def decorator(fn):
        if tmp_device is None:
            return fn

        def wrapped(*args, **kwargs):
            result = call_on_device(f, tmp_device, *args, **kwargs)

            _ = call_on_device(lambda *args, **kwargs: None,
                               original_device, *args, **kwargs)
            return result
        return wrapped



class MyModule(torch.nn.Module):
    def __init__(self, *args):
        pass # initialization goes here


    @on_device('cpu')
    def forward(self, x):
        # this should execute on the CPU, but any surrounding code should
        # execute via XLA
        return self.inner_module(x)


        # return move_all(

    # def move_all(to_device, *args, **kwargs):
    #     def try_move(el):
    #         return el.to(to_device)
