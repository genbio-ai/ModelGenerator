
def make_them_on_same_device(*args):
    if "cpu" in [d.device.type for d in args]:
        out = [d.cpu() for d in args]
        return out
    else:
        return args