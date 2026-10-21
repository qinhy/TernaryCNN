
def summ(model, verbose=True, include_buffers=True):
    info = []

    for name, module in model.named_modules():
        # Parameters and buffers belonging to this module only
        params = list(module.named_parameters(recurse=False))
        buffers = (
            list(module.named_buffers(recurse=False))
            if include_buffers else []
        )

        nparams = sum(p.numel() for _, p in params)
        nbuffers = sum(b.numel() for _, b in buffers)

        def get_dtype(t):
            return str(t.dtype).replace("torch.", "")

        param_info = [
            {
                "name": n,
                "shape": tuple(p.shape),
                "numel": p.numel(),
                "dtype": get_dtype(p),
            }
            for n, p in params
        ]

        buffer_info = [
            {
                "name": n,
                "shape": tuple(b.shape),
                "numel": b.numel(),
                "dtype": get_dtype(b),
            }
            for n, b in buffers
        ]

        out_chs = getattr(module, "out_channels", None)

        row = {
            "name": name,
            "type": module.__class__.__name__,
            "nparams": nparams,
            "nbuffers": nbuffers,
            "parameters": param_info,
            "buffers": buffer_info,
            "out_channels": out_chs,
        }
        info.append(row)

        if verbose:
            extra = (
                f"  out_channels={out_chs}"
                if out_chs is not None else ""
            )

            print(
                f"{name:35} {module.__class__.__name__:25} "
                f"params={nparams:8d}  buffers={nbuffers:8d}"
                f"{extra}"
            )

            for p in param_info:
                print(
                    f"    [P] {p['name']:25} "
                    f"shape={str(p['shape']):20} "
                    f"dtype={p['dtype']:10} "
                    f"numel={p['numel']}"
                )

            for b in buffer_info:
                print(
                    f"    [B] {b['name']:25} "
                    f"shape={str(b['shape']):20} "
                    f"dtype={b['dtype']:10} "
                    f"numel={b['numel']}"
                )

    return info
