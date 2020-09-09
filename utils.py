import h5py as h5


def pack_hdf(pth, data_dict):
    """Takes data organized in a python dict, and creates an hdf5 with the
    same structure."""
    def rec(data, grp):
        for k, v in data.items():
            if type(v) is dict:
                rec(v, grp.create_group(k))
            else:
                grp.create_dataset(k, data=v)

    with h5.File(pth + ".h5", "w") as pckg:
        rec(data_dict, pckg)


def unpack_hdf(group):
    """Recursively unpack an hdf5 of nested Groups (and Datasets) to dict."""
    return {
        k: v[()] if type(v) is h5._hl.dataset.Dataset else unpack_hdf(v)
        for k, v in group.items()
    }
