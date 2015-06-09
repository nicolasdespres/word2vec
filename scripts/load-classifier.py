def load_classifier(pathname, nwords, ndim):
    with open(pathname, "rb") as stream:
        inner_nodes = np.fromfile(stream, dtype=np.int64, count=(2*nwords-2))
        leaves = np.fromfile(stream, dtype=np.float32, count=nwords*ndim)
        leaves = leaves.reshape((nwords, ndim))
        return (inner_nodes, leaves)
