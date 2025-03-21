
tileSize = (1024, 1024)
anchorParams = {
    'sizes':  [32, 64, 128, 256, 512],
    #'sizes':  [16, 32, 64, 128, 256],
    'strides': [8, 16, 32, 64, 128],
    #'sizes': [8, 16, 32, 64, 128],  # [8, 16, 32, 64, 128],
    #'strides':  [4, 8, 16, 32, 64], # [8, 16, 32, 64, 128],
    'ratios': [0.5, 1., 2], #[0.5, 1, 1.5],  # can be reduced
    'scales': [1., 1.2, 1.5],
}
ssf = 1
batchSize = 3

# AnchorParameters.default = AnchorParameters(
#     sizes   = [32, 64, 128, 256, 512],
#     strides = [8, 16, 32, 64, 128],
#     ratios  = np.array([0.5, 1, 2], keras.backend.floatx()),
#     scales  = np.array([2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)], keras.backend.floatx()),
# )