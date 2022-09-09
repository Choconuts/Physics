import gin
import gin.torch
import torch


@gin.configurable
def dnn(inputs,
        num_outputs,
        layer_sizes=(512, 512),
        activation_fn=torch.nn.ReLU):
    return layer_sizes


gin.parse_config_file('tst.gin')


with gin.config_scope('generate'):
    print(gin.current_scope())
    print(dnn(3, 3))


if __name__ == '__main__':
    pass
