from core.networks.double_nodes.DenseDoubleNode import DenseDoubleNode

DENSE_LAYER = "Dense"


class NodeConstructor():
    def _get_layer(self, layer_type, bottom_layer_shape, **args):
        if layer_type == DENSE_LAYER:
            return DenseDoubleNode(size_bottom=bottom_layer_shape, **args)
        else:
            raise Exception("No such layer implemented: {}".format(layer_type))

    def construct_layers(self, input_shape, network_list, **common_args):
        layers = []
        output_shape = input_shape
        for i in range(len(network_list)):
            print(output_shape)
            node_name, node_args = network_list[i]
            layer = self._get_layer(node_name, output_shape, **{**node_args, **common_args})
            output_shape = layer.get_output_shape()
            layers.append(layer)
        print(output_shape)
        return layers


def get_layer_id(layer_name, layer_args):
    if layer_name == DENSE_LAYER:
        return "D{}".format(layer_args["size_top"])
    else:
        return "B1"
