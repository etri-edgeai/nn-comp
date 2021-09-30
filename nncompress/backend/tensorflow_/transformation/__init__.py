from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import json

from tensorflow import keras

def parse(model, parser_class, name=None, **kwargs):
    parsers = {}
    if name is not None:
        parser = parser_class(model, basestr=name+"_", **kwargs)
    else:
        parser = parser_class(model, **kwargs)
    parser.parse()
    if name is None:
        parsers["root"] = parser
    else:
        parsers[name] = parser

    for layer in model.layers:
        if layer.__class__.__name__ == "Functional":
            parsers_ = parse(layer, parser_class, name=layer.name, **kwargs)
            for name_, parser_ in parsers_.items():
                parsers[name_] = parser_

    return parsers

def inject(parsers, name=None, avoid=None, with_splits=False):

    if name is None:
        parser = parsers["root"]
    else:
        parser = parsers[name]

    model = parser._model
    imodel, igate_mapping = parser.inject(avoid=avoid, with_mapping=True, with_splits=with_splits)
    imodel_dict = json.loads(imodel.to_json())
    weights = {layer.name:layer.get_weights() for layer in imodel.layers}

    for idx, layer in enumerate(model.layers):
        if layer.name in parsers:
            isub_model, isub_gate_mapping = inject(parsers, layer.name, avoid=avoid, with_splits=with_splits)
            isub_model_dict = json.loads(isub_model.to_json())

            imodel_dict["config"]["layers"][idx]["config"]["layers"] = isub_model_dict["config"]["layers"]
            weights[layer.name] = isub_model.get_weights()

            igate_mapping.update(isub_gate_mapping)

    model_json = json.dumps(imodel_dict)
    custom_objects = {parser._gate_class.__name__:parser._gate_class}
    custom_objects.update(parser._custom_objects)
    ret = keras.models.model_from_json(model_json, custom_objects=custom_objects)
    for layer in model.layers:
        ret.get_layer(layer.name).set_weights(weights[layer.name])
    return ret, igate_mapping

def cut(parsers, gmodel, name=None):

    if name is None:
        parser = parsers["root"]
    else:
        parser = parsers[name]

    icmodel = parser.cut(gmodel)
    icmodel_dict = json.loads(icmodel.to_json()) 
    weights = {layer.name:layer.get_weights() for layer in icmodel.layers}
    for idx, layer in enumerate(icmodel.layers):
        if layer.name in parsers:
            gmodel_layer = gmodel.get_layer(layer.name)
            cmodel = cut(parsers, gmodel_layer, name=layer.name)
            
            cmodel_dict = json.loads(cmodel.to_json())
            icmodel_dict["config"]["layers"][idx]["config"]["layers"] = cmodel_dict["config"]["layers"]
            weights[layer.name] = cmodel.get_weights()
            
    model_json = json.dumps(icmodel_dict)
    ret = keras.models.model_from_json(model_json, custom_objects=parser._custom_objects)
    for layer in icmodel.layers:
        ret.get_layer(layer.name).set_weights(weights[layer.name])
    return ret
