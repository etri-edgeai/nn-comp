
def get_model_handler(model_name):

    if model_name == "efnet":
        from models import efficientnet as model_handler
    elif model_name == "efnet1":
        from models import efficientnet1_keras as model_handler
    elif model_name == "efnet2":
        from models import efficientnet2 as model_handler
        #from models import efficientnet2_keras as model_handler
    elif model_name == "vit":
        from models import vit as model_handler
    elif model_name == "densenet":
        from models import densenet as model_handler
    elif model_name == "densenet121":
        from models import densenet121 as model_handler
    elif model_name == "resnet50":
        from models import resnet50 as model_handler
    elif model_name == "randwired":
        from models import randwired as model_handler
    elif model_name == "mobilenetv2":
        from models import mobilenetv2 as model_handler
    elif model_name == "mobilenet":
        from models import mobilenetv1 as model_handler
    elif model_name == "nasnetmobile":
        from models import nasnetmobile as model_handler
    elif model_name == "nasnetlarge":
        from models import nasnetlarge as model_handler
    elif model_name == "mobilenetv3large":
        from models import mobilenetv3large as model_handler
    elif model_name == "mobilenetv3small":
        from models import mobilenetv3small as model_handler
    else:
        raise NotImplementedError()

    return model_handler
