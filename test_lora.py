from main import (
    load_custom_lora,
    load_peft_lora,
    load_pretrained_model
)


def _get_trainable_parameters(model):
    """
    Returns the number of trainable parameters and all parameters
    
    Args:
        model (Model): The model to get the trainable parameters from
    
    Returns:
        int: The number of trainable parameters
        int: The total number of parameters
        float: The percentage of trainable parameters
    """
    trainable_params = sum(
        p.numel() for (_, p) in model.named_parameters() if p.requires_grad
    )
    all_params = sum(p.numel() for (_, p) in model.named_parameters())
    
    return trainable_params, all_params


# Test if the num. of trainable params is the same for both implementations
def test_load_peft_lora():
    model, _ = load_pretrained_model()
    peft_lora = load_peft_lora(model)
    trainable_params_peft, all_params_peft = _get_trainable_parameters(
        peft_lora
    )
    
    model, _ = load_pretrained_model()
    custom_lora = load_custom_lora(model)
    trainable_params_custom, all_params_custom = _get_trainable_parameters(
        custom_lora
    )
    
    assert trainable_params_peft == trainable_params_custom and \
        all_params_peft == all_params_custom