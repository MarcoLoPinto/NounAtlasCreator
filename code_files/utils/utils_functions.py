def load_dotenv(path):
    """loads .env variables

    Args:
        path (str): The path of the .env variable

    Returns:
        dict: a key-value dictionary
    """
    e_dict = {}
    with open(path) as f:
        for line in f:
            v = line.strip().split('=')
            e_dict[v[0].strip()] = v[1].strip()
    return e_dict

def print_summary(model, short = False):
    """prints the summary for a model

    Args:
        model (any): The torch model
        short (bool, optional): If the print must be synthetic. Defaults to False.
    """
    if not short:
        print(model)
        print('----------------------')
    p = sum(p.numel() for p in model.parameters())
    tp = sum(p.numel() for p in model.parameters() if p.requires_grad)
    ntp = p - tp
    print('parameters:', f'{p:,}')
    print('trainable parameters:', f'{tp:,}')
    print('non-trainable parameters:', f'{ntp:,}')