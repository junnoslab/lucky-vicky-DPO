from .model import ModelLoader, Models


class Runner:
    def __init__(self):
        pass

    def run(self):
        loader = ModelLoader()
        tokenizer, model = loader.load_tokenizer_and_model(Models.BLOSSOM)

        print(tokenizer, model)
