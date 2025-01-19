import numpy as np
import os
import pickle 


class Sequential:
    def __init__(self):
        self.layers = []
        self.built = False
        self.loss_func = None
        self.optimizer = None
        self.metrics = None
        self.regularizers = 0


    def add(self, layer):
        self.layers.append(layer)

    def compile(self, loss, optimizer, metrics=None):
        self.loss_func = loss
        self.optimizer = optimizer
        self.metrics = metrics

    def fit(self, X, y, epoch=5, batch_size=64, validation_data=None, shuffle=True):
        for count in range(len(self.layers)):
            self.layers[count].predicting = False

        rand_id = np.array(range(0, len(X)))
        for _ in range(epoch):
            
            if shuffle:
                np.random.shuffle(rand_id)
            X_batches = np.array_split(X, np.ceil(len(X) / batch_size))
            y_batches = np.array_split(y, np.ceil(len(y) / batch_size))
            print(f"Epoch {_+1}/{epoch} - {int(np.ceil(len(X) / batch_size))} batches")

            for X_batch, y_batch in zip(X_batches, y_batches):
                self.regularizers = 0
                
                self.parameters = []
                self.grads = []


                training_metrics = self.evaluate(X_batch, y_batch)

                error = np.array(self.loss_func.backward())
                # print("LOSS ERROR: ", np.max(np.abs(error)))
                # print(self.layers[-1].W.shape)
                for count, layer in zip(range(len(self.layers)-1, -1, -1), reversed(self.layers)):
                    # print(error.shape)
                    error = np.array(self.layers[count].backward(error))
                    # print("LAYER ERROR: ", np.mean(np.abs(error)))

                    if layer.trainable:
                        config = self.layers[count].get_config()
                        self.parameters += config["parameters"]
                        self.grads += config["grads"]

                        # print(np.mean(np.abs(config["grads"][0])))

                # for i in range(len(self.grads)):
                #     if self.layers[i].trainable:
                #         print(f"GRADS: {np.mean(np.abs(self.grads[i][0]))}" )
                new_params = self.optimizer.step(self.parameters, self.grads)

                param_counter = 0            
                for count, layer in zip(range(len(self.layers)-1, -1, -1), reversed(list(self.layers))):
                    if layer.trainable:
                        self.layers[count].update_W(new_params[param_counter])
                        param_counter += 1
                        self.layers[count].update_b(new_params[param_counter])
                        param_counter += 1
                    
                # print("W and dW: ", [(np.mean(np.abs(self.layers[i].W)), np.mean(np.abs(self.layers[i].dW))) for i in range(len(self.layers)) if self.layers[i].trainable])
                # sys.exit()


            for training_metric in training_metrics:
                print(f"{training_metric.capitalize()}: {training_metrics[training_metric]}", end=" - ")
                
            if validation_data:
                X_val, y_val = validation_data[0], validation_data[1]

                validation_metrics = self.evaluate(X_val, y_val)
                for validation_metric in validation_metrics:
                    print(f"{validation_metric.capitalize()}: {validation_metrics[validation_metric]}", end=" - ")

            print()

            

        for count in range(len(self.layers)):
            self.layers[count].predicting = True
                
    
    def evaluate(self, X, y):
        y_pred = self.predict(X)
        # print(self.regularizers)
        pre_reg_loss = self.loss_func.forward(y_pred, y)
        loss =  pre_reg_loss + self.regularizers
        
        return_metrics= {}
        return_metrics["loss"] = round(pre_reg_loss, 6)
        if self.metrics:
            for metric in self.metrics:
                return_metrics[metric.__name__] = round(metric(y, y_pred), 6)
            

        return return_metrics
    
    def predict(self, X):
        data = X.copy()
        for count, layer in enumerate(self.layers):
            # print(data.shape)
            if layer.trainable and not layer.built:
                self.layers[count].build(data.shape)
            # if layer.trainable:
                # print("WEIGHT: ", self.layers[count].W.shape, end="   ")
            # print("INPUT: ", data.shape, end="  ")
            # if self.layers[count].trainable:
            #     print(data.shape, self.layers[count].W.shape)
            # else:
            #     print(data.shape)
            data = self.layers[count].forward(data)
            # print("OUTPUT: ", np.mean(np.abs(data)))
            if self.layers[count].trainable and not self.layers[count].predicting:
                config = self.layers[count].get_config()
                if "regularizers" in config:
                    self.regularizers += config["regularizers"][0]
            # print("OUTPUT: ", (data.shape))
            

        return data

    def save_model(self, name):
        if not os.path.exists("models"):
            os.makedirs("models")

        with open(f"models\\{name}.pkl", 'wb') as f:
            pickle.dump(self, f)


def load_model(name):
    with open(f"models\\{name}.pkl", 'rb') as f:
        model = pickle.load(f)
    return model


# def save_model(name, model):
#     if "models" not in os.listdir():
#         os.mkdir("models")

#     with open(f"models\\{name}.pkl", 'wb') as f:
#         pickle.dump(model, f)


# def load_model(name):
#     with open(f"models\\{name}.pkl", 'wb') as f:
#         model = pickle.load(f)

#     return model
