from model import build_model, train_model, save_model, load_the_model, is_model_saved, get_train_data

def get_model():
    if is_model_saved():
        model = build_model()
        model = load_the_model(model)
        (X_tr, Y_tr), (X_te, Y_te) = get_train_data()
        print(model.evaluate(X_te, Y_te))
        return model
    else:
        print("The model is not saved, so building it, training it and then saving it.")
        (X_tr, Y_tr), (X_te, Y_te) = get_train_data()
        model = build_model()
        model = train_model(model, X_tr, Y_tr, X_te, Y_te)
        save_model(model)
        return model

if __name__ == "__main__":
    model = get_model()
