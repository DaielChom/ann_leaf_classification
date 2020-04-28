import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage import io

def get_splitted_data(data_dir, split=0.7, check_id_sets=False, use_center_images=False, use_resize_images=False, verbose=0):

    # read train features
    data = pd.read_csv(data_dir+"/train.csv").sort_values("id")

    # read shape features and selected train data
    shapes = pd.read_csv(data_dir+"/shapes.csv", index_col=0).sort_values("id")
    shapes = shapes[shapes.id.isin(data.id.values)]
    shapes = shapes.reset_index(drop=True)

    # sample data to train set
    sample = lambda x:x.id.sample(frac=split).values.tolist()
    train_ids = data.groupby("species").apply(sample).values.tolist()
    train_ids = np.array(train_ids).flatten()
    train_ids = train_ids[train_ids.argsort()]

    # sample data to test set
    get_test_ids = lambda x:x not in train_ids
    test_ids = data[data.id.apply(get_test_ids)].id.sort_values().values

    if check_id_sets and verbose:
        print("The intersection between train and test set is", len(set(train_ids).intersection(test_ids)))
        
    # split features
    train_features = data[data.id.isin(train_ids)]
    test_features = data[data.id.isin(test_ids)]

    # extract target information
    num_classes = data.species.value_counts().shape[0]

    if verbose:
        print("There are {} classes for the classification task.".format(num_classes))

    species = data.species.value_counts().index.tolist()
    species = {species[i]:i for i in range(len(species))}

    set_target = lambda x:species[x]
    train_features.loc[:, "target"] = train_features.species.apply(set_target).values
    test_features.loc[:, "target"] = test_features.species.apply(set_target).values

    _ = train_features.pop("species")
    _ = train_features.pop("id")

    _ = test_features.pop("species")
    _ = test_features.pop("id")


    # split shapes
    train_shapes = shapes[shapes.id.isin(train_ids)]
    test_shapes = shapes[shapes.id.isin(test_ids)]

    train_features["image_height"] = train_shapes["image_height"]
    train_features["image_width"] = train_shapes["image_width"]

    test_features["image_height"] = test_shapes["image_height"]
    test_features["image_width"] = test_shapes["image_width"]

    X_train_fe = None
    X_train_ci = None
    X_train_ri = None
    y_train = None

    X_test_fe = None
    X_test_ci = None
    X_test_ri = None
    y_test = None

    if use_center_images:

        center_images = {i:io.imread(data_dir+"/center_resize_image/"+str(i)+".jpg")/255 for i in data.id.values}
        
        train_center_images = {i:center_images[i] for i in center_images if i in train_ids}
        X_train_ci = np.array(list(train_center_images.values()))

        test_center_images =  {i:center_images[i] for i in center_images if i in test_ids}
        X_test_ci = np.array(list(test_center_images.values()))

    
    if use_resize_images:
        
        resize_images = {i:io.imread(data_dir+"/resize_image/"+str(i)+".jpg")/255 for i in data.id.values}
        
        train_resize_images = {i:resize_images[i] for i in resize_images if i in train_ids}
        X_train_ri = np.array(list(train_resize_images.values()))

        test_resize_images =  {i:resize_images[i] for i in resize_images if i in test_ids}
        X_test_ri = np.array(list(test_resize_images.values()))

    # prepare feature arrays
    cols = [i for i in train_features.columns if "target" not in i]

    X_train_fe = train_features[cols].values
    y_train = train_features["target"].values  

    X_test_fe = test_features[cols].values
    y_test = test_features["target"].values
        

    return X_train_fe, X_train_ci, X_train_ri, y_train, X_test_fe, X_test_ci, X_test_ri, y_test, species, num_classes, train_ids, test_ids

def get_submission_data(data_dir, use_center_images=False, use_resize_images=False, verbose=0):

    # read train features
    data = pd.read_csv(data_dir+"/test.csv").sort_values("id")
    submission_ids = data.id.sort_values().values

    # read shape features and selected train data
    shapes = pd.read_csv(data_dir+"/shapes.csv", index_col=0).sort_values("id")
    shapes = shapes[shapes.id.isin(submission_ids)]
    shapes = shapes.reset_index(drop=True)
          
    # split features
    train_features = data[data.id.isin(submission_ids)]

    # split shapes
    train_shapes = shapes[shapes.id.isin(submission_ids)]
  
    train_features["image_height"] = train_shapes["image_height"]
    train_features["image_width"] = train_shapes["image_width"]

    X_fe = None
    X_ci = None
    X_ri = None
    
    if use_center_images:
        pass
    
    if use_resize_images:
        pass

    # prepare feature arrays
    cols = [i for i in train_features.columns if i not in ["target", "id"]]
    X_fe = train_features[cols].values
        
    return X_fe, X_ci, X_ri, submission_ids

def multimodal_experiment(get_model, img_X_train, fea_X_train, img_X_test, fea_X_test, y_train, y_test, num_classes, verbose=0, epochs=10, batch_size=15):

    print("Building onehot target ...", end=" ")
    y_train_oh = np.eye(num_classes)[y_train]
    y_test_oh  = np.eye(num_classes)[y_test]
    print("ok")

    print("Reshaping image ...", end=" ")
    img_X_train = img_X_train.reshape((img_X_train.shape[0], img_X_train.shape[1], img_X_train.shape[2], 1))
    img_X_test = img_X_test.reshape((img_X_test.shape[0], img_X_test.shape[1], img_X_test.shape[2], 1))
    print("ok")

    print("Training model ...", end=" ")
    model = get_model(input_dim=img_X_train.shape, extra_info_dim=fea_X_train.shape[1], num_classes=num_classes)

    if verbose:
        print(model.summary())
    
    history = model.fit([img_X_train, fea_X_train], y_train_oh, epochs=epochs, batch_size=batch_size, verbose=verbose)
    print("ok")

    print("Checking results ...")
    plt.figure(figsize=(10,3))
    plt.plot(history.history["loss"], label="loss", marker="o")
    plt.plot(history.history["accuracy"], label="accuracy", marker="o")
    plt.grid()
    plt.legend()
    plt.title("Train performance")

    preds_train = model.predict([img_X_train, fea_X_train]).argmax(axis=1)
    preds_test = model.predict([img_X_test, fea_X_test]).argmax(axis=1)

    print("train accuracy {}".format((preds_train == y_train).mean()))
    print("test accuracy  {}".format((preds_test == y_test).mean()))

