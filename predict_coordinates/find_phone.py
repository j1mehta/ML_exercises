import train_phone_finder as train
import sys


def get_model(transfer_learn = False):
    
    if(transfer_learn):
        model = train.transfer_learning_vgg('weights.h5')
    else:
        model = train.CNN('weights.h5')

    return model


if __name__ == '__main__':
    img_path = sys.argv[1]
    model = get_model()
    img = train.get_im(img_path)
    img = img.reshape(1,224,224,3)
    res = model.predict(img)
    print(res)
