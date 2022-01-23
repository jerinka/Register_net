# register_net
CNN for registering two images- calculates y shift only as of now.

[Medium page for reg_net](https://jerin-electronics.medium.com/image-registration-cnn-model-7b6114922fd9)


# Download data

```wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O cats_and_dogs_filtered.zip```

```unzip cats_and_dogs_filtered -d cat_dog```

# Train

```python train.py```

# Infer on images
```python infer.py```
