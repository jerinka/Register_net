# register_net
CNN for registering two images- calculates y shift only as of now.

# Download data

```wget --no-check-certificate https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip -O cats_and_dogs_filtered.zip```

```unzip cats_and_dogs_filtered -d cat_dog```

# Train

```python train.py```

# Infer on images
```python infer.py```