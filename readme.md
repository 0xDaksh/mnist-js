# Build the model

> ipython generate_model.py

- saves the model in save directory

# Convert the model to tensorflowjs

```
tensorflowjs_converter --input_format keras \
                       save/model.h5 \
                       save/
```

# Try it using

```
	python serve.py
```