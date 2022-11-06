# NYU-Deep-Learning-Fall-2022

## Environment set up


Download the packages:

```
$ python3 -m venv env
$ . env/bin/activate
$ pip install -r ./requirements.txt
```

Download the data from Drive, save it to `data/` with the following layout:

```
data/
  labeled_data/
    training/
      images/
        1.JPEG
        2.JPEG
        ...
      labels/
        1.yml
        2.yml
        ...
    validation/
      images/
        ...
      labels/
        ..
    unlabled_data/
      1.PNG
      2.PNG
      ...
```
