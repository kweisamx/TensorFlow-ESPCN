# Implement ESCPN with Tensorflow

## Dependency
pip
* Tensorflow
* Opencv
* h5py


If you meet the problem with opencv when run the program
```
libSM.so.6: cannot open shared object file: No such file or directory
```

please install dependency package

```
sudo apt-get install libsm6
sudo apt-get install libxrender1
```


## How to train
```
python main.py
```

if you want to see the flag 
```
python main.py -h
```

## How to test

If you don't input a Test image, it will be default image
```
python main.py --is_train False
```
then result will put in the result directory


If you want to Test your own iamge

use `test_img` flag

```
python main.py --is_train False --test_img Train/t20.bmp
```

then result image also put in the result directory

## Result 
    
    
    
    
## References

   [kweisamx/SRCNN-Tensorflow](https://github.com/kweisamx/TensorFlow-SRCNN)
