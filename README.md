# Implement ESCPN with Tensorflow

![Imgur](https://i.imgur.com/T1ZXLM0.png)

## Dependency
### pip
* Tensorflow
* Opencv
* h5py



## How to train
```
python main.py
```

if you want to see all flag 
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


## Subpixel CNN layer

[source](https://github.com/tetrachrome/subpixel/blob/master/README.md)

In numpy, we can write this as

```python
def PS(I, r):
  assert len(I.shape) == 3
  assert r>0
  r = int(r)
  O = np.zeros((I.shape[0]*r, I.shape[1]*r, I.shape[2]/(r*2)))
  for x in range(O.shape[0]):
    for y in range(O.shape[1]):
      for c in range(O.shape[2]):
        c += 1
        a = np.floor(x/r).astype("int")
        b = np.floor(y/r).astype("int")
        d = c*r*(y%r) + c*(x%r)
        print a, b, d
        O[x, y, c-1] = I[a, b, d]
  return O
```

To implement this in Tensorflow we would have to create a custom operator and
its equivalent gradient. But after staring for a few minutes in the image
depiction of the resulting operation we noticed how to write that using just
regular `reshape`, `split` and `concatenate` operations. To understand that
note that phase shift simply goes through different channels of the output
convolutional map and builds up neighborhoods of `r x r` pixels. And we can do the
same with a few lines of Tensorflow code as:

```python
   def _phase_shift(self, I, r):
       # Helper function with main phase shift operation
       bsize, a, b, c = I.get_shape().as_list()
       X = tf.reshape(I, (self.batch_size, a, b, r, r))
       X = tf.split(X, a, 1)  # a, [bsize, b, r, r]
       X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, b, a*r, r
       X = tf.split(X, b, 1)  # b, [bsize, a*r, r]
       X = tf.concat([tf.squeeze(x) for x in X], 2)  # bsize, a*r, b*r
       return tf.reshape(X, (self.batch_size, a*r, b*r, 1))

   def PS(self, X, r):
       # Main OP that you can arbitrarily use in you tensorflow code
       Xc = tf.split(X, 3, 3)
       if self.is_train:
           X = tf.concat([self._phase_shift(x, r) for x in Xc], 3) # Do the concat RGB
       else:
           X = tf.concat([self._phase_shift_test(x, r) for x in Xc], 3) # Do the concat RGB
       return X
```

## Result 

   * origin 255 x 255 x 3
   
   ![Imgur](https://i.imgur.com/UtX10XD.png)
    
   * upscaling 3 times, 765 x 765 x 3
   
   ![Imgur](https://i.imgur.com/oAmX8QF.png)
   
        
    
    
    
## References

   * [kweisamx/SRCNN-Tensorflow](https://github.com/kweisamx/TensorFlow-SRCNN)
   * [tetrachrome/subpixel](https://github.com/tetrachrome/subpixel)
   
## problem
If you meet the problem with opencv when run the program
```
libSM.so.6: cannot open shared object file: No such file or directory
```

please install dependency package

```
sudo apt-get install libsm6
sudo apt-get install libxrender1
```

