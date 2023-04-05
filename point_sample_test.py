import numpy as np
import tensorflow as tf
from point_sample import point_sample
import point_sample_test_vals as test_vals

class PointSampleTest(tf.test.TestCase):
    
    def testBasic(self):
        
        basic_res = point_sample(test_vals.basic_in, 
                            tf.tile(test_vals.basic_coords, [test_vals.basic_in.shape[0], 1, 1]), 
                            align_corners=False)
        basic_res = tf.cast(basic_res, dtype=tf.float32)
        self.assertAllClose(test_vals.basic_exp, basic_res.numpy())

if __name__ == '__main__':
    tf.test.main()