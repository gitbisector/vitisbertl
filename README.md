# vitisbertl
experiments for low latency BERT large inference on Alveo

vitis_hls C++ code module called "feeder" is a matrix multiplication kernel with 1024 parallel DSPs.

It implements (nmat*1024, 1024) . (1024, vec)
  when nmat in [1,8] and vec in [1,128]
  
