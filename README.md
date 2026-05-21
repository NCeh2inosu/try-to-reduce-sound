# try-to-reduce-sound
This project uses two different microphones to record sound simultaneously. 
Due to hardware differences, there is a delay between the two microphones. This project uses four methods (Cross-Correlation, Least Squares Method, Average Magnitude Difference Function, and Ratio AMDF) to find the points of displacement between the two sound waves and then align them. 
By inverting one of the sound waves, the project understands the ratio at which the two sound waves can cancel each other out and visualizes this result.

．Cross-Correlation
  Used to measure the similarity of two signals at different time delays.
  
．Least Squares Method
  The optimal matching parameters or fitted curve are found by minimizing the sum of squared errors.
  
．AMDF (Average Magnitude Difference Function)
  After aligning the signals, subtract the values ​​point by point, take the absolute values, and sum them. When the delay time is exactly equal to the signal period, the corresponding AMDF value will reach a local minimum (valley).
  
．Ratio AMDF
  AMDF is used to calculate the ratio of minimum to maximum values, or the ratio between local valleys and peaks. This is suitable for real-world environments with limited hardware performance, asymmetrical microphone gain, dynamic volume shifts over time, or low-frequency DC bias.
