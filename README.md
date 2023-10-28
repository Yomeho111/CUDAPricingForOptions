For this project, we use two methods to run the knockout options pricing with Black–Scholes model using local volatility, CPU 
multithreading and CUDA parallel computation. At the start, to not install other module to deal with File I/O, we transform the 
Excel file to three csv files. It is more easy for C++ to process. 

For the simulation, we assume that the spot prices for both equities are 100 (so the options are at the money currently). We set the risk-free 
rate at 3% and run 20000 times simulation. For CUDA setting, we set 512 threads for each block (to take advantage of warp scheduling). 
However, since I cannot update the CUDA version on the university computing cluster, I can only use CUDA 11.7 with gcc/11.2.0 
(supporting C++17) for this project. For the final value of portfolio, we just assume that there is only one share for each option 
and add them up. I also print out the breakdown value for each option in the table.

We find that for GPU method, the value of the portfolio is 4398.34, while for CPU method, the value of the portfolio is 4476.78. 
Since there are 500 options, those values are accurate.

As for running time, for CPU version, it takes 4150 milliseconds, while for GPU, it takes 150.50 milliseconds. The running time 
decreased by 96.37%. For my device, as below:
    CPU: 24 Intel(R) Xeon(R) Gold 6126 CPU @ 2.60GHz, flop: 31.95 GFLOPS
    GPU: 2 Tesla V100 16G memory, flop: 31,380 GFLOPS

We can find that through CUDA simulation, the running time can be sharply decreased. However, since we don't use a much larger simulation
times due to the limitation of cpu, the time decrement can be larger. We can find that a much larger simulation times can also be done in reasonable 
time in CUDA module due to the high GFLOPS.


Some points, I want to point out: 
1. As for CPU multithreading, since I haven't used threadpool, and I put every simulation to a thread 
directly, if you set a very large simulation number, the program may be crashed. (To deal with this problem, we need to use thread pool,
or limit the total threads we could create).

2. As for CUDA, To deal with transfering data from device and host easily, I define all the array as static array. However, to make program more scalable, we should 
use malloc or cudaMalloc to define them dynamically.

3. I did some memory optimization here. I enclose all the data into struct, so we only need to transfer struct between host and device.
Decreased the I/O time between device and host.

4. For now, the bottleneck is that we store our simulation path in global memory, which is slow for memory access. To further improve this,
we can combine simulate func and computePortValue func, so we don't need to care about the data access of two separate function. So we can
store simulation path in heap instead of global.
    

