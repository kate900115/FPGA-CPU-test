# modules:
	v2p2v				module for physical and virtual address conversion
	gpumem				module to expose gpu physical memory

# testcases:
	app				the original app that exposes the GPU memory address
	app_gpu_fpga			the full-version of GPU FPGA communication 
	app_gpumem_test			the version for CPU and GPU communication
	app_reset_counter_writeGPU	the version test the correctness of Ran's FPGA code 
	app_v2p2v_test			the version that tests the v2p2v module
	app_GPUwritesFPGA		the version that only allows GPU to write/read FPGA memory (not allow FPGA to write/read GPU)

Before you run testcases, you must first load the module into your kernel
The steps are as follow:
	* sudo su
	* cd /home/zyuxuan/gpudma/gpumem
	* sh drvload.sh
	* cd /home/zyuxuan/gpudma/v2p2v
	* sh drvload.sh
	* cd /home/zyuxuan/gpudma/app_* (the testcase that you want to use)
