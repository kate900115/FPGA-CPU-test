#// zyuxuan
#include <iostream>
#include <ctime>
#include <chrono>
#include <string>
#include <stdint.h>
#include <sys/types.h>
#include <inttypes.h>

//#include "cuda.h"
//#include "cuda_runtime_api.h"
#include "v2p2vioctl.h"

#include <dirent.h>
#include <signal.h>
#include <pthread.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <errno.h>
#include <sys/uio.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>


unsigned int MEM_SEG_SIZE = 16;
unsigned int AQ_ENTRY_SIZE = 16;
int SEND_BUF_SIZE = 512;
int RECV_BUF_SIZE = 512;

uint64_t interpAddr = 0x8;
uint64_t FPGA_BASE = 0xc8000000;

	

uint64_t read_user_reg( uint64_t usr_reg_addr){
	uint64_t physAddr = FPGA_BASE + (interpAddr << 20) + (usr_reg_addr<<3);
	return physAddr;
}

uint64_t get_doorbell_addr(uint64_t kid){
	uint64_t doorbellAddr = FPGA_BASE + (kid<<8) + 0x1000000;
	return doorbellAddr;
}

uint64_t get_AQ_addr (uint64_t kid){
	uint64_t aqAddr = FPGA_BASE + (kid<<8) + 0x1000000 + 2;
	return aqAddr;
}


void* getAddrWithOffset(void* addr, uint64_t usr_reg_addr){
	char* tmp = (char*) addr;
	//void* returnAddr = (void*)(tmp + (interpAddr<<20) + (usr_reg_addr<<3));
	void* returnAddr = (void*)(tmp + (usr_reg_addr<<3));

	return returnAddr;
}


int main(int argc, char *argv[])
{
	int res = -1;

	int fd = open("/dev/v2p2v", O_RDWR, 0);
	//int fd = open("/dev/"GPUMEM_DRIVER_NAME, O_RDWR, 0);
	if (fd < 0) {
		printf("Error open file %s\n", "/dev/v2p2v");
		return -1;
	}

	long kid = 0b000000;
	long function_code = 0b000;

	// get the virtual address to access to FPGA's config registers
	uint64_t FPGA_config_base_phys_addr = read_user_reg(0x0);
	cpuaddr_t* FPGA_config_user2kernel_parameter;
	FPGA_config_user2kernel_parameter = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	FPGA_config_user2kernel_parameter->paddr = FPGA_config_base_phys_addr;
	res = ioctl(fd, IOCTL_P2V, FPGA_config_user2kernel_parameter); //1

	printf("FPAG_config_base_phys_addr = 0x%ld\n", FPGA_config_base_phys_addr);

	void* FPGA_config_base_virt_addr = mmap(0, 1024*1024, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);


	if (FPGA_config_base_virt_addr == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	volatile uint64_t* FPGA_config_addr = (uint64_t*) getAddrWithOffset(FPGA_config_base_virt_addr, 0x0);
	//volatile uint64_t* FPGA_config_data = (uint64_t*) getAddrWithOffset(FPGA_config_base_virt_addr, 0x6);
	//volatile uint64_t* FPGA_config_valid_signal = (uint64_t*) getAddrWithOffset(FPGA_config_base_virt_addr, 0x1);

	printf("FPGA_config_addr = %p;\n", FPGA_config_addr);
	//printf("FPGA_config_data = %p;\n", FPGA_config_data);
	//printf("FPGA_config_valid_signal = %p;\n", FPGA_config_valid_signal);


	// get the virtual address to access to the doorbell register on FPGA
	uint64_t FPGA_doorbell_reg_phys_addr = get_doorbell_addr(3);


	cpuaddr_t* FPGA_doorbell_user2kernel_parameter;
	FPGA_doorbell_user2kernel_parameter =  (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	FPGA_doorbell_user2kernel_parameter->paddr = FPGA_doorbell_reg_phys_addr;

	res = ioctl(fd, IOCTL_P2V, FPGA_doorbell_user2kernel_parameter);//2


	void* FPGA_doorbell_reg_virt_addr = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (FPGA_doorbell_reg_virt_addr == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}


	volatile uint64_t* FPGA_doorbell_reg = (uint64_t*) getAddrWithOffset(FPGA_doorbell_reg_virt_addr, 0);


	// get the virtual address to access to AQ cursor addr on FPGA
	uint64_t FPGA_AQ_cursor_reg_phys_addr = get_AQ_addr(3);
	cpuaddr_t* FPGA_AQ_cursor_user2kernel_parameter;
	FPGA_AQ_cursor_user2kernel_parameter = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	FPGA_AQ_cursor_user2kernel_parameter->paddr = FPGA_AQ_cursor_reg_phys_addr;
	res = ioctl(fd, IOCTL_P2V, FPGA_AQ_cursor_user2kernel_parameter);//3

	void* FPGA_AQ_cursor_reg_virt_addr = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

	if (FPGA_AQ_cursor_reg_virt_addr == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	// configure send buffer address
	// generate virtual addr of CPU(GPU) memory
	// generate phys addr according the virtual addr
	void* va_sendBuf = mmap(0,512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va_sendBuf == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	printf("virtual addr: %p\n", va_sendBuf);

	cpuaddr_t *pa_sendBuf;
	pa_sendBuf = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	pa_sendBuf->paddr = 0;
	
	res = ioctl(fd, IOCTL_V2P, pa_sendBuf);
		
	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	printf("send buffer physical address = %p\n", pa_sendBuf->paddr);	
	printf("FPGA haven't written the CPU memory: %lx\n", *(int*)va_sendBuf);


	// send the physical address of send buffer to FPGA
	// by setting the config_addr, config_data, and valid signal.

	// to update send buffer address
	// function code should be 0;
	kid = 3;
	function_code = 0;

	printf("FPGA_config_addr = %p\n", FPGA_config_addr);

	for (int i=0; i<MEM_SEG_SIZE; i++){
		//*FPGA_config_addr = {thread_id, mem_seg_idx};
		//FPGA_config_addr = 
		*FPGA_config_addr = (kid<<10) + (i<<3) + function_code;
	//	*FPGA_config_data = pa_sendBuf->paddr;
	//	//*FPGA_config_valid_signal = 0;
	}


	// configure receive buffer address
	void* va_recvBuf = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va_recvBuf ==MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}
		
	printf("virtual recv buf addr: %p\n", va_recvBuf);

	cpuaddr_t* pa_recvBuf;
	pa_recvBuf = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	pa_recvBuf->paddr = 0;

	res = ioctl(fd, IOCTL_V2P, pa_recvBuf);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	printf("recv buffer physical address = %p\n", pa_recvBuf->paddr);
	printf("FPGA haven't written the CPU memory: %lx\n", *(int*)va_recvBuf);

	//set config_addr, config_data and valid signal
	// to update receive buffer
	// function code should be 1


	kid = 3;
	function_code = 1;
	for (int i=0; i<MEM_SEG_SIZE; i++){
		*FPGA_config_addr = (kid<<10) + (i<<3) + function_code;
		//*FPGA_config_data = pa_recvBuf->paddr;
		//*FPGA_config_valid_signal = 0;
	}

	

	// configure AQ address
	void* va_AQ = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va_AQ==MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}

	printf("virtual AQ addr: %p\n", va_AQ);

	cpuaddr_t* pa_AQ;
	pa_AQ = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	pa_AQ->paddr = 0;

	res = ioctl(fd, IOCTL_V2P, pa_AQ);
	
	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	printf("AQ physical address = %p\n", pa_AQ->paddr);
	printf("FPGA haven't written the CPU memory: %lx\n", *(int*)va_AQ);

	//set config_addr, config_data and valid signal

	// to update AQ address function code should be 3

	kid = 3; 
	function_code = 3;

	for (int i=0; i<AQ_ENTRY_SIZE; i++){
		*FPGA_config_addr = (kid<<10) + (i<<3) + function_code;
		//*FPGA_config_data = pa_AQ->paddr;
		//*FPGA_config_valid_signal = 0;
	}

	// configure doorbell buffer on CPU(GPU)
	
	void* va_doorbellBuf = mmap(0, 512, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
	if (va_doorbellBuf == MAP_FAILED){
		fprintf(stderr, "%s():%s\n", __FUNCTION__, strerror(errno));
	}
	
	printf("virtual doorbell buffer: %p\n", va_doorbellBuf);

	cpuaddr_t* pa_doorbellBuf;
	pa_doorbellBuf = (struct cpuaddr_t*)malloc(sizeof(struct cpuaddr_t));
	pa_doorbellBuf->paddr = 0;

	res = ioctl(fd, IOCTL_V2P, pa_doorbellBuf);

	if (res<0){
		fprintf(stderr, "Error in IOCTL_V2P\n");
		exit(-1);
	}

	printf("doorbell buf phys addr = %p\n", pa_doorbellBuf);
	printf("FPGA haven't written the CPU memory: %lx\n", *(int*)va_doorbellBuf);

	//to update the address of doorbell buffer
	// the function code should be 2

	kid = 3; 
	function_code = 2;
	*FPGA_config_addr = (kid<<10) + function_code ;
	//*FPGA_config_data = pa_doorbellBuf->paddr;
	//*FPGA_config_valid_signal = 0;




	// configure sendBuf size on CPU(GPU)
	// the function code should be 4
	kid = 3; 
	function_code = 4;
	*FPGA_config_addr = (kid<<10) + function_code ;
	//*FPGA_config_data = SEND_BUF_SIZE;
	//*FPGA_config_valid_signal = 0;

	// configure recvBuf size on CPU(GPU)
	kid = 3;
	function_code = 5;
	*FPGA_config_addr = (kid<<10) + function_code ;
	//*FPGA_config_data = RECV_BUF_SIZE;
	//*FPGA_config_valid_signal = 0;



	// GPU copy the data into recv buffer
	double* recvBuf = (double*) va_recvBuf;
	for (int i = 0; i<(RECV_BUF_SIZE/sizeof(double)); i++){
		recvBuf[i] = i*i;
	}	

	// GPU do the execution
	double* sendBuf = (double*) va_sendBuf;
	for (int i=0; i<(SEND_BUF_SIZE/sizeof(double)); i++){
		sendBuf[i] = recvBuf[i]/i;
	}
	

	// GPU send a doorbell to FPGA
	// (1) fill the doorbell buffer on GPU(CPU)
	long* doorbell_buf = (long*)va_doorbellBuf;
	kid = 3;
	int mem_seg_idx = 0;
	*doorbell_buf = (mem_seg_idx<<8) + kid;	

	// (2) send the doorbell

	*FPGA_doorbell_reg = 3;//kid


	munmap(FPGA_config_base_virt_addr, 512);
	munmap(FPGA_doorbell_reg_virt_addr, 512);
	munmap(FPGA_AQ_cursor_reg_virt_addr, 512);
	munmap(va_recvBuf, 512);
	munmap(va_sendBuf, 512);
	munmap(va_AQ, 512);
	munmap(va_doorbellBuf, 512);

	close(fd);

	return 0;
}


