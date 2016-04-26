#include"gpu.h"



void GpuOpenCl::GetPlatformIDs()
{
	cl_int err;
	err = clGetPlatformIDs(1, &platform, NULL);
	if(err < 0)
	{
		cerr<<"Get platform IDs error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::GetDeviceIDs()
{
	cl_int err;
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	if(err < 0)
	{
		cerr<<"Get Device IDs error"<<endl;
		exit(-1);
	}
}



void GpuOpenCl::CreateContext()
{
	cl_int err;
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0)
	{
		cerr<<"Create Context error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::CreateCommandQueue()
{
	cl_int err;
	cl_queue_properties properties = CL_QUEUE_PROFILING_ENABLE;
	queue = clCreateCommandQueue(context, device, properties, &err);
	if(err < 0)
	{
		cerr<<"Create Command Queue error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::CreateProgram(const char* kernelfile)
{
	cl_int err;
	size_t buffer_size;
	char* program_buffer;
	ifstream infile(kernelfile, ios::binary);
	infile.seekg(0, ios::end);
	buffer_size = infile.tellg();
	infile.seekg(0, ios::beg);
	program_buffer = new char[buffer_size+1];
	program_buffer[buffer_size] = '\0';
	
	infile.read(program_buffer, buffer_size);
	infile.close();

	program = clCreateProgramWithSource(context, 1, (const char**)&program_buffer, &buffer_size, &err);
	delete[] program_buffer;
	if(err < 0)
	{
		cerr<<"Create Program error"<<endl;
		exit(-1);
	}

}


void GpuOpenCl::BuildProgram()
{
	cl_int err;
	char* program_log;
	size_t log_size;

	err = clBuildProgram(program, 1, &device, NULL, NULL, NULL);
	if(err < 0)
	{
		cerr<<"Build Program error"<<endl;
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);
		program_log = new char[log_size];
		program_log[log_size] = '\0';
		clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size+1, program_log, NULL);
		cout<<program_log<<endl;
		delete[] program_log;
		exit(-1);
	}
}


void GpuOpenCl::CreateKernelFern(const char* kernelname)
{
	cl_int err;
	kernelFE = clCreateKernel(program, kernelname, &err);
	if(err < 0)
	{
		cerr<<"Create Kernel error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::CreateMemBufferFern(unsigned char* img_buffer, int img_size, 
																 int* grid_buffer, int grid_size,
																 unsigned char* features_buffer, int features_size,
																int* col_buffer
																 )
{
	cl_int err;
	img_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*img_size, img_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error"<<endl;
		exit(-1);
	}

	grid_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*grid_size, grid_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error"<<endl;
		exit(-1);
	}

	fern_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*10*grid_size, NULL, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error"<<endl;
		exit(-1);
	}

	features_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(unsigned char)*features_size, features_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error"<<endl;
		exit(-1);
	}

	col_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int), col_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error"<<endl;
		exit(-1);
	}

}


void GpuOpenCl::ReleaseMemObjectFern()
{
	clReleaseMemObject(img_buf);
	clReleaseMemObject(features_buf);
	clReleaseMemObject(col_buf);
	clReleaseMemObject(grid_buf);
	clReleaseMemObject(fern_buf);
}


void GpuOpenCl::SetKernelArgFern()
{
	cl_int err;
	err = clSetKernelArg(kernelFE, 0, sizeof(cl_mem), &img_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelFE, 1, sizeof(cl_mem), &grid_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelFE, 2, sizeof(cl_mem), &features_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}

	err = clSetKernelArg(kernelFE, 3, sizeof(cl_mem), &fern_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}

	err = clSetKernelArg(kernelFE, 4, sizeof(cl_mem), &col_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}

}


void GpuOpenCl::EnqueueTaskFern(int grid_size)
{
	cl_int err;
	cl_event event;
	cl_int ret;
	size_t global_size[1] = {grid_size*13*10};
	size_t local_size[1] = {13*10};
	err = clEnqueueNDRangeKernel(queue, kernelFE, 1, NULL, global_size, local_size, 0, NULL, &event);
	if(err < 0)
	{
		perror("Enqueue ND Range Kernel error\n");
		exit(CL_FALSE);
	}
	clFinish(queue);
/*
	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("FE run time = %f ms\n", run_time_ms);
*/
}


void GpuOpenCl::ReadBufferFern(int* fern_buffer, int fern_size)
{
	cl_uint ret;
	cl_event event;
	ret = clEnqueueReadBuffer(queue, fern_buf, CL_TRUE, 0, sizeof(int)*fern_size, fern_buffer, 0, NULL, &event);
	if(ret != CL_SUCCESS)
	{
		cerr<<"Read Queue Buffer failed"<<endl;
		exit(-1);
	}

/*
	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("FE run time = %f ms\n", run_time_ms);
*/
}



void GpuOpenCl::CreateKernelMF(const char* kernelname)
{
	cl_int err;
	kernelMF = clCreateKernel(program, kernelname, &err);
	if(err < 0)
	{
		cerr<<"Create Kernel error"<<endl;
		exit(-1);
	}
}
 
void GpuOpenCl::CreateMemBufferMF( int* fern_buffer, int fern_size,
																		float* posteriors_buffer, int posteriors_size,
																		float* conf_buffer, int conf_size
																 )
{
	cl_int err;
	posteriors_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*posteriors_size, posteriors_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error2"<<endl;
		exit(-1);
	}

	fern_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*fern_size, fern_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error1"<<endl;
		exit(-1);
	}

	conf_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*conf_size, NULL, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error3"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::SetKernelArgMF()
{
	cl_int err;
	err = clSetKernelArg(kernelMF, 0, sizeof(cl_mem), &fern_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelMF, 1, sizeof(cl_mem), &posteriors_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelMF, 2, sizeof(cl_mem), &conf_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}

}


void GpuOpenCl::EnqueueTaskMF(int grid_size)
{
	cl_int err;
	cl_int ret;
	size_t global_size[1] = {grid_size*10};
	size_t local_size[1] = {10};
	cl_event event;
	err = clEnqueueNDRangeKernel(queue, kernelMF, 1, NULL, global_size, local_size, 0, NULL, &event);
	if(err < 0)
	{
		perror("Enqueue ND Range Kernel error\n");
		exit(CL_FALSE);
	}
	clFinish(queue);
/*
	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("MF run time = %f ms\n", run_time_ms);
*/



}



void GpuOpenCl::ReadBufferMF(float* conf_buffer, int conf_size)
{
	cl_uint ret;
	cl_event event;
	ret = clEnqueueReadBuffer(queue, conf_buf, CL_TRUE, 0, sizeof(float)*conf_size, conf_buffer, 0, NULL, &event);
	if(ret != CL_SUCCESS)
	{
		cerr<<"Read Queue Buffer failed"<<endl;
		exit(-1);
	}
/*
	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("MF read time = %f ms\n", run_time_ms);

*/
}


void GpuOpenCl::ReleaseMemObjectMF()
{
	clReleaseMemObject(fern_buf);
	clReleaseMemObject(posteriors_buf);
	clReleaseMemObject(conf_buf);
}


void GpuOpenCl::CreateKernelNCC(const char* kernelname)
{
	cl_int err;
	kernelNCC = clCreateKernel(program, kernelname, &err);
	if(err < 0)
	{
		cerr<<"Create Kernel error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::CreateMemBufferNCC(float* patch_buffer, int patch_size,
																	float* pnEx_buffer, int pnEx_size,
																	int* pn_size_buffer,
																  float ncc_size
																 )
{
	cl_int err;
	patch_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*patch_size, patch_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error2"<<endl;
		exit(-1);
	}

	pnEx_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*pnEx_size, pnEx_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error1"<<endl;
		exit(-1);
	}

	pn_size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*2, pn_size_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error1"<<endl;
		exit(-1);
	}

	ncc_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*ncc_size, NULL, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error3"<<endl;
		exit(-1);
	}
}



void GpuOpenCl::SetKernelArgNCC()
{
	cl_int err;
	err = clSetKernelArg(kernelNCC, 0, sizeof(cl_mem), &patch_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelNCC, 1, sizeof(cl_mem), &pnEx_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelNCC, 2, sizeof(cl_mem), &pn_size_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelNCC, 3, sizeof(cl_mem), &ncc_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}

}



void GpuOpenCl::EnqueueTaskNCC(int dim_0, int dim_1)
{
	cl_int err;
	cl_int ret;
	size_t global_size[3] = {dim_0, dim_1, 225};
	size_t local_size[3] = {1, 1, 225};
	cl_event event;
	err = clEnqueueNDRangeKernel(queue, kernelNCC, 3, NULL, global_size, local_size, 0, NULL, &event);
	if(err < 0)
	{
		perror("Enqueue ND Range Kernel error\n");
		exit(CL_FALSE);
	}
	clFinish(queue);
/*
	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("NCC run time = %f ms\n", run_time_ms);
*/
}


void GpuOpenCl::ReadBufferNCC(float* ncc_buffer, int ncc_size)
{
	cl_uint ret;
	cl_event event;
	ret = clEnqueueReadBuffer(queue, ncc_buf, CL_TRUE, 0, sizeof(float)*ncc_size, ncc_buffer, 0, NULL, &event);
	if(ret != CL_SUCCESS)
	{
		cerr<<"Read Queue Buffer failed"<<endl;
		exit(-1);
	}

/*	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("NCC read time = %f ms\n", run_time_ms);

*/
}



void GpuOpenCl::ReleaseMemObjectNCC()
{
	clReleaseMemObject(patch_buf);
	clReleaseMemObject(pnEx_buf);
	clReleaseMemObject(pn_size_buf);
	clReleaseMemObject(ncc_buf);
}


/*
void GpuOpenCl::CreateKernelCO(const char* kernelname)
{
	cl_int err;
	kernelCO = clCreateKernel(program, kernelname, &err);
	if(err < 0)
	{
		cerr<<"Create Kernel error"<<endl;
		exit(-1);
	}
}


void GpuOpenCl::CreateMemBufferCO(float* ncc_buffer, int ncc_size,
																	int* pn_size_buffer,
																  float conf_size)
{
	cl_int err;
	ncc_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(float)*ncc_size, ncc_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error2"<<endl;
		exit(-1);
	}

	pn_size_buf = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, sizeof(int)*3, pn_size_buffer, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error1"<<endl;
		exit(-1);
	}

	sconf_buf = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*conf_size, NULL, &err);
	if(err < 0)
	{
		cerr<<"Create Buffer error3"<<endl;
		exit(-1);
	}
}



void GpuOpenCl::SetKernelArgCO()
{
	cl_int err;
	err = clSetKernelArg(kernelCO, 0, sizeof(cl_mem), &ncc_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelCO, 1, sizeof(cl_mem), &pn_size_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
	err = clSetKernelArg(kernelCO, 2, sizeof(cl_mem), &sconf_buf);
	if(err < 0)
	{
		cerr<<"Set Kernel Arg error"<<endl;
		exit(-1);
	}
}



void GpuOpenCl::EnqueueTaskCO(int dim_0)
{
	cl_int err;
	cl_int ret;
	size_t global_size[1] = {dim_0};
	cl_event event;
	err = clEnqueueNDRangeKernel(queue, kernelCO, 3, NULL, global_size, NULL, 0, NULL, &event);
	if(err < 0)
	{
		perror("Enqueue ND Range Kernel error\n");
		exit(CL_FALSE);
	}
	clFinish(queue);

	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("CO run time = %f ms\n", run_time_ms);

}


void GpuOpenCl::ReadBufferCO(float* sconf_buffer, int sconf_size)
{
	cl_uint ret;
	cl_event event;
	ret = clEnqueueReadBuffer(queue, sconf_buf, CL_TRUE, 0, sizeof(float)*sconf_size, sconf_buffer, 0, NULL, &event);
	if(ret != CL_SUCCESS)
	{
		cerr<<"Read Queue Buffer failed"<<endl;
		exit(-1);
	}

	cl_ulong start_time = 0;
	cl_ulong finish_time = 0;
	cl_ulong run_time = 0;
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start_time), &start_time, NULL);
	ret = clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(finish_time), &finish_time, NULL);
	run_time = finish_time - start_time;
	float run_time_ms = (float)run_time/1000000;
	printf("CO read time = %f ms\n", run_time_ms);


}


void GpuOpenCl::ReleaseMemObjectCO()
{
	clReleaseMemObject(ncc_buf);
	clReleaseMemObject(pn_size_buf);
	clReleaseMemObject(sconf_buf);
}

*/
