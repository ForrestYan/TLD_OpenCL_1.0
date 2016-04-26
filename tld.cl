#pragma OPENCL EXTENSION cl_amd_printf:enable
__kernel void detect_fern(__global uchar* img, 
											__global int* grid, 
											__global uchar* features, 
											__global int* ferns_global,
											__global int* col
										 )
{
	uint group_size = get_num_groups(0);
	uint local_id = get_local_id(0);
	uint group_id = get_group_id(0);
	
	
	int scale_idx;
	int x1;
	int y1;
	int x2;
	int y2;
	uchar pixel1;
	uchar pixel2;
	
	
	scale_idx = grid[group_id*3+2];
	x1 = features[scale_idx*130*4+((local_id/13)*10+(local_id%13))*4+0]+grid[group_id*3+0]; 
	y1 = features[scale_idx*130*4+((local_id/13)*10+(local_id%13))*4+1]+grid[group_id*3+1];
	x2 = features[scale_idx*130*4+((local_id/13)*10+(local_id%13))*4+2]+grid[group_id*3+0]; 
	y2 = features[scale_idx*130*4+((local_id/13)*10+(local_id%13))*4+3]+grid[group_id*3+1];
	pixel1 = img[y1*col[0]+x1];
	pixel2 = img[y2*col[0]+x2];

	
	__local int ferns[130];
	int weight = pown((float)2, (12 - (local_id%13)));
	ferns[local_id] = (pixel1>pixel2)*weight; 
	barrier(CLK_LOCAL_MEM_FENCE);

	int fern;
	if(local_id%13 == 0)
	{
		fern = ferns[local_id]+ferns[local_id+1]+ferns[local_id+2]+ferns[local_id+3]+ferns[local_id+4]+ferns[local_id+5]+
										 +ferns[local_id+6]+ferns[local_id+7]+ferns[local_id+8]+ferns[local_id+9]+ferns[local_id+10]+ferns[local_id+11]
										 +ferns[local_id+12];
		ferns_global[group_id*10+local_id/13] = fern;
	}

}


__kernel void MeasureForest(__global int* ferns,
											 __global float* posteriors,
											 __global float* conf)
{
	uint local_id = get_local_id(0);
	uint group_id = get_group_id(0);
	__local float votes[10];	
	int fern;
	float vote;
	fern = ferns[group_id*10+local_id];
	votes[local_id] = posteriors[8192*local_id+fern];
	barrier(CLK_LOCAL_MEM_FENCE);

	if(local_id == 0)
	{
		vote = votes[local_id+0]+votes[local_id+1]+votes[local_id+2]+votes[local_id+3]+votes[local_id+4]+
				   votes[local_id+5]+votes[local_id+6]+votes[local_id+7]+votes[local_id+8]+votes[local_id+9];
		conf[group_id] = vote;
	}
}



__kernel void NCC(__global float* patch,
										 __global float* pnEx,
										 __global int* pn_size,
										 __global float* global_ncc)
{	
	uint localX_id = get_local_id(0);
	uint localY_id = get_local_id(1);
	uint localZ_id = get_local_id(2);
	uint groupX_id = get_global_id(0);
	uint groupY_id = get_global_id(1);
	uint groupZ_id = get_global_id(2);



	__local float local_patch[225];
	__local float local_pnEx[225];

	__local float pnEx_product[225];
	__local float pnEx_square[225];
	__local float patch_square[225];

	__local float sum_pnEx_product;
	__local float sum_pnEx_square;
	__local float sum_patch_square;


	local_patch[localZ_id] = patch[groupX_id*225+localZ_id];	
	local_pnEx[localZ_id] = pnEx[groupY_id*225+localZ_id];

	patch_square[localZ_id] = pown(local_patch[localZ_id], 2);
	pnEx_square[localZ_id] = pown(local_pnEx[localZ_id], 2);
	//pnEx_square[localZ_id] = local_pnEx[localZ_id]*local_pnEx[localZ_id];
	pnEx_product[localZ_id] = local_pnEx[localZ_id]*local_patch[localZ_id];
	barrier(CLK_LOCAL_MEM_FENCE);
	

	if(localZ_id == 0)
	{
		sum_patch_square = 0;
		for(int i=0; i<225; i++)
		{
			sum_patch_square += patch_square[localZ_id+i];
		}
	}	
	if(localZ_id == 1)
	{
		sum_pnEx_product = pnEx_product[0];
		for(int i=0; i<224; i++)
		{
			sum_pnEx_product += pnEx_product[localZ_id+i];
		}
	}
	if(localZ_id == 2)
	{
		sum_pnEx_square = pnEx_square[0]+pnEx_square[1];
		for(int i=0; i<223; i++)
		{
			sum_pnEx_square += pnEx_square[(localZ_id+i)];
		}
	}

	barrier(CLK_LOCAL_MEM_FENCE);
	if(localZ_id == 0)
	{
		sum_pnEx_square = pow(sum_pnEx_square*sum_patch_square, 0.5);
		global_ncc[groupX_id*(pn_size[0]+pn_size[1])+groupY_id] = ((sum_pnEx_product/sum_pnEx_square+1)*0.5);
	}
}



__kernel void Conf(__global float* ncc,
									 __global int* pn_size,
									 __global float* sconf)
{
	uint global_id = get_global_id(0);

	float maxP = ncc[global_id*(pn_size[0]+pn_size[1])];
	float maxN = ncc[global_id*(pn_size[0]+pn_size[1])+pn_size[0]];
	float csmaxP = maxP;

	for(int i=0; i<pn_size[0]; i++)
	{
		if(maxP < ncc[global_id*(pn_size[0]+pn_size[1])+i])
		{
			maxP = ncc[global_id*(pn_size[0]+pn_size[1])+i];
		}
	}

	for(int i=0; i<pn_size[2]; i++)
	{
		if(csmaxP < ncc[global_id*(pn_size[0]+pn_size[1])+i])
		{
			csmaxP = ncc[global_id*(pn_size[0]+pn_size[1])+i];
		}
	}

	for(int i=pn_size[0]; i<pn_size[0]+pn_size[1]; i++)
	{
		if(maxN < ncc[global_id*(pn_size[0]+pn_size[1])+i])
		{
			maxN = ncc[global_id*(pn_size[0]+pn_size[1])+i];
		}
	}
	sconf[global_id*2] = (1-maxN)/(2-maxP-maxN);
	sconf[global_id*2+1] = (1-maxN)/(2-csmaxP-maxN);
}


