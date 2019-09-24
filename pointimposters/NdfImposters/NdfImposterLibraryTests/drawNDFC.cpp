#version 430

uniform ivec2 histogramDiscretizations;

int HistogramWidth = histogramDiscretizations.x;
int HistogramHeight = histogramDiscretizations.y;

uniform int phys_tex_dim;
uniform int tile_w;
uniform int tile_h;


layout(std430) buffer NdfVoxelData
{
	coherent float histograms[];
} ndfVoxelData;

layout(std430) buffer NDFImage
{
	coherent float pixel[];
}img;

layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;

void main()  
{
	//globalinvocationid.x  -> ndfcacheW*histogrambins.width*2
	//globalinvocationid.y  -> ndfcacheH*histogrambins.height
	//ivec2 pixel_indx = ivec2(gl_GlobalInvocationID.x/(HistogramWidth*2), gl_GlobalInvocationID.y/HistogramHeight);
	//ivec2 tile_indx = ivec2(pixel_indx/tile_w,pixel_indx/tile_h);
	//ivec2 pixelInTile = ivec2(pixel_indx % tile_w, pixel_indx % tile_h);
	//ivec2 binIndx = ivec2(gl_GlobalInvocationID.x % (HistogramWidth * 2), gl_GlobalInvocationID.y % HistogramHeight);

	////const unsigned int lookup = unsigned int((Pixelcoord_InPhysTex.y * VolumeResolutionX + Pixelcoord_InPhysTex.x) * HistogramHeight * HistogramWidth);
	//int lookup = int(((pixel_indx.y) * phys_tex_dim + pixel_idnx.x) * HistogramHeight * HistogramWidth);
	{
		//ivec2 img_ind = ivec2(gl_GlobalInvocationID.x, gl_GlobalInvocationID.y);
		//ivec2 img_pixel_indx = ivec2(gl_GlobalInvocationID.x / (HistogramWidth * 2), gl_GlobalInvocationID.y / HistogramHeight);
		//ivec2 img_bin_indx = ivec2(gl_GlobalInvocationID.x % (HistogramWidth * 2), gl_GlobalInvocationID.y % HistogramHeight);

		//ivec2 cache_ind = ivec2(gl_GlobalInvocationID.x / 2, gl_GlobalInvocationID.y);
		//ivec2 cache_pixel_indx = ivec2(gl_GlobalInvocationID.x / HistogramWidth, gl_GlobalInvocationID.y / HistogramHeight);
		//ivec2 cache_bin_indx = ivec2(gl_GlobalInvocationID.x % HistogramWidth, gl_GlobalInvocationID.y % HistogramHeight);
		//int cache_1dindx = cache_bin_indx.y*HistogramWidth + cache_bin_indx.x;

		//float s = ndfVoxelData.histograms[(cache_pixel_indx.y*(phys_tex_dim)+cache_pixel_indx.x)*HistogramWidth*HistogramHeight + cache_1dindx];
		//vec2 components = unpackHalf2x16(floatBitsToUint(s));

		//if (img_ind.x % 2 == 0)
		//{
		//	img.pixel[img_ind.y*(phys_tex_dim*HistogramWidth * 2) + img_ind.x] = components.x;
		//}
		//else
		//{
		//	img.pixel[img_ind.y*(phys_tex_dim*HistogramWidth * 2) + img_ind.x] = components.y;
		//}

		//if (cache_pixel_indx.x%tile_w == 0 || cache_pixel_indx.x%tile_w == tile_w - 1 || cache_pixel_indx.y%tile_h == 0 || cache_pixel_indx.y%tile_h == tile_h - 1)
		//	img.pixel[img_ind.y*(phys_tex_dim*HistogramWidth * 2) + img_ind.x] = 1;
	}

	//{
	//float s = ndfVoxelData.histograms[(gl_GlobalInvocationID.x)*HistogramWidth*HistogramHeight + gl_GlobalInvocationID.y];
	//vec2 components = unpackHalf2x16(floatBitsToUint(s));

	////
	//int bin1 = int(gl_GlobalInvocationID.y) * 2;
	//int bin2 = int(gl_GlobalInvocationID.y) * 2 + 1;



	//ivec2 pixel_ind = ivec2(gl_GlobalInvocationID.x%phys_tex_dim, gl_GlobalInvocationID.x / phys_tex_dim);

	//ivec2 bin1_ind = ivec2(bin1 % (HistogramWidth * 2), bin1 / (HistogramWidth * 2));
	//ivec2 s1_ind = ivec2(pixel_ind.x*(HistogramWidth * 2) + bin1_ind.x, pixel_ind.y*HistogramHeight + bin1_ind.y);

	//ivec2 bin2_ind = ivec2(bin2 % (HistogramWidth * 2), bin2 / (HistogramWidth * 2));
	//ivec2 s2_ind = ivec2(pixel_ind.x*(HistogramWidth * 2) + bin2_ind.x, pixel_ind.y*HistogramHeight + bin2_ind.y);

	//img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth * 2 + s1_ind.x] = components.x;
	//img.pixel[s2_ind.y*phys_tex_dim*HistogramWidth * 2 + s2_ind.x] = components.y;

	////if (pixel_ind.x%tile_w == 0 && bin1%(HistogramWidth*2)==0)
	////{
	////	img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth * 2 + s1_ind.x] = 100000;
	////}

	////if (pixel_ind.y%tile_h == 0 )
	////{
	////	if(bin1 / (HistogramWidth * 2) == 0)
	////	img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth * 2 + s1_ind.x] = 100000;
	////	if(bin2 / (HistogramWidth * 2) == 0)
	////		img.pixel[s2_ind.y*phys_tex_dim*HistogramWidth * 2 + s2_ind.x] = 100000;
	////}



	//}

	{
		//x: pixel coordinate in ndf cache
		//y: bin index withing pixel

		//get tile indx
		int tile_indx = int(gl_GlobalInvocationID.x)/(tile_w*tile_h);
		int pixel_indx_withinTile = int(gl_GlobalInvocationID.x) % (tile_w*tile_h);

		float s = ndfVoxelData.histograms[(gl_GlobalInvocationID.x)*HistogramWidth*HistogramHeight + gl_GlobalInvocationID.y];
		vec2 components;// = unpackHalf2x16(floatBitsToUint(s));
		components.x = s;

		//
		//int bin1 = int(gl_GlobalInvocationID.y) * 2;
		//int bin2 = int(gl_GlobalInvocationID.y) * 2 + 1;


		//get tile 2d indx in image
		int tilesPerW =  phys_tex_dim/tile_w;
		ivec2 tile2dindx = ivec2(tile_indx%tilesPerW,tile_indx/tilesPerW)*tile_w;

		ivec2 pixel_ind = ivec2(pixel_indx_withinTile%tile_w, pixel_indx_withinTile/tile_w)+tile2dindx;

		ivec2 bin1_ind = ivec2(gl_GlobalInvocationID.y % (HistogramWidth), gl_GlobalInvocationID.y / (HistogramWidth));
		ivec2 s1_ind = ivec2(pixel_ind.x*(HistogramWidth) + bin1_ind.x, pixel_ind.y*HistogramHeight + bin1_ind.y);

		//ivec2 bin2_ind = ivec2(bin2 % (HistogramWidth ), bin2 / (HistogramWidth));
		//ivec2 s2_ind = ivec2(pixel_ind.x*(HistogramWidth ) + bin2_ind.x, pixel_ind.y*HistogramHeight + bin2_ind.y);
		/*if (bin1_ind.x == (HistogramHeight - 1) || bin1_ind.y == (HistogramHeight - 1))
			img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth + s1_ind.x] = 10000;
		else*/
		//if (pixel_ind % tile_w == 0 )
		//	img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth + s1_ind.x] = 100;
		//else
			img.pixel[s1_ind.y*phys_tex_dim*HistogramWidth + s1_ind.x] = components.x;
		//img.pixel[s2_ind.y*phys_tex_dim*HistogramWidth  + s2_ind.x] = components.y;
	}
}