#include "hog_features.h"
#include "config.h"
#include <math.h>

hog_features_generator::hog_features_generator() : HOG_WTSCALE(2), HOG_NORM_EPS(1.0f), HOG_NORM_EPS2(0.01f), HOG_NORM_MAXVAL(0.2f), 
			NUM_HOG_BINS(config::get().N_HOG_DIMS), 
			bandwidth(config::get().CELL_DIMS.x/config::get().N_HOG_DIMS.x,config::get().CELL_DIMS.y/config::get().N_HOG_DIMS.y), 
			HOG_CELL_DIMS(config::get().CELL_DIMS), NUM_ANGLES(config::get().NUM_ANGLES) {
	float_point var2;
	for (int i = 0; i < 2; i++){
		var2[i] = HOG_CELL_DIMS[i] / (2*HOG_WTSCALE);
		var2[i] = var2[i]*var2[i]*2;	  
	}
	float_point half_bin(NUM_HOG_BINS.x/2.0f, NUM_HOG_BINS.y/2.0f);
	float_point cenBand(HOG_CELL_DIMS.x/2.0f, HOG_CELL_DIMS.y/2.0f);

	_w.resize(boost::extents[HOG_CELL_DIMS.x][HOG_CELL_DIMS.y]);
	_pt.resize(boost::extents[HOG_CELL_DIMS.x][HOG_CELL_DIMS.y]);
	for (int x = 0; x < HOG_CELL_DIMS.x; x++){
		for (int y = 0; y < HOG_CELL_DIMS.y; y++)
		{
			float xx = (x - 0.5f*HOG_CELL_DIMS.x);
			float yy = (y - 0.5f*HOG_CELL_DIMS.y);
			_w[x][y] = expf( -(xx*xx / var2.x) -(yy*yy / var2.y));

			_pt[x][y][0] = half_bin[0] - 0.5f + (x+0.5f-cenBand[0]) / bandwidth[0];
			_pt[x][y][1] = half_bin[1] - 0.5f + (y+0.5f-cenBand[1]) / bandwidth[1];
		}
	}
}


unsigned char sqrt(unsigned char x) {
        #define DECL(x,n,foo) (unsigned char) sqrt(double(foo+n)),
	#define DECL16(x,n,foo) BOOST_PP_REPEAT(16,DECL,n*16)
	static const unsigned char lookup[256] = {
	  		BOOST_PP_REPEAT(16,DECL16,0)
	};
	#undef DECL
	#undef DECL16

	return lookup[x];
}



