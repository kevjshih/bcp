
#include "poselet_detector.h"
#include <iostream>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include "hog_features.h"
#include <boost/timer.hpp>
#include <boost/format.hpp>

#include "channel_transform_view.h"
#include "imdebug.h"


using namespace std;
using namespace boost;
using namespace gil;
//using namespace sdl;

#pragma optimize("",off)

int main( int argc, char* argv[] ) {
	gil::rgb8_image_t img;
	gil::jpeg_read_image("data/test.jpg",img);

	poselet_detector pd;
	pd.init("data/poselets.xml");
	
	boost::timer t;
	t.restart();

	std::vector<object_hypothesis> hits;
	pd.detect_objects(const_view(img),hits);
	std::sort(hits.begin(), hits.end());

	double scoresum=0;
	for (size_t i=0; i<hits.size(); ++i) {
		if (i<4)
			hits[i].draw(view(img), rgb8_pixel_t(255,0,0), 3);
		scoresum+=hits[i].score();
	}
	debug_display_view(view(img));
	cout << "Found "<<hits.size()<<" objects with scoresum "<<scoresum<<" in "<<t.elapsed()<<" sec."<<endl;

//	while(1);
	return 0;
}