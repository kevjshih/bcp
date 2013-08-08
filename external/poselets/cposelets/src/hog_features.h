#pragma once
#ifndef NO_OMP
#include <omp.h>
#endif

#include <math.h>
#include <boost/gil/gil_all.hpp>
#include <boost/multi_array.hpp>
#include <vector>
#include <boost/array.hpp>
#include "config.h"
#include "channel_transform_view.h"

using namespace boost;

// Represents HOG features associated with a given image
class hog_features {
public:
	std::vector<float>& values() { return _values; }
	const int_point& num_bins() const { return _num_bins; }
	const int_point& offset() const { return _offset; }
	int bin_size() const { return _bin_size; }

	// Returns a pointer to the first element in bin (bin_x,bin_y)
	const float* get_hog_at_bin(const int_point& p) const { return &_values[0] + _bin_size*(p.y*_num_bins.x + p.x); }
private:
	int _bin_size;
	int_point _num_bins;
	int_point _offset;
	int_point _step;
	std::vector<float> _values; // size = _num_bins.x * _num_bins.y * _cell_dims

	friend class hog_features_generator;
};

class hog_features_generator {
public:
	hog_features_generator();

	template <typename View> void compute(const View& src_view, hog_features& vals) const;

private:
	const float HOG_WTSCALE;
	const float HOG_NORM_EPS;
	const float HOG_NORM_EPS2;
	const float HOG_NORM_MAXVAL;
	const int_point NUM_HOG_BINS;
	const int_point bandwidth;
	const int_point HOG_CELL_DIMS;
	const int NUM_ANGLES;

	template <int N>
	void append2hog(int width, gil::gray8c_step_ptr_t ori_it, gil::gray32fc_step_ptr_t grad_it, int bin_size, 
		const boost::array<int,N>& offsets, const boost::array<float,N>& weights, float* hog_vals) const;
	void append2hog_core(int hog_offset, int hog_offset2, float w1, float w2, const boost::array<int,1>& offsets, const boost::array<float,1>& weights, float* hog_vals) const;
	void append2hog_core(int hog_offset, int hog_offset2, float w1, float w2, const boost::array<int,2>& offsets, const boost::array<float,2>& weights, float* hog_vals) const;
	void append2hog_core(int hog_offset, int hog_offset2, float w1, float w2, const boost::array<int,4>& offsets, const boost::array<float,4>& weights, float* hog_vals) const;

	boost::multi_array<float,2> _w;
	boost::multi_array<float_point,2> _pt;
};

unsigned char sqrt(unsigned char x);

template <typename ChannelValue>
struct sqrt_channels_op : public std::unary_function<ChannelValue,ChannelValue> {
	ChannelValue operator()(ChannelValue src) const { return sqrt(src); }
};

/*
// View is a homogeneous view
// Computes the gradient magnitute and orientation at each pixel. Uses central difference, i.e. filters [-1 0 1]
// If the source has multiple channels, uses the channel with largest gradient magnitude
// Sets the values at the fringe to the values if their neighbors on the inside
template <typename View>
void compute_gradient_mag_oriOLD(const View& v, const gil::gray32f_view_t& gmag, const gil::gray8_view_t& gori) {
	assert(v.dimensions()==gmag.dimensions() && gmag.dimensions()==gori.dimensions());
	if (v.height()<2 || v.width()<2)
		return;
	static const float M_PI=3.14159265358979323846f;

	typename View::xy_locator::cached_location_t above = v.pixels().cache_location(0,-1);
	typename View::xy_locator::cached_location_t below = v.pixels().cache_location(0, 1);
	typename View::xy_locator::cached_location_t left  = v.pixels().cache_location(-1, 0);
	typename View::xy_locator::cached_location_t right = v.pixels().cache_location(1, 0);
	int_point max_pt=v.dimensions()-int_point(1,1);
	for (int y=1; y<max_pt.y; ++y) {
		typename View::xy_locator loc=v.xy_at(1,y);
		gil::gray32f_view_t::x_iterator mag_xit=gmag.row_begin(y);
		gil::gray8_view_t  ::x_iterator ori_xit=gori.row_begin(y);
		for (int x=1; x<max_pt.x; ++x) {
			int max_channel=0;
			float max_mag2=0;
			for (int c=0; c<gil::num_channels<View>::value; ++c) {
				float dx=float(-loc[left ][c] + loc[right][c]);
				float dy=float(-loc[above][c] + loc[below][c]);
				float mag2=dx*dx+dy*dy;
				if (mag2>max_mag2) {
					max_mag2=mag2;
					max_channel=c;
				}
			}
			mag_xit[x]=sqrt(max_mag2);
			float dx=float(-loc[left ][max_channel] + loc[right][max_channel]);
			float dy=float(-loc[above][max_channel] + loc[below][max_channel]);

			float angle=atan2f(dy,dx);
			if ((angle = angle * (180.0f/M_PI) + 180.0f) > 180.0f)
				angle = angle-180.0f;
			angle=floorf(angle);
			assert(angle>=0);

			ori_xit[x]=int(angle);
			++loc.x();
		}
		// pad the first and last column
		mag_xit[0]=mag_xit[1];
		mag_xit[max_pt.x]=mag_xit[max_pt.x-1];
		ori_xit[0]=ori_xit[1];
		ori_xit[max_pt.x]=ori_xit[max_pt.x-1];
	}
	// pad the first and last row
	std::copy(gmag.row_begin(1),gmag.row_end(1),gmag.row_begin(0));
	std::copy(gmag.row_begin(max_pt.y-1),gmag.row_end(max_pt.y-1),gmag.row_begin(max_pt.y));
	std::copy(gori.row_begin(1),gori.row_end(1),gori.row_begin(0));
	std::copy(gori.row_begin(max_pt.y-1),gori.row_end(max_pt.y-1),gori.row_begin(max_pt.y));
}
*/
	const float rad2grad=180.0f / 3.14159265358979323846f;

template <typename View>
void compute_gradient_mag_ori(const View& v, const gil::gray32f_view_t& gmag, const gil::gray8_view_t& gori) {
	assert(v.dimensions()==gmag.dimensions() && gmag.dimensions()==gori.dimensions());
	if (v.height()<2 || v.width()<2)
		return;

	//	typename View::xy_locator::cached_location_t above = v.pixels().cache_location(0,-1);
	//typename View::xy_locator::cached_location_t below = v.pixels().cache_location(0, 1);
	//typename View::xy_locator::cached_location_t left  = v.pixels().cache_location(-1, 0);
	//typename View::xy_locator::cached_location_t right = v.pixels().cache_location(1, 0);
	typename View::point_t max_pt=v.dimensions();
	for (int y=0; y<max_pt.y; ++y) {
		typename View::xy_locator loc=v.xy_at(0,std::min(std::max(y,1),int(max_pt.y)-2));
		gil::gray32f_view_t::x_iterator mag_xit=gmag.row_begin(y);
		gil::gray8_view_t  ::x_iterator ori_xit=gori.row_begin(y);
		typename View::x_iterator src_xit=v.x_at(1,y);
		typename View::x_iterator src_above,src_below;
		if (y==0) {
			src_above=v.x_at(0,0);
			src_below=v.x_at(0,2);
		} else if (y==max_pt.y-1) {
			src_above=v.x_at(0,max_pt.y-3);
			src_below=v.x_at(0,max_pt.y-1);
		} else {
			src_above=v.x_at(0,y-1);
			src_below=v.x_at(0,y+1);
		}
		for (int x=0; x<max_pt.x; ++x) {
			int max_channel=0;
			float max_mag2=0;
			for (int c=0; c<gil::num_channels<View>::value; ++c) {
				float dx=float(-src_xit[-1][c] + src_xit[1][c]);
				float dy=float(-src_above[x][c] + src_below[x][c]);

				float mag2=dx*dx+dy*dy;
				if (mag2>max_mag2) {
					max_mag2=mag2;
					max_channel=c;
				}
			}
			mag_xit[x]=sqrt(max_mag2);
			float dx=float(-src_xit[-1] [max_channel] + src_xit[1][max_channel]);
			float dy=float(-src_above[x][max_channel] + src_below[x][max_channel]);

			float angle=atan2f(dy,dx);
			angle = angle * rad2grad + 180.0f;
			if (angle > 180.0f)
				angle = angle-180.0f;
//			angle=floorf(angle);
//			assert(angle>=0);

			ori_xit[x]=int(angle);
			if (x>0 && x<max_pt.x-2)
				++src_xit;
		}
	}
}

template <typename View>
void hog_features_generator::compute(const View& src_view, hog_features& hog_features) const
{
	int_point num_bins(int(floor((src_view.width ()-2)/float(bandwidth.x)) - NUM_HOG_BINS.x+1),
							   int(floor((src_view.height()-2)/float(bandwidth.y)) - NUM_HOG_BINS.y+1));

	int_point topleft(int(floorf (0.5f*(src_view.width () - (num_bins.x-1)*bandwidth.x - HOG_CELL_DIMS.x))),
							 int(floorf (0.5f*(src_view.height() - (num_bins.y-1)*bandwidth.y - HOG_CELL_DIMS.y))));

	const int hog_bin_size = NUM_ANGLES * NUM_HOG_BINS.x * NUM_HOG_BINS.y;
	const int num_hog_bins = num_bins.x * num_bins.y;

	std::vector<float>& hog_vals=hog_features._values;

	hog_vals.clear();
	hog_vals.resize(num_hog_bins*hog_bin_size,0);

	// Sqrt the input values
	typename gil::image_type<float, gil::layout<typename gil::color_space_type<View>::type> >::type img(src_view.dimensions());
	copy_pixels(channel_transform_view(src_view, sqrt_channels_op<float>()),view(img));

	// Compute the gradient magnitude and orientation
	gil::gray32f_image_t grad_mag(src_view.dimensions());
	gil::gray8_image_t grad_ori(src_view.dimensions());
	compute_gradient_mag_ori(view(img), view(grad_mag),view(grad_ori));

	// Distribute the orientation gradients into histogram bins
	int_point tl;
//#pragma omp for
	for (int i = 0; i < num_bins.y; ++i) {
//#pragma omp parallel
		for (tl.y = 0; tl.y < HOG_CELL_DIMS.y; ++tl.y)
			for (tl.x = 0; tl.x < HOG_CELL_DIMS.x; ++tl.x){
			{
				int_point tl_rowbegin=topleft+tl+int_point(0,i*bandwidth.y);
				gil::gray8c_step_ptr_t ori_it = gil::make_step_iterator(const_view(grad_ori).x_at(tl_rowbegin.x, tl_rowbegin.y),bandwidth.x);
				gil::gray32fc_step_ptr_t grad_it = gil::make_step_iterator(const_view(grad_mag).x_at(tl_rowbegin.x, tl_rowbegin.y),bandwidth.x*sizeof(float));

				// this generates better results but is wrong!
//				int_point fl_new=cast_point<int_point>(_pt[tl.x][tl.y]);

				int_point fl=cast_point<int_point>(gil::ifloor(_pt[tl.x][tl.y]));
//				gil::point2<ptrdiff_t> fl=gil::ifloor(_pt[tl.x][tl.y]);
				float_point fr=_pt[tl.x][tl.y]-float_point(float(fl.x),float(fl.y));

				float* hog_ptr=&hog_vals[hog_bin_size*num_bins.x*i];

				float w=_w[tl.x][tl.y];
				if (fl.x==-1) { // (cases 3,4)
					if (fl.y==-1) { // case 4
						boost::array<int,1> offsets={{0}};
						boost::array<float,1> weights={{w*fr.x*fr.y}};
						append2hog<1>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else if (fl.y+1==NUM_HOG_BINS.y) { // case 3
//						boost::array<int,1> offsets={{fl.y*NUM_HOG_BINS.x}};
						boost::array<int,1> offsets={{fl.y}};
						boost::array<float,1> weights={{w*fr.x*(1-fr.y)}};
						append2hog<1>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else { // case 3+4
//						boost::array<int,2> offsets={{fl.y*NUM_HOG_BINS.x, (fl.y+1)*NUM_HOG_BINS.x}};
						boost::array<int,2> offsets={{fl.y, fl.y+1}};
						boost::array<float,2> weights={{w*fr.x*(1-fr.y), w*fr.x*fr.y}};
						append2hog<2>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					}
				} else if (fl.x+1==NUM_HOG_BINS.x) { // (cases 1,2)
					if (fl.y==-1) { // case 2
//						boost::array<int,1> offsets={{fl.x}};
						boost::array<int,1> offsets={{fl.x*NUM_HOG_BINS.y}};
						boost::array<float,1> weights={{w*(1-fr.x)*fr.y}};
						append2hog<1>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else if (fl.y+1==NUM_HOG_BINS.y) { // case 1
//						boost::array<int,1> offsets={{fl.y*NUM_HOG_BINS.x + fl.x}};
						boost::array<int,1> offsets={{fl.y + fl.x*NUM_HOG_BINS.y}};
						boost::array<float,1> weights={{w*(1-fr.x)*(1-fr.y)}};
						append2hog<1>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else { // case 1+2
//						boost::array<int,2> offsets={{fl.y*NUM_HOG_BINS.x + fl.x, (fl.y+1)*NUM_HOG_BINS.x + fl.x}};
						boost::array<int,2> offsets={{fl.y + fl.x*NUM_HOG_BINS.y, (fl.y+1) + fl.x*NUM_HOG_BINS.y}};
						boost::array<float,2> weights={{w*(1-fr.x)*(1-fr.y), w*(1-fr.x)*fr.y}};
						append2hog<2>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					}
				} else { // (cases 1,2,3,4)
					if (fl.y==-1) { // case 2,4
//						boost::array<int,2> offsets={{fl.x, fl.x+1}};
						boost::array<int,2> offsets={{fl.x*NUM_HOG_BINS.y, (fl.x+1)*NUM_HOG_BINS.y}};
						boost::array<float,2> weights={{w*(1-fr.x)*fr.y, w*fr.x*fr.y}};
						append2hog<2>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else if (fl.y+1==NUM_HOG_BINS.y) { // cases 1,3
//						boost::array<int,2> offsets={{fl.y*NUM_HOG_BINS.x + fl.x, fl.y*NUM_HOG_BINS.x + fl.x+1}};
						boost::array<int,2> offsets={{fl.y + fl.x*NUM_HOG_BINS.y, fl.y + (fl.x+1)*NUM_HOG_BINS.y}};
						boost::array<float,2> weights={{w*(1-fr.x)*(1-fr.y), w*fr.x*(1-fr.y)}};
						append2hog<2>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					} else { // cases 1,3,2,4
//						boost::array<int,4> offsets={{fl.y*NUM_HOG_BINS.x + fl.x, fl.y*NUM_HOG_BINS.x + fl.x+1, (fl.y+1)*NUM_HOG_BINS.x + fl.x, (fl.y+1)*NUM_HOG_BINS.x + fl.x+1}};
//						boost::array<float,4> weights={{w*(1-fr.x)*(1-fr.y),w*fr.x*(1-fr.y),w*(1-fr.x)*fr.y,w*fr.x*fr.y}};
						boost::array<int,4> offsets={{fl.y + fl.x*NUM_HOG_BINS.y, fl.y+1 + fl.x*NUM_HOG_BINS.y, fl.y + (fl.x+1)*NUM_HOG_BINS.y, fl.y+1 + (fl.x+1)*NUM_HOG_BINS.y}};
						boost::array<float,4> weights={{w*(1-fr.x)*(1-fr.y),w*(1-fr.x)*fr.y,w*fr.x*(1-fr.y),w*fr.x*fr.y}};
						append2hog<4>(num_bins.x, ori_it, grad_it, hog_bin_size, offsets,weights, hog_ptr);
					}
				}
			}
		}
	}

	// Normalize the histogram bins
	for (size_t i=0; i<hog_vals.size(); i+=hog_bin_size) {
		float sumsq=0;
		for (size_t j=i; j<i+hog_bin_size; ++j)
			sumsq+=hog_vals[j]*hog_vals[j];
		sumsq=sqrtf(sumsq);

		for (size_t j=i; j<i+hog_bin_size; ++j)
			hog_vals[j] =std::min((float)HOG_NORM_MAXVAL, hog_vals[j]/(sumsq+HOG_NORM_EPS*hog_bin_size));
		
		sumsq=0;
		for (size_t j=i; j<i+hog_bin_size; ++j)
			sumsq+=hog_vals[j]*hog_vals[j];
		sumsq=sqrtf(sumsq)+HOG_NORM_EPS2;

		for (size_t j=i; j<i+hog_bin_size; ++j)
			hog_vals[j]/=sumsq;		
	}

	hog_features._num_bins=num_bins;
	hog_features._bin_size=hog_bin_size;
	hog_features._offset=topleft;
	hog_features._step=bandwidth;
}

inline void hog_features_generator::append2hog_core(int hog_offset, int hog_offset2, float w1, float w2,
								   const boost::array<int,1>& offsets, const boost::array<float,1>& weights, 
								   float* hog_vals) const {
	hog_vals[hog_offset +offsets[0]]+=weights[0]*w1;
	hog_vals[hog_offset2+offsets[0]]+=weights[0]*w2;
}

inline void hog_features_generator::append2hog_core(int hog_offset, int hog_offset2, float w1, float w2,
								   const boost::array<int,2>& offsets, const boost::array<float,2>& weights, 
								   float* hog_vals) const {
	hog_vals[hog_offset +offsets[0]]+=weights[0]*w1;
	hog_vals[hog_offset2+offsets[0]]+=weights[0]*w2;
	hog_vals[hog_offset +offsets[1]]+=weights[1]*w1;
	hog_vals[hog_offset2+offsets[1]]+=weights[1]*w2;
}

inline void hog_features_generator::append2hog_core(int hog_offset, int hog_offset2, float w1, float w2,
								   const boost::array<int,4>& offsets, const boost::array<float,4>& weights, 
								   float* hog_vals) const {
	hog_vals[hog_offset +offsets[0]]+=weights[0]*w1;
	hog_vals[hog_offset2+offsets[0]]+=weights[0]*w2;
	hog_vals[hog_offset +offsets[1]]+=weights[1]*w1;
	hog_vals[hog_offset2+offsets[1]]+=weights[1]*w2;
	hog_vals[hog_offset +offsets[2]]+=weights[2]*w1;
	hog_vals[hog_offset2+offsets[2]]+=weights[2]*w2;
	hog_vals[hog_offset +offsets[3]]+=weights[3]*w1;
	hog_vals[hog_offset2+offsets[3]]+=weights[3]*w2;
}

template <int N>
void hog_features_generator::append2hog(int width, gil::gray8c_step_ptr_t ori_it, gil::gray32fc_step_ptr_t grad_it, int bin_size, 
							  const boost::array<int,N>& offsets, const boost::array<float,N>& weights, float* hog_vals) const {
	// TODO: This can be avoided if bottleneck (precompute when initializing array)
	boost::array<int,N> offsets2;
	for (int i=0; i<N; ++i)
		offsets2[i]=offsets[i]*NUM_ANGLES;

	for (int j = 0; j < width; ++j) {
		const float step=180.0f/NUM_ANGLES;
//		float ofrac = (*ori_it - step/2) / step;
		float ofrac = (*ori_it)/step - 0.5f;
		if (ofrac<0)
			ofrac+=NUM_ANGLES;
//		int a=int(floor(ofrac));
		int a=int(ofrac);
		++ori_it;

		int a2=(a+1)%NUM_ANGLES;

		float sori = 1-(ofrac-a);
		float smag = (*grad_it)[0];
		++grad_it;
		append2hog_core(bin_size*j+a,bin_size*j+a2,sori*smag,(1-sori)*smag,offsets2,weights,hog_vals);
	}
}
/*
template <int N>
void hog_features_generator::append2hog(int width, gil::gray8c_step_ptr_t ori_it, gil::gray32fc_step_ptr_t grad_it, int bin_size, 
							  const boost::array<int,N>& offsets, const boost::array<float,N>& weights, float* hog_vals) const {
	for (int j = 0; j < width; ++j) {
		const float step=180.0f/NUM_ANGLES;
		float ofrac = (*ori_it - step/2) / step;
		if (ofrac<0)
			ofrac+=NUM_ANGLES;
		int a=int(floor(ofrac));
		++ori_it;

		int a2=(a+1)%NUM_ANGLES;
		int hog_offset_a  = bin_size*j+a *NUM_HOG_BINS.x*NUM_HOG_BINS.y;
		int hog_offset_a2 = bin_size*j+a2*NUM_HOG_BINS.x*NUM_HOG_BINS.y;

		float sori = 1-(ofrac-a);
		float smag = (*grad_it)[0];
		++grad_it;
		append2hog_core(hog_offset_a,hog_offset_a2,sori*smag,(1-sori)*smag,offsets,weights,hog_vals);
	}
}
*/
