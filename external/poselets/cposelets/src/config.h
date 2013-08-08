#pragma once

#ifdef min
	#undef min
#endif
#ifdef max
	#undef max
#endif
#include "bounds.h"
#include <iostream>
#include <vector>

typedef boost::gil::point2<int> int_point;
typedef boost::gil::point2<float> float_point;
typedef boost::gil::point2<double> double_point;
//static float_point int2float_point(const int_point& p) { return float_point(float(p.x),float(p.y)); }

template <typename DstPoint, typename SrcPoint>
DstPoint cast_point(const SrcPoint& p) { return DstPoint(typename DstPoint::value_type(p.x),typename DstPoint::value_type(p.y)); }

inline boost::gil::point2<ptrdiff_t> ptrdiff_pt(const int_point& p) { return boost::gil::point2<ptrdiff_t>(p.x,p.y); }

template <typename DstBounds, typename SrcBounds>
DstBounds cast_bounds(const SrcBounds& b) { return DstBounds(cast_point<typename DstBounds::point_type>(b._min), cast_point<typename DstBounds::point_type>(b._max), false); }

namespace boost { namespace gil {
template <typename T> GIL_FORCEINLINE
point2<T> operator/(const point2<T>& p, int t)      { return t==0 ? point2<T>(0,0):point2<T>(p.x/t,p.y/t); }

template <typename T> GIL_FORCEINLINE
point2<T> operator*(const point2<T>& p, float t)      { return point2<T>(p.x*t,p.y*t); }

template <typename Point1, typename Point2, typename Scalar>
inline void interpolate_point(const Point2& p2, Scalar w2, Point1& accum, Scalar w1=1) {
	accum.x=accum.x*w1 + p2.x*w2;
	accum.y=accum.y*w1 + p2.y*w2;
}

template <typename Point, typename Scalar>
inline Point get_interpolated_point(const Point& p1, Scalar w1, const Point& p2, Scalar w2) {
	return Point(p1.x*w1 + p2.x*w2, p1.y*w1 + p2.y*w2);
}


template <typename Point>
inline double l2_norm(const Point& p) {
	return sqrt(p.x*p.x + p.y*p.y);
}

template <typename T> std::ostream& operator<<(std::ostream& os, const point2<T>& p) { return os << p.x <<" "<<p.y<<" ";; }
template <typename T> std::istream& operator>>(std::istream& is,       point2<T>& p) { return is >> p.x >> p.y; }

} }

inline float_bounds operator*(const float_bounds& b, float scale) {
    return float_bounds(b._min*scale,b._max*scale,true);
}

struct config {
	static const config& get() {
		static config gConfig;
		return gConfig;
	}

	const int_point CELL_DIMS;
	const int_point N_HOG_DIMS;
	const int DETECTION_IMG_MIN_NUM_PIX;
	const int DETECTION_IMG_MAX_NUM_PIX;
	const int_point IMAGE_MARGIN;
	const double PYRAMID_SCALE_RATIO;
	const float HYPOTHESIS_PRIOR_VAR;
	const float HYPOTHESIS_PRIOR_VARIANCE_WEIGHT;
	const int NUM_ANGLES;
	const size_t POSELET_CLUSTER_HITS2CHECK;
	const size_t HYP_CLUSTER_MAXIMUM;
	const float TORSO_ASPECT_RATIO;
	const float CLUSTER_HITS_CUTOFF;
	const float HYP_CLUSTER_THRESH;

	int pix_per_bin() const { return CELL_DIMS.x/N_HOG_DIMS.x; }
	int bin_size() const { return N_HOG_DIMS.x*N_HOG_DIMS.y*NUM_ANGLES; }
private:
	config();
};

