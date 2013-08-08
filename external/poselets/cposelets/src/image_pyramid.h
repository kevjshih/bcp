#pragma once

#include "config.h"

// Given an image, generates scaled versions of it and calls the provided function object with them.
// The scaled version are padded using reflection
template <typename View, typename Op>
void generate_image_pyramid(const View& v, Op& op) {
	using namespace boost::gil;

	double scale=std::max(1.0,sqrt(config::get().DETECTION_IMG_MIN_NUM_PIX / double(v.size())));
	if (scale==1)
		scale=std::min(1.0, sqrt(config::get().DETECTION_IMG_MAX_NUM_PIX / double(v.size())));

	// create initial image
	typedef point2<ptrdiff_t> ptrdiff_point;
	int_point MARGIN = config::get().IMAGE_MARGIN;
	int_point half_margin=MARGIN/2;

	ptrdiff_point marginp = ptrdiff_pt(MARGIN);
	ptrdiff_point half_marginp=ptrdiff_pt(half_margin);

	image<typename View::value_type,false> imgs[2];
	int curImg=0;

	double min_scale=std::max(MARGIN.x/double(v.width()), MARGIN.y/double(v.height()));
	while (true) {
		// Rescale the image and add reflective margin
		int_point init_size(int(v.width()*scale), int(v.height()*scale));
		ptrdiff_point init_sizep=ptrdiff_pt(init_size);
		if (std::min(init_size.x,init_size.y)<64)
			break;
//		img.recreate(init_size+MARGIN);
		imgs[curImg].recreate(init_sizep+marginp);
		if (imgs[1-curImg].width()==0)
			resize_view<bilinear_sampler>(v,subimage_view(view(imgs[curImg]),half_marginp,init_sizep));
		else {
//			downsample_view(subimage_view(view(imgs[1-curImg]),half_margin,imgs[1-curImg].dimensions()-MARGIN),
//							subimage_view(view(imgs[curImg]),half_margin,init_size));
//			downsample_view(v,subimage_view(view(imgs[curImg]),half_margin,init_size));
//			resize_view<bilinear_sampler>(v,subimage_view(view(imgs[curImg]),half_margin,init_size));
			resize_view<bilinear_sampler>(subimage_view(view(imgs[1-curImg]),half_marginp,imgs[1-curImg].dimensions()-marginp),
										  subimage_view(view(imgs[curImg]),half_marginp,init_sizep));
		}


		copy_pixels(                     subimage_view(const_view(imgs[curImg]),half_margin.x,half_margin.y,init_size.x,half_margin.y),
					flipped_up_down_view(subimage_view(      view(imgs[curImg]),half_margin.x,0            ,init_size.x,half_margin.y)));
		copy_pixels(                     subimage_view(const_view(imgs[curImg]),half_margin.x,init_size.y              ,init_size.x,half_margin.y),
					flipped_up_down_view(subimage_view(      view(imgs[curImg]),half_margin.x,init_size.y+half_margin.y,init_size.x,half_margin.y)));
		copy_pixels(                        subimage_view(const_view(imgs[curImg]),half_margin.x,0,half_margin.x,init_size.y+MARGIN.y),
					flipped_left_right_view(subimage_view(      view(imgs[curImg]),0            ,0,half_margin.x,init_size.y+MARGIN.y)));
		copy_pixels(                        subimage_view(const_view(imgs[curImg]),init_size.x              ,0,half_margin.x,init_size.y+MARGIN.y),
					flipped_left_right_view(subimage_view(      view(imgs[curImg]),init_size.x+half_margin.x,0,half_margin.x,init_size.y+MARGIN.y)));

		op(const_view(imgs[curImg]),scale);

		scale/=config::get().PYRAMID_SCALE_RATIO;
		if (scale<min_scale)
			break;
		curImg=1-curImg;
	}
}
