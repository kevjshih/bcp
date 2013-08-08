#pragma once
#include "config.h"
#include "hog_features.h"
#include "rapidxml.hpp"
#include "hypothesis.h"
#include "poselet_hit.h"
#include "model.h"
#include "object_hypothesis.h"
#include <iostream>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/numeric/resample.hpp>
#include <boost/gil/extension/numeric/sampler.hpp>
#include <boost/multi_array.hpp>
#include "image_pyramid.h"


class poselet_detector {
public:
	poselet_detector() {}
	void init(const char* model_name) { _model.init(model_name); }

	template <typename View> void detect_poselets(const View& v, poselet_hits_vector& hits_out) const;

	template <typename View> void detect_objects(const View& v, std::vector<object_hypothesis>& object_hits) const;

	template <typename View> void get_features_of_hits(const View& v, const poselet_hits_vector& hits, std::vector<std::vector<float> >& features) const;

	void hog2poselet_hits(const hog_features& hog, poselet_hits_vector& hits) const;
	const poselets_model& model() const { return _model; }
	void get_feature(const hog_features& hog, const int_point& p, const int_point& template_dims, std::vector<float>& feature) const;
private:
	typedef std::vector<size_t> selection_t;
	boost::multi_array<float,2> _poselet_svms;
	hog_features_generator _hg;
	poselets_model _model;
	
	void nonmax_suppress_hits(const poselet_hits_vector& hits_in, poselet_hits_vector& hits_out) const;
	void poselet_hits2bigq_poselet_hits(poselet_hits_vector& hits) const;
	void poselet_hits2hit_clusters(const poselet_hits_vector& hits, std::vector<poselet_hits_vector>& clustered_hits) const;
	void poselet_hits2objects(const poselet_hits_vector& hits, std::vector<object_hypothesis>& obj_hits) const;
	hit cluster2torso_hit(const poselet_hits_vector& hits, const std::vector<hypothesis>& hyps, const selection_t& cluster) const;

	void	save_poselet_hits(const poselet_hits_vector& hits, const char* filename) const;
	void	load_poselet_hits(      poselet_hits_vector& hits, const char* filename) const;
};


class image2hits {
public:
	image2hits(const poselet_detector* pd) : _hg(), _poselet_detector(pd) {}
	
	template <typename View>
	void compute_hits(const View& v) {
		_hits.clear();
		generate_image_pyramid(v, *this);
	}

	// called for each level of the image pyramid
	template <typename View>
	void operator()(const View& v, double scale) {
//		std::cout << "["<<v.width()<<","<<v.height()<<"]"<<std::endl;

//		std::cout << "Computing hog..."<<std::endl;
		hog_features hog;
		_hg.compute(v,hog);

//		std::cout << "Finding poselets...";
		poselet_hits_vector hits_of_scale;
		_poselet_detector->hog2poselet_hits(hog,hits_of_scale);
//		std::cout << "found at this scale: "<<hits_of_scale.size()<<std::endl;

		for (size_t i=0; i<hits_of_scale.size(); ++i)
			hits_of_scale[i].transform(cast_point<float_point>(hog.offset()-config::get().IMAGE_MARGIN/2+int_point(1,1)),float(1.0/scale));

		_hits.insert(_hits.end(), hits_of_scale.begin(), hits_of_scale.end());
	}

	poselet_hits_vector _hits;
private:
	hog_features_generator _hg;
	const poselet_detector* _poselet_detector;
};

//#pragma optimize("",off)

static void error_if(bool cond) {
	if (cond)
		throw;
}
// Given a set of hits and an image returns the features that generated these hits
class hits2features {
public:
 hits2features(const poselet_detector* pd, const poselet_hits_vector& hits) : _hits(hits), _poselet_detector(pd) {}
	
	template <typename View>
	void compute_features(const View& v) {
		_norm_width.resize(_hits.size());
		const poselets_model& model=_poselet_detector->model();
		for (size_t i=0; i<_hits.size(); ++i)
			_norm_width[i]=model[_hits[i].poselet_id()].dims().x;
		_features.resize(_hits.size());
		generate_image_pyramid(v, *this);

		for (size_t i=0; i<_hits.size(); ++i) {
			error_if(_features[i].empty());
			const poselet& ps=model[_hits[i].poselet_id()];
			double score=ps.probability(ps.inner_product(&_features[i][0]));
			error_if(fabs(score-_hits[i].score())>0.001);
		}
	}

	// called for each level of the image pyramid
	template <typename View>
	void operator()(const View& v, double scale) {
		const poselets_model& model=_poselet_detector->model();

		// find the indices of the hits that were active at this scale
		std::vector<size_t> hits_at_this_scale;
		for (size_t i=0; i<_hits.size(); ++i)
			if (fabs(_norm_width[i] - _hits[i].bounds().width()*scale)<0.01) {
				error_if(!_features[i].empty());
				hits_at_this_scale.push_back(i);
			}

		if (hits_at_this_scale.empty())
			return;

		hog_features hog;
		_hg.compute(v,hog);

		float_point offset(hog.offset().x - config::get().IMAGE_MARGIN.x/2.0f+1.0f,
						   hog.offset().y - config::get().IMAGE_MARGIN.y/2.0f+1.0f);

		for (size_t i=0; i<hits_at_this_scale.size(); ++i) {
			size_t j=hits_at_this_scale[i];
			int_point template_dims = model[_hits[j].poselet_id()].dims()/config::get().pix_per_bin() - int_point(1,1);
			float_point fp(float(_hits[j].bounds()._min.x * scale - offset.x) / config::get().pix_per_bin(),
						   float(_hits[j].bounds()._min.y * scale - offset.y) / config::get().pix_per_bin());
			assert(fabs(fp.x-int(fp.x))<0.0001 && fabs(fp.y-int(fp.y))<0.0001);
			int_point p=intround(fp);
			_features[j].resize(hog.bin_size()*template_dims.x*template_dims.y);
			_poselet_detector->get_feature(hog,p,template_dims,_features[j]);
		}
/*
	for (size_t dims_type=0; dims_type<_model.dims_groups().size(); ++dims_type) {
		int_point template_dims=_model.dims_groups()[dims_type]._dims;

		get_feature(hog,p,template_dims,feature);

		tr = hog.offset()-config::get().IMAGE_MARGIN/2+int_point(1,1);
		_bounds._min=(p*pix_per_bin+tr)/scale;

		hits.push_back(poselet_hit(pids[pd],score, 
		float_bounds(cast_point<float_point>(p*pix_per_bin),
						cast_point<float_point>((template_dims+int_point(1,1))*pix_per_bin))));

//		for (size_t i=0; i<hits_of_scale.size(); ++i)
//			hits_of_scale[i].transform(cast_point<float_point>(hog.offset()-config::get().IMAGE_MARGIN/2+int_point(1,1)),float(1.0/scale));
*/
	}

	std::vector<std::vector<float> > _features;
private:
	std::vector<int> _norm_width;
	poselet_hits_vector _hits;
	hog_features_generator _hg;
	const poselet_detector* _poselet_detector;
};

template <typename View> 
void poselet_detector::detect_poselets(const View& v, poselet_hits_vector& hits_out) const {
     image2hits ih(this);
     ih.compute_hits(v);
     nonmax_suppress_hits(ih._hits,hits_out);
}

template <typename View> 
void poselet_detector::detect_objects(const View& v, std::vector<object_hypothesis>& object_hits) const {
		poselet_hits_vector poselet_hits;
//		const char* filename = "D:/poselets/tmp/poselet_hits.xml";
		if (1) {
			detect_poselets(v,poselet_hits);
//			save_poselet_hits(poselet_hits,filename);
		}
//			load_poselet_hits(poselet_hits,filename);

		poselet_hits2bigq_poselet_hits(poselet_hits);
		poselet_hits2objects(poselet_hits,object_hits);
 }


template <typename View> 
void poselet_detector::get_features_of_hits(const View& v, const poselet_hits_vector& hits, std::vector<std::vector<float> >& features) const {
		hits2features h2f(this,hits);
		h2f.compute_features(v);
		using std::swap;
		swap(h2f._features, features);
	}
