#include "object_hypothesis.h"
#include "model.h"
#include <iostream>

// Constructs object hypothesis from a set of clustered poselet hits.
// Computes the score and bounds
void object_hypothesis::init(const poselet_hits_vector& poselet_hits, const poselets_model& model) {
	_poselet_hits = poselet_hits;
	_hit = hit(compute_score(model), compute_bounds(model));
}


double object_hypothesis::compute_score(const poselets_model& model) const {
	double score=0;
	for (size_t i=0; i<_poselet_hits.size(); ++i) {
		score+=_poselet_hits[i].score() * model.wts(_poselet_hits[i].poselet_id());
	}
	return score;
}

float_bounds predict_bounds(const poselet_hit& hit, const poselet& poselet) {
	float scale=std::min(hit.bounds().height(),hit.bounds().width());
	double_point image2poselet_ctr=hit.bounds().center();
	float_bounds scaled_bounds = poselet.obj_bounds()*scale;
	double_point poselet2bounds_ctr=scaled_bounds.center();

	float_point image2bounds_ctr = cast_point<float_point>(image2poselet_ctr+poselet2bounds_ctr);
	return float_bounds(image2bounds_ctr-scaled_bounds.dimensions()/2, scaled_bounds.dimensions());
}

float_bounds object_hypothesis::compute_bounds(const poselets_model& model) const {
	float_bounds pred_bounds(0,0,0,0);
	float_bounds weight_sum=pred_bounds;
	for (size_t i=0; i<_poselet_hits.size(); ++i) {
		const poselet_hit& hit=_poselet_hits[i];
		float_bounds pred=predict_bounds(hit,model[hit.poselet_id()]);

		for (size_t j=0; j<4; ++j) {
			float w=float(hit.score()) / (model[hit.poselet_id()].obj_bounds_var()[j]*hit.bounds().width());
			weight_sum[j]+=w;
			pred_bounds[j]+=pred[j]*w;
		}
	}

	for (size_t j=0; j<4; ++j)
		pred_bounds[j]/=weight_sum[j];
	return pred_bounds;
}
