#include "poselet_detector.h"
#include "poselet_cluster.h"
#include "agglomerative_cluster.h"
#include "xml_utils.h"
#include <algorithm>
#include <list>
#include <fstream>
#include <boost/lexical_cast.hpp>

using namespace std;


void poselet_detector::get_feature(const hog_features& hog, const int_point& top_left, const int_point& template_dims, std::vector<float>& feature) const {
	int_point p=top_left;
	float* f=&feature[0];
	int width=template_dims.x*hog.bin_size();
	assert(int(feature.size())==width*template_dims.y);
	for (int y=0; y<template_dims.y; ++y) {
		const float* src=hog.get_hog_at_bin(p);
		std::copy(src,src+width,f);
		f+=width;
		++p.y;
	}
}

void poselet_detector::hog2poselet_hits(const hog_features& hog, poselet_hits_vector& hits) const {
	const float pix_per_bin=float(config::get().pix_per_bin());

	for (size_t dims_type=0; dims_type<_model.dims_groups().size(); ++dims_type) {
		int_point template_dims=_model.dims_groups()[dims_type]._dims;
		const std::vector<size_t>& pids=_model.dims_groups()[dims_type]._poselet_ids;

		std::vector<float> feature(hog.bin_size()*template_dims.x*template_dims.y);
        	std::vector<double> scores(pids.size());

		int_point p;
		for (p.y=0; p.y<=hog.num_bins().y-template_dims.y; ++p.y)
			for (p.x=0; p.x<=hog.num_bins().x-template_dims.x; ++p.x) {
				// extract the feature at the current bin coords
				get_feature(hog,p,template_dims,feature);

 #pragma omp parallel for
				for (int pd=0; pd<int(pids.size()); ++pd) {
				       scores[pd] = _model[pids[pd]].inner_product(&feature[0]);
				}
				for (size_t pd=0; pd<pids.size(); ++pd)
					if (scores[pd]>=0)
						hits.push_back(poselet_hit(int(pids[pd]),scores[pd], 
								float_bounds(p.x*pix_per_bin, p.y*pix_per_bin,
											 (template_dims.x+1)*pix_per_bin, (template_dims.y+1)*pix_per_bin)));
			       
			}
	}
}

//#pragma optimize("",off)
void poselet_detector::nonmax_suppress_hits(const poselet_hits_vector& hits_in, poselet_hits_vector& hits_out) const {
	list<poselet_hit> merged;

	// Split the hits by poselet id
	vector<vector<poselet_hit> > hits_by_poselet(_model.num_poselets());
	for (size_t i=0; i<hits_in.size(); ++i)
		hits_by_poselet[hits_in[i].poselet_id()].push_back(hits_in[i]);

	for (size_t p=0; p<_model.num_poselets(); ++p) {
		// sort the hits of poselet p by score
		std::sort(hits_by_poselet[p].begin(), hits_by_poselet[p].end());

		// append them at the end of merged. Set first_s as iterator to the beginning of the sorted elements of p in merged
		list<poselet_hit>::iterator first_s;
		if (merged.empty()) {
			merged.assign(hits_by_poselet[p].begin(), hits_by_poselet[p].end());
			first_s=merged.begin();
		} else {
			first_s=merged.end(); --first_s;
			merged.insert(merged.end(),hits_by_poselet[p].begin(), hits_by_poselet[p].end());
			++first_s;
		}

		// remove any of the hits that are overlapped by more than 0.5 by another hit of the same poselet with higher score
		list<poselet_hit>::iterator listend=merged.end();
		if (hits_by_poselet[p].size()>1) {
			list<poselet_hit>::iterator fit=first_s;
			do {
				assert(fit!=listend);
				list<poselet_hit>::iterator bit=fit;
				++bit;
				if (bit==listend)
					break;
				do {
					if (bounds_overlap(fit->bounds(), bit->bounds())>=0.5)
						bit=merged.erase(bit);
					else
						++bit;
				} while (bit!=listend);
				++fit;
			} while (fit!=listend);
		}

		// set the score of the survived hits to probability. It is still sorted because of monotonicity of the logistic
		for (list<poselet_hit>::iterator it=first_s; it!=listend; ++it)
			it->set_score(_model[p].probability(float(it->score())));

		// merge the survivor hits into the list. The list is sorted by score
		std::inplace_merge(merged.begin(), first_s, merged.end());
	}

	hits_out.assign(merged.begin(), merged.end());
}



hit poselet_detector::cluster2torso_hit(const poselet_hits_vector& hits, const vector<hypothesis>& hyps, const vector<size_t>& cluster) const {
	// Get the hips/shoulders coordinates
	const int kp_idx[]={1-1, 4-1, 7-1, 10-1};//L_Shoulder, R_Shoulder, L_Hip, R_Hip
	double_point torso[4];
	double weight_sum=0;

	for (size_t j=0; j<4; ++j)
		torso[j].x=torso[j].y=0;

	for (size_t i=0; i<cluster.size(); ++i) {
		size_t hit_id=cluster[i];
		double w=hits[hit_id].score();
		for (size_t j=0; j<4; ++j) {
			interpolate_point(hyps[hit_id].kp_mean(kp_idx[j]),w,torso[j]);
		}
		weight_sum+=w;
	}
	for (size_t j=0; j<4; ++j)
		torso[j]/=weight_sum;
	double_point m_shoulder = (torso[0]+torso[1])/2;
	double_point m_hip      = (torso[2]+torso[3])/2;
	double_point ctr = (m_shoulder+m_hip)/2;
	double torso_len = l2_norm(m_hip-m_shoulder);
	double_point torso_dims(torso_len/config::get().TORSO_ASPECT_RATIO,torso_len);
	return hit(weight_sum,float_bounds(cast_point<float_point>(ctr-torso_dims/2),cast_point<float_point>(torso_dims)));
}

struct hit_cluster {
	hit _hit;
	std::vector<size_t> _src_idx;
};


struct hit_overlap_dist_fn : public std::binary_function<float,hit_cluster,hit_cluster> {
	float operator()(const hit_cluster& h1, const hit_cluster& h2) const {
		return 1-float(bounds_overlap(h1._hit.bounds(),h2._hit.bounds()));
	}
};

struct hit_merge_fn : public std::binary_function<hit,hit,hit> {
	hit_cluster operator()(const hit_cluster& h1, const hit_cluster& h2) const {
		double score_sum=h1._hit.score() + h2._hit.score();
		hit_cluster ret;
		ret._hit = hit(score_sum,get_interpolated_bounds(h1._hit.bounds(),float(h1._hit.score()/score_sum), h2._hit.bounds(),float(h2._hit.score()/score_sum)));
		ret._src_idx=h1._src_idx;
		ret._src_idx.insert(ret._src_idx.end(), h2._src_idx.begin(), h2._src_idx.end());
		return ret;
	}
};

void poselet_detector::save_poselet_hits(const poselet_hits_vector& hits, const char* filename) const {
	ofstream os(filename);
	os << "<?xml version=\"1.0\" encoding=\"utf-8\"?>" << endl;
	os << "<poselet_hits num=\""<<hits.size()<<"\">"<<endl;
	for (size_t i=0; i<hits.size(); ++i)
		os << hits[i];

	os << "</poselet_hits>"<<endl;
	os.close();
}

//#pragma optimize("",off)
void poselet_detector::load_poselet_hits(      poselet_hits_vector& hits, const char* filename) const {
	ifstream is(filename);
	string line;
	getline(is,line,is.widen('\255'));
	is.close();
	using namespace rapidxml;
	xml_document<> doc;
	doc.parse<parse_non_destructive>(const_cast<char*>(line.c_str()));

	xml_node<> *root = doc.first_node("poselet_hits");
	assert(root);
	int num = boost::lexical_cast<int>(get_xml_value(root->first_attribute("num")));
	hits.resize(num); 

	size_t i=0;
	xml_node<> *node = root->first_node("poselet_hit");
	assert(node);
	do {
		hits[i].init(node);
		++i;
	} while ((node=node->next_sibling("poselet_hit")));
	assert(int(i)==num);
}
//#pragma optimize("",on)

// This can be sped up if necessary: hypotheses need not be initialized twice. The distance need not be computed twice
void poselet_detector::poselet_hits2bigq_poselet_hits(poselet_hits_vector& hits) const {
	// instantiate the hypotheses (duplicated in poselet_hits2objects)
	vector<hypothesis> hyps(hits.size());
	for (size_t i=0; i<hits.size(); ++i) {
		float_point ctr(hits[i].min_pt().x + hits[i].dims().x/2, hits[i].min_pt().y + hits[i].dims().y/2);
		hyps[i]=_model[hits[i].poselet_id()].get_hypothesis();
		hyps[i].transform(ctr, min(hits[i].dims().x, hits[i].dims().y));
	}

	vector<float> bigq_scores(hits.size());

	vector<float> bigq_features(_model.num_poselets(),0);
	for (size_t i=0; i<hits.size(); ++i) {
		std::fill(bigq_features.begin(), bigq_features.end(), 0.0f);
		for (size_t j=0; j<hits.size(); ++j) {
			if ((bigq_features[hits[j].poselet_id()] < hits[j].score()) && 
				(hyps[i].distance(hyps[j]) < config::get().HYP_CLUSTER_THRESH))
				bigq_features[hits[j].poselet_id()] = float(hits[j].score());
		}
		bigq_scores[i]=float(_model[hits[i].poselet_id()].bigq_probability(&bigq_features[0]));
	}

	for (size_t i=0; i<hits.size(); ++i)
		hits[i].set_score(bigq_scores[i]);
}

//#pragma optimize("",off)

void poselet_detector::poselet_hits2objects(const poselet_hits_vector& hits, std::vector<object_hypothesis>& obj_hits) const {
	// instantiate the hypotheses
	vector<hypothesis> hyps(hits.size());
	for (size_t i=0; i<hits.size(); ++i) {
		float_point ctr(hits[i].min_pt().x + hits[i].dims().x/2, hits[i].min_pt().y + hits[i].dims().y/2);
		hyps[i]=_model[hits[i].poselet_id()].get_hypothesis();
		hyps[i].transform(ctr, min(hits[i].dims().x, hits[i].dims().y));
	}
	
	// greedy cluster
	vector<selection_t> hit_clusters;
	cluster_poselet_hits(hits,hyps,_model.cluster_thresh(), hit_clusters);

	// Compute torso hits for each cluster
	vector<hit_cluster> torso_hits(hit_clusters.size());
	torso_hits.reserve(hit_clusters.size());
	for (size_t c=0; c<hit_clusters.size(); ++c) {
		torso_hits[c]._hit=cluster2torso_hit(hits,hyps,hit_clusters[c]);
		torso_hits[c]._src_idx.push_back(c);
	}

	// Cluster them
	vector<hit_cluster> clustered_torso_hits;
	agglomerative_cluster(torso_hits.begin(),torso_hits.size(), hit_overlap_dist_fn(), hit_merge_fn(), config::get().CLUSTER_HITS_CUTOFF, back_inserter(clustered_torso_hits)); 
	
	size_t num_objects=clustered_torso_hits.size();

	// for each cluster find all the hits in it. Remove any duplicate hits, or hits of the same poselet type and smaller score
	obj_hits.clear();
	obj_hits.resize(num_objects);

	selection_t pav(_model.num_poselets());
  poselet_hits_vector phv;
	for (size_t c=0; c<num_objects; ++c) {
		std::fill(pav.begin(),pav.end(),-1);
		const std::vector<size_t>& src_idx=clustered_torso_hits[c]._src_idx;
		for (size_t j=0; j<src_idx.size(); ++j) {
			const selection_t& ssrc_idx=hit_clusters[src_idx[j]];
			for (size_t k=0; k<ssrc_idx.size(); ++k) {
				size_t hit_idx = ssrc_idx[k];
				size_t poselet_id = hits[hit_idx].poselet_id();
				if (pav[poselet_id]>hits.size() || hits[pav[poselet_id]].score()<hits[hit_idx].score())
					pav[poselet_id]=hit_idx;
			}
		}
		phv.clear();
		for (size_t i=0; i<pav.size(); ++i)
		  if (int(pav[i])!=-1)
				phv.push_back(hits[pav[i]]);
		obj_hits[c].init(phv,_model);
	}
}

