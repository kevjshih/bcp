#include "model.h"
#include "xml_utils.h"
#include <fstream>
#include <boost/lexical_cast.hpp>

using namespace std;







void poselets_model::init(const char* model_file) {
	_cluster_thresh = 5; // if not specified by model

	ifstream is(model_file);
	if (!is || is.bad())
		throw std::ios_base::failure("Cannot open model file");

	string line;
	getline(is,line,is.widen('\255'));
	using namespace rapidxml;
	xml_document<> doc;
	doc.parse<parse_non_destructive>(const_cast<char*>(line.c_str()));

	xml_node<> *root = doc.first_node("model");
	assert(root);

	int num_keypoints = boost::lexical_cast<int>(get_xml_value(root->first_attribute("num_keypoints")));
	int num_poselets = boost::lexical_cast<int>(get_xml_value(root->first_attribute("num_poselets")));
	_poselets.reserve(num_poselets);

	_wts.reserve(num_poselets);
	{
	stringstream str(get_xml_value(root->first_node("wts")));
	float w;
	for (int i=0; i<num_poselets; ++i) {
		str>>w;
		assert(!str.eof() && !str.bad());
		_wts.push_back(w);
	}
	}

	xml_node<> *node = root->first_node("poselets")->first_node("poselet");
	assert(node);
	do {
		_poselets.push_back(poselet(node,num_poselets,num_keypoints));
	} while ((node=node->next_sibling("poselet")));
	assert(_poselets.size()==size_t(num_poselets));

	for (size_t i=0; i<_poselets.size(); ++i) {
		int_point dims(_poselets[i].dims().x/config::get().pix_per_bin()-1, _poselets[i].dims().y/config::get().pix_per_bin()-1);
		size_t j=0;
		while (j<_dims_groups.size() && (_dims_groups[j]._dims!=dims)) {
			++j;
		}
		if (j==_dims_groups.size()) {
			_dims_groups.push_back(dims_group());
			_dims_groups.back()._dims=dims;
		}
		_dims_groups[j]._poselet_ids.push_back(i);
	}
	// to match Matlab
	if (_dims_groups.size()==4) {
		swap(_dims_groups[0],_dims_groups[2]);
		swap(_dims_groups[1],_dims_groups[3]);
	}
}
