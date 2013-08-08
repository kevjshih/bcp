#include "poselet_hit.h"
#include "xml_utils.h"
#include <boost/lexical_cast.hpp>

using namespace std;

void hit::init(rapidxml::xml_node<>* node) {
	_score = boost::lexical_cast<double>(get_xml_value(node->first_attribute("score")));

	stringstream str(get_xml_value(node->first_attribute("bounds")));
	str >> _bounds;
}

void poselet_hit::init(rapidxml::xml_node<>* node) {
	hit::init(node);
	_poselet_id = boost::lexical_cast<int>(get_xml_value(node->first_attribute("poselet_id")));
}

ostream& operator<<(ostream& os, const hit& hit) {
	os << "   <hit score=\""<<hit.score()<<"\" bounds=\""<<hit.bounds()<<"\" />"<<endl;
	return os;
}

ostream& operator<<(ostream& os, const poselet_hit& hit) {
	os << "   <poselet_hit score=\""<<hit.score()<<"\" bounds=\""<<hit.bounds()<<"\" poselet_id=\""<<hit.poselet_id()<<"\" />"<<endl;
	return os;
}
