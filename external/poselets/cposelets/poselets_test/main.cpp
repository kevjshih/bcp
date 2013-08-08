#include <iostream>
#include <vector>
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#include <boost/gil/extension/io/jpeg_io.hpp>
#include <boost/bind.hpp>
#include <boost/timer.hpp>
#include <boost/format.hpp>
#include <boost/ref.hpp>

#include "poselet_api.h"
#include "bounds.h"
#include "gil_draw.h"

#ifdef USE_IMDEBUG  // Requires imdebug which is only available on Windows http://www.billbaxter.com/projects/imdebug/
#include "gil_imdebug.h"
#endif

#include <sys/types.h>

#include <stdio.h>
#include <fstream>
#include <sstream>

using namespace std;
using namespace boost;
using namespace gil;
using namespace poselet_api;


template <typename Value>
void append_hit(const Value& oh, vector<Value>& hits) {
  hits.push_back(oh);
}

template <typename Hit>
bool hit_less(const Hit& a, const Hit& b) {
  return a.score > b.score;
}


#ifdef USE_IMDEBUG
template <typename View, typename Hit, typename Color>
void draw_hit(const View& v, const Hit& hit, const Color& c, int thickness=1) {
  typename View::value_type color;
  boost::gil::color_convert(c,color);
  int_bounds b(hit.x0,hit.y0,hit.width,hit.height);
  draw_bounds(v,color,b,thickness);

  render_text(v, b._min+int_point(thickness,thickness), str(boost::format("%4.2f") % hit.score), c, 1, 1);
}
#endif

bool file_exists(const char *filename)
{
  ifstream ifile(filename);
  return !!ifile;
}

int main( int argc, char* argv[] ) {
  if (argc!=3) {
    cout << "Usage: "<<argv[0]<<" <detector.xml> <image.jpg>"<<endl;
    return 0;
  }
  if (!file_exists(argv[1])) {
    cout << "Cannot open model file "<<argv[1]<<endl;
    return 0;
  }
  if (!file_exists(argv[2])) {
    cout << "Cannot open image file "<<argv[2]<<endl;
    return 0;
  }

  gil::rgb8_image_t img;
  gil::jpeg_read_image(argv[2],img);

  InitDetector(argv[1]);

  boost::timer t;

  Image img_proxy(img.width(), img.height(), const_view(img).pixels().row_size(), Image::k8Bit, Image::kRGB, interleaved_view_get_raw_data(const_view(img)));

  vector<ObjectHit> object_hits;
  vector<PoseletHit> poselet_hits;

  RunDetector(img_proxy, boost::bind(append_hit<PoseletHit>,_1,boost::ref(poselet_hits)), boost::bind(append_hit<ObjectHit>,_1,boost::ref(object_hits)), true, 0, false);

  std::sort(object_hits.begin(), object_hits.end(), hit_less<ObjectHit>);

  double scoresum=0;
  for (size_t i=0; i<object_hits.size(); ++i) {
#ifdef USE_IMDEBUG
    if (i<2)
      draw_hit(view(img), object_hits[i], rgb8_pixel_t(255,0,0), 3);
#endif
    scoresum+=object_hits[i].score;
  }
#ifdef USE_IMDEBUG
  debug_display_view(view(img));
#endif

  cout << "Found "<<object_hits.size()<<" objects with scoresum "<<scoresum<<" in "<<t.elapsed()<<" sec."<<endl;
  return 0;
}
