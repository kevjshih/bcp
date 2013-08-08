
//#define CHAR16_T unsigned short
//#include <yvals.h>
#include "mex.h"
#include "poselet_api.h"
#include <boost/gil/gil_all.hpp>
#include <boost/bind.hpp>
#include <vector>

using namespace std;
using namespace boost::gil;
using namespace poselet_api;

void doError() {
	 mexErrMsgTxt(
		 "success = poselets_mex(init_file)\n"
		 "  classifier_file: path to an .xml file defining the poselets model\n"
		 "  success: a boolean\n"
		 "  This signature must be called before using the poselets\n"
		 "\n"
		 "\n"
		 "Signature for doing image analysis:\n"
		 " object_hits                            = poselets_mex(img)\n"
		 "[object_hits, poselet_hits]             = poselets_mex(img)\n"
		 "[poselet_hits, features]                = poselets_mex(img, max_hits_per_poselet)\n"
		 "[object_hits, attributes]               = poselets_mex(img, 0)\n"
		 "[object_hits, attributes, poselet_hits] = poselets_mex(img, 0)\n"
		 "\n"
		 "  img: H x W x 3 uint8 array representing an RGB image or a H x W uint8 array representing a Gray image\n"
		 "\n"
		 "  max_hits_per_poselet: non-negative integer used when the poselet features are needed\n"
		 "     When positive, limits the number of poselet hits of a given poselet type to \n"
		 "     the top max_hits_per_poselet ones with highest scores\n"
		 "\n"
		 "  object_hits: N x 6 double array, where N is the number of objects found\n"
		 "    The columns of the hits array are as follows:\n"
		 "      x0, y0, width, height, confidence, detectorID\n"
		 "    The first four columns define a bounding box for the object\n"
		 "    The confidence is a positive scalar proportional to how likely the object is there\n"
		 "    DetectorID is id of the object detector that reported, if multiple ones are used\n"
		 "\n"
		 "  poselet_hits: M x 7 double array, where M is the number of found poselets\n"
		 "    The columns of the poselet_hits array are as follows:\n"
		 "      x0, y0, width, height, confidence, objectID, poseletID\n"
		 "    The first five are the same as in object_hits\n"
		 "    objectID is index of the object for which the poselet fired, 1..N, in the order of object_hits\n"
		 "	  poseletID is the poselet type\n"
		 "\n"
		 "  features: a list of feature vectors corresponding to the poselet_hits \n"
		 "\n"
		 "  attributes: N x A array of doubles indicating the probability of each of A attributes for each object\n"
	 );
}

bool mxArray2Image(const mxArray* imgArray, Image& imageProxy) {
	// Only 8-bit RGB or 8-bit grayscale image are supported
	if ((mxGetNumberOfDimensions(imgArray)!=2 && (mxGetNumberOfDimensions(imgArray)!=3 || mxGetDimensions(imgArray)[2]!=3))
		|| mxGetClassID(imgArray)!=mxUINT8_CLASS) {
			mexErrMsgTxt("The image must be RGB or Gray image of channel type uint8");
			return false;
	}

	Image::DepthType depth = Image::k8Bit;
	size_t bytesPerChannel = 1;

	const mwSize* dim = mxGetDimensions(imgArray);
    size_t width = size_t(dim[1]);
    size_t height = size_t(dim[0]);
	const unsigned char* src = (const unsigned char*) mxGetData(imgArray);
	
	if (mxGetNumberOfDimensions(imgArray)==2) {
		// Grayscale
		static gray8_image_t gGrayImage;
	    size_t bytesPerRow = width * bytesPerChannel;
		
		// transpose it
		gGrayImage.recreate(width,height);
		gray8c_view_t ps = interleaved_view<gray8c_ptr_t>(height,width,gray8c_ptr_t(src),height*bytesPerChannel);
		copy_pixels(transposed_view(ps), view(gGrayImage));

		imageProxy = Image(width,height,bytesPerRow,depth, Image::kGray, interleaved_view_get_raw_data(view(gGrayImage)));
	} else {
		// RGB
		static rgb8_image_t gRGBImage;
		size_t nc = size_t(dim[2]);
		size_t bytesPerRow = nc * width * bytesPerChannel;
		
		const size_t c1 = width * height;
		gRGBImage.recreate(width,height);

		// transpose and interleave the image
		rgb8c_planar_view_t ps = planar_rgb_view(height,width,src,src+c1,src+2*c1,height*bytesPerChannel);
		copy_pixels(transposed_view(ps), view(gRGBImage));

		imageProxy = Image(width,height,bytesPerRow,depth, Image::kRGB, interleaved_view_get_raw_data(view(gRGBImage)));
	}

	return true;
}

template <typename Value>
void append_hit(const Value& oh, vector<Value>& hits) {
	hits.push_back(oh);
}

inline void append_hit_and_feature(const PoseletHit& oh, vector<PoseletHit>& hits, vector<vector<float> >& features) {
	hits.push_back(oh);
	features.push_back(vector<float>(oh.feature,oh.feature+oh.featureSize));
}

mxArray* object_hits2mx(const vector<ObjectHit>& hits) {
	int cols = 6;
	int rows = int(hits.size());
    mwSize dims[2];
    dims[0] = rows;
    dims[1] = cols;
    mxArray* hitArray = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double* p = (double*) mxGetData(hitArray);
    for (size_t i = 0; i < dims[0]; ++i) {
        const ObjectHit& hit = hits[i];
        p[0] = (int) hit.x0;
        p[rows] = (int) hit.y0;
        p[2*rows] = (int) hit.width;
        p[3*rows] = (int) hit.height;
        p[4*rows] = hit.score;
        p[5*rows] = hit.category;
        ++p;
    }
	return hitArray;
}

mxArray* poselet_hits2mx(const vector<PoseletHit>& hits) {
	int cols = 7;
	int rows = int(hits.size());
    mwSize dims[2];
    dims[0] = rows;
    dims[1] = cols;
    mxArray* hitArray = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double* p = (double*) mxGetData(hitArray);
    for (size_t i = 0; i < dims[0]; ++i) {
        const PoseletHit& hit = hits[i];
        p[0] = (int) hit.x0;
        p[rows] = (int) hit.y0;
        p[2*rows] = (int) hit.width;
        p[3*rows] = (int) hit.height;
        p[4*rows] = hit.score;
        p[5*rows] = hit.clusterID;
        p[6*rows] = hit.poseletID;
        ++p;
    }
	return hitArray;
}

//#pragma optimize("",off)
mxArray* features2mx(const vector<vector<float> >& features) {
    mxArray* featureArray = mxCreateCellMatrix(int(features.size()), 1);
	for (size_t i = 0; i < features.size(); ++i) {
	    mwSize dims[1];
		dims[0]= features[i].size();
		mxArray* fArray = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
		double* p = (double*) mxGetData(fArray);
		for (size_t j=0; j<features[i].size(); ++j)
			p[j]=features[i][j];
		mxSetCell(featureArray,i,fArray);
	}
	return featureArray;
}

/*
mxArray* poselet_hits2features(const vector<PoseletHit>& hits) {
    mxArray* featureArray = mxCreateCellMatrix(int(hits.size()), 1);
	for (size_t i = 0; i < hits.size(); ++i) {
	    mwSize dims[1];
		dims[0]= hits[i].featureSize;
		mxArray* fArray = mxCreateNumericArray(1, dims, mxDOUBLE_CLASS, mxREAL);
		double* p = (double*) mxGetData(fArray);
		for (size_t j=0; j<hits[i].featureSize; ++j)
			p[j]=hits[i].feature[j];
		mxSetCell(featureArray,i,fArray);
	}
	return featureArray;
}
*/

mxArray* object_hits2attributes(const vector<ObjectHit>& hits, int numAttributes) {
	int cols = numAttributes;
	int rows = int(hits.size());
    mwSize dims[2];
    dims[0] = rows;
    dims[1] = cols;
    mxArray* hitArray = mxCreateNumericArray(2, dims, mxDOUBLE_CLASS, mxREAL);
    double* p = (double*) mxGetData(hitArray);
    for (size_t i = 0; i < dims[0]; ++i) {
		assert(hits[i].attributes);
		for (int j=0; j<numAttributes; ++j)
			p[j*rows] = hits[i].attributes[j];
        ++p;
    }
	return hitArray;
}


void mexFunction(int nlhs, mxArray *plhs[],int nrhs, const mxArray *prhs[]) {
	if (nlhs==0 || nrhs==0 || nrhs>2) {
		doError();
		return;
	}
	if (nrhs==1 && nlhs<=1 && mxGetClassID(prhs[0])==mxCHAR_CLASS) {
		// init call
		// success = poselets_mex(init_file)
		const int MAX_BUF=255;
		char buffer[MAX_BUF];
		int err=-1;
		int success=!mxGetString(prhs[0], buffer, MAX_BUF);
		if (!success) {
			mexErrMsgTxt("Failed to read init_file");
		} else if ((err=poselet_api::InitDetector(buffer))!=0) {
			mexErrMsgTxt("Failed to initialize the poselets module");
		}

		if (nlhs==1)
			plhs[0] = mxCreateLogicalScalar(err==kNoErr);
		return;
	}

	if (!poselet_api::IsInitialized()) {
		mexErrMsgTxt("You must first initialize the poselets module");
		doError();
		return;
	}
	Image imageProxy;
	if (!mxArray2Image(prhs[0],imageProxy))
		return;

	if (nrhs==1) {
		if (nlhs==1) {
			// object_hits = poselets_mex(img);
			vector<ObjectHit> object_hits;
			RunDetector(imageProxy, NULL, boost::bind(append_hit<ObjectHit>,_1,boost::ref(object_hits)), true, 0, false);
			plhs[0] = object_hits2mx(object_hits);
		} else {
			// [object_hits, poselet_hits] = poselets_mex(img);
			mxAssert(nlhs==2, "General failure");
			vector<ObjectHit> object_hits;
			vector<PoseletHit> poselet_hits;
			RunDetector(imageProxy, boost::bind(append_hit<PoseletHit>,_1,boost::ref(poselet_hits)), boost::bind(append_hit<ObjectHit>,_1,boost::ref(object_hits)), true, 0, false);
			plhs[0] = object_hits2mx(object_hits);
			plhs[1] = poselet_hits2mx(poselet_hits);
		}
	} else {
		if (!mxIsNumeric(prhs[1])) {
			mexErrMsgTxt("Second argument must be a non-negative integer");
			doError(); return;
		}
		double val=mxGetScalar(prhs[1]);
		if (val<0 || val!=int(val)) {
			mexErrMsgTxt("Second argument must be a non-negative integer");
			doError(); return;
		}
		if (val==0) {
			if (nlhs!=2 && nlhs!=3) {
				doError(); return; 
			}
			if (NumAttributes()==0) {
				mexErrMsgTxt("Attributes are not specified in the initialization file");
				return;
			}
			if (nlhs==2) {
				// [object_hits, attributes]               = poselets_mex(img, 0)
				vector<ObjectHit> object_hits;
				RunDetector(imageProxy, NULL, boost::bind(append_hit<ObjectHit>,_1,boost::ref(object_hits)), true, 0, true);
				plhs[0] = object_hits2mx(object_hits);
				plhs[1] = object_hits2attributes(object_hits, NumAttributes());
			} else {
				// [object_hits, attributes, poselet_hits] = poselets_mex(img, 0)
				vector<ObjectHit> object_hits;
				vector<PoseletHit> poselet_hits;
				RunDetector(imageProxy, boost::bind(append_hit<PoseletHit>,_1,boost::ref(poselet_hits)), boost::bind(append_hit<ObjectHit>,_1,boost::ref(object_hits)), true, 0, true);
				plhs[0] = object_hits2mx(object_hits);
				plhs[1] = object_hits2attributes(object_hits, NumAttributes());
				plhs[2] = poselet_hits2mx(poselet_hits);
			}
		} else {
			if (nlhs!=2) {
				doError(); return; 
			}
			// [poselet_hits, features] = poselets_mex(img, max_hits_per_poselet)
			int max_hits_per_poselet=int(val);
			vector<PoseletHit> poselet_hits;
			vector<vector<float> > features;
			RunDetector(imageProxy, boost::bind(append_hit_and_feature,_1,boost::ref(poselet_hits), boost::ref(features)), NULL, true, max_hits_per_poselet, false);
			plhs[0] = poselet_hits2mx(poselet_hits);
			plhs[1] = features2mx(features);
		}
	}
}
