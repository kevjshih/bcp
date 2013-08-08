#pragma once

#include <boost/function.hpp>
#include <iostream>

#ifdef _MSC_VER
	#ifdef POSELET_EXPORTS
		#define POSELET_API __declspec(dllexport)
	#else
		#ifdef POSELET_IMPORTS
			#define POSELET_API __declspec(dllimport)
		#else
			#define POSELET_API
		#endif
	#endif
#else
	#define POSELET_API
#endif

namespace poselet_api {

const double VERSION=1.0;

///////////////////////////////////////////////////
//////
//////		Structs and Enums
//////
///////////////////////////////////////////////////

// Error type as a return value of API calls
typedef enum { 
	kNoErr=0,
	kInternalErr,			// internal error
	kParamErr,				// incorrect input parameter
	kOutOfMemoryErr			// out of memory error
} ErrorType;

// Represents the image to be processed. All memory is owned by the client
struct Image {
	typedef enum { kGray, kRGB, kBGR, kRGBA, kARGB, kCMYK, kNUM_CS_TYPES } ColorSpaceType;
	typedef enum { k8Bit, k16Bit, k32Bit, kNUM_DEPTH_TYPES } DepthType;

	// Interleaved constructor
	Image(size_t width, size_t height, size_t rowBytes, DepthType depth,
				ColorSpaceType cs, const void* pixels);

	// Planar RGB constructor
	Image(size_t width, size_t height, size_t rowBytes, DepthType depth, 
				const void* red, const void* green, const void* blue);

	// Planar CMYK constructor	- CMYK not yet supported
	Image(size_t width, size_t height, size_t rowBytes, DepthType depth,
				const void* cyan, const void* magenta, const void* yellow, const void* black);

//private:
//	friend class ImageProxy;
	Image() {}
	size_t _width, _height, _rowBytes;
	DepthType _depth;
	ColorSpaceType _cs;
	bool _isPlanar;
	const void* _planarPtrs[5];
};

// Represents a poselet hit
struct PoseletHit {
	size_t x0,y0,width,height;	// bounding box of the hit. Note that it may be partially outside the image bounds!
	double score;				// measure of the confidence of the hit
	int poseletID;

	// unique ID of the cluster from which the poselet fires
	int clusterID;

	// optional pointer to the feature
	const float* feature;
	size_t featureSize;
};

// Represents a location in the image where an object is found
struct ObjectHit {
	int x0,y0,width,height;	// bounding box of the hit. Note that it may be partially outside the image bounds!
	double score;				// measure of the confidence of the hit
	int clusterID;				// id of the object. One per ObjectHit
	int category;				// visual category (person, dog, etc)

	const double* attributes;	// optional attribute values
};

// Called by the object detector when a poselet is found
typedef boost::function<void (const PoseletHit& poseletHit)> PoseletHitCB;
typedef boost::function<void (const ObjectHit& objHit     )> ObjectHitCB;


///////////////////////////////////////////////////
//////
//////		Object detector API
//////
//////		Calls are guaranteed to never throw exceptions
//////		
//////
///////////////////////////////////////////////////


// Specify the location of the .xml file defining the classifier
// Must be the first API call and must be called once only.
extern "C" ErrorType POSELET_API InitDetector(const char* configFileName);

extern "C" bool POSELET_API IsInitialized();
extern "C" int  POSELET_API NumAttributes();

extern "C" ErrorType POSELET_API RunDetector(const Image& img, PoseletHitCB poseletHitCB, ObjectHitCB objectHitCB,
													bool useBigQ,			// enable bigQ step if available
													int maxFeatures,		// when >0 number of top poselets to report along with their features. 
													bool extractAttributes	// whether to compute attributes
													);




// Ignore this

inline Image::Image(size_t width, size_t height, size_t rowBytes, DepthType depth, ColorSpaceType cs, const void* pixels)
: _width(width), _height(height), _rowBytes(rowBytes), _depth(depth), _cs(cs), _isPlanar(false) { _planarPtrs[0]=pixels; }

inline Image::Image(size_t width, size_t height, size_t rowBytes, DepthType depth, 
					const void* red, const void* green, const void* blue)
: _width(width), _height(height), _rowBytes(rowBytes), _depth(depth), _cs(kRGB), _isPlanar(true) {
	_planarPtrs[0]=red; _planarPtrs[1]=green; _planarPtrs[2]=blue;
}

inline Image::Image(size_t width, size_t height, size_t rowBytes, DepthType depth, 
				const void* cyan, const void* magenta, const void* yellow, const void* black) 
: _width(width), _height(height), _rowBytes(rowBytes), _depth(depth), _cs(kCMYK), _isPlanar(true) {
	_planarPtrs[0]=cyan; _planarPtrs[1]=magenta; _planarPtrs[2]=yellow; _planarPtrs[3]=black;
}

}		// end of namespace
