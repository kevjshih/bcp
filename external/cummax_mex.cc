#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"

#define max(a,b) (a)>(b)?(a):(b)

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
   if(nrhs!=1)
      mexErrMsgTxt("Wrong number of inputs"); 

   if(nlhs!=1)
      mexErrMsgTxt("Wrong number of outputs"); 
    
   int N = mxGetNumberOfElements(prhs[0]);
   double *in = mxGetPr(prhs[0]);

   mxArray *mxOut = mxCreateNumericMatrix(N,1, mxDOUBLE_CLASS, mxREAL);
   double *out = mxGetPr(mxOut);

   if(N>0) {
      out[0] = in[0];
      for(int i=1; i<N; i++)
         out[i] = max(in[i], out[i-1]);
   }
   plhs[0] = mxOut;
}
