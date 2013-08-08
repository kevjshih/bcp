#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// Input: A, B, [quick_th]
// output: ov

   if(nrhs<2 || nrhs>3)
      mexErrMsgTxt("Wrong number of inputs"); 
   
   if(nlhs!=1)
      mexErrMsgTxt("Wrong number of outputs"); 

   if(mxGetN(prhs[0])!=4 || mxGetN(prhs[1])!=4)
      mexErrMsgTxt("Boxes have 4 parameters");

   double th = 0;

   if(nrhs>=3)
      th = mxGetScalar(prhs[2]);

   int NA = mxGetM(prhs[0]);
   int NB = mxGetM(prhs[1]);

   double *A = mxGetPr(prhs[0]);
   double *B = mxGetPr(prhs[1]);

   mxArray *ov_out = mxCreateNumericMatrix(NA, NB, mxDOUBLE_CLASS, mxREAL);
   plhs[0] = ov_out;

   if(NA==0 || NB==0) // Empty array, don't do any computation
      return;

   double *ov = mxGetPr(ov_out);

   // Precompute areas of B
   double *Aarea = (double *)mxMalloc(sizeof(double)*NA);

   //[0*NB NB 2*NB 3*NB]
   for(int i=0; i<NA; i++)
      Aarea[i] = (A[i+2*NA]-A[i]+1)*(A[i+3*NA]-A[i+NA]+1);

   for(int j=0; j<NB; j++) {
      double Barea = (B[j+2*NB]-B[j]+1)*(B[j+3*NB]-B[j+NB]+1);

      if(th>0) {
         double ratio = Barea/Aarea[0];
         if(ratio<th || ratio>1/th)
            continue;
      }
      for(int i=0; i<NA; i++) {
         double iw = min(A[i+2*NA], B[j+2*NB]) - max(A[i], B[j])+1;
         double ih = min(A[i+3*NA], B[j+3*NB]) - max(A[i+NA], B[j+NB])+1;
   
         if(iw<0 || ih<0) {
            ov[i + NA*j] = 0;
         } else {
            double ia = iw*ih;
            double ua = Aarea[i] + Barea - ia;
            ov[i + NA*j] = ia/ua;
         }
      }
   }


   mxFree(Aarea);
}
