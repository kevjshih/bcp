#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"
#include <list>


#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


bool check_overlap(double *a, double *b, int step)
{
   double cx = 1./2*(a[0*step]+a[2*step]);
   double cy = 1./2*(a[1*step]+a[3*step]);

   bool xin = b[0*step]<=cx && cx<=b[2*step];
   bool yin = b[1*step]<=cy && cy<=b[3*step];

   return xin && yin;
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// Input: boxes
// output: picked

   if(nrhs!=1)
      mexErrMsgTxt("Wrong number of inputs"); 
   
   if(nlhs!=1)
      mexErrMsgTxt("Wrong number of outputs"); 

   if(mxGetN(prhs[0])<5)
      mexErrMsgTxt("Boxes need at least 5 dimensions");

   const int Nbox = mxGetM(prhs[0]);
   const int end = mxGetN(prhs[0]);

   double *boxes = mxGetPr(prhs[0]);

   mxArray *scores = mxCreateNumericMatrix(Nbox, 1, mxDOUBLE_CLASS, mxREAL);
   double *ptr = mxGetPr(scores);
   
   // Copy over scores to be sorted
   for(int i=0; i<Nbox; i++)
      ptr[i] = -1.*boxes[(end-1)*Nbox + i];
      
   // Lazy and using matlab's sort function (not sure what it's based on)
   mxArray *input[1];
   input[0] = scores;
   
   mxArray *output[2];
   mexCallMATLAB(2, output, 1, input, "sort");

   double *inds = mxGetPr(output[1]); // These will be 1 indexed

   // Load up queue here
   std::list<int> remaining;
   std::list<int> picked;
   for(int i=0; i<Nbox; i++)
      remaining.push_back((int)(inds[i]-1));

   int rem = Nbox;
   std::list<int>::iterator it;
   while(!remaining.empty()) {
      int best = remaining.front();
      remaining.pop_front();
      picked.push_back(best);

      it = remaining.begin();
      while(it!=remaining.end()) { // Iterate over queue
         int cur = *it;
         // Check overlap
         bool in = check_overlap(boxes+best, boxes+cur, Nbox);

         if(in) {// Delete from queue
            it = remaining.erase(it);
         } else { // Advance queue
            ++it;
         }
      }
   }

   mxArray *picked_out = mxCreateNumericMatrix(picked.size(), 1, mxDOUBLE_CLASS, mxREAL);
   double *picked_pr = mxGetPr(picked_out);
   
   it = picked.begin();
   for(int i=0; i<picked.size(); ++i, ++it)
      picked_pr[i] = (double(*it + 1));

   plhs[0] = picked_out;   
}

