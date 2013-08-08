#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
   const double INF = mxGetInf();

   double reg_ov_l =  0.;
   double reg_ov_u = INF;
   double box_ov_l = 0.;
   double box_ov_u = INF;
   double iou_l = 0.5;
   double iou_u = INF;
   bool skip_all = true; // Can skip large chunks of boxes by assuming they're all the same size
   // Setup variables
   switch(nrhs)
   {
      case 2:
         mexErrMsgTxt("Incorrect number of arguments");
         break;
      case 3: // Correct arguments, but nothing to do
         break;
      case 10:
         if(mxGetNumberOfElements(prhs[9])==1) {
            skip_all = mxGetScalar(prhs[9]);
         }
      case 9:
         if(mxGetNumberOfElements(prhs[8])==1) {
            iou_u = mxGetScalar(prhs[8]);
         }
      case 8:
         if(mxGetNumberOfElements(prhs[7])==1) {
            iou_l = mxGetScalar(prhs[7]);
         }
      case 7:
         if(mxGetNumberOfElements(prhs[6])==1) {
            box_ov_u = mxGetScalar(prhs[6]);
         }
      case 6:
         if(mxGetNumberOfElements(prhs[5])==1) {
            box_ov_l = mxGetScalar(prhs[5]);
         }
      case 5:
         if(mxGetNumberOfElements(prhs[4])==1) {
            reg_ov_u = mxGetScalar(prhs[4]);
         }
      case 4:
         if(mxGetNumberOfElements(prhs[3])==1) {
            reg_ov_l = mxGetScalar(prhs[3]);
         }
         break;
      default:
         mexErrMsgTxt("Incorrect number of arguments");
   }

   int NA = mxGetM(prhs[0]);
   int NB = mxGetM(prhs[1]);

   double *A = mxGetPr(prhs[0]);
   double *B = mxGetPr(prhs[1]);

   double *B_score = mxGetPr(prhs[2]);

   // Setup output
   // index of chosen box for each window
   mxArray *output_score = mxCreateNumericMatrix(NA, 1, mxDOUBLE_CLASS, mxREAL);
   double *score = mxGetPr(output_score);

   mxArray *output_ind = mxCreateNumericMatrix(NA, 1, mxDOUBLE_CLASS, mxREAL);
   double *ind = mxGetPr(output_ind);

   // Precompute areas of B
   double *Aarea = (double *)mxMalloc(sizeof(double)*NA);
   double *Barea = (double *)mxMalloc(sizeof(double)*NB);

   //[0*NB NB 2*NB 3*NB]
   for(int i=0; i<NA; i++)
      Aarea[i] = (A[i+2*NA]-A[i]+1)*(A[i+3*NA]-A[i+NA]+1);

   for(int j=0; j<NB; j++) 
      Barea[j] = (B[j+2*NB]-B[j]+1)*(B[j+3*NB]-B[j+NB]+1);

   // Process it
// get_best_part(regions, boxes, scores, reg_ov_l, reg_ov_u, box_ov_l, box_ov_l, iou_l, iou_u)
   for(int i=0; i<NA; i++) {
      double max_score = -1*INF;
      int max_ind = -2;
      bool skip = false;
      if(iou_l>0 && NB>0 && skip_all) {
         double ratio = Barea[0]/Aarea[i];
         skip = (ratio<iou_l || ratio>1/iou_l);
      }

      for(int j=0; j<NB && !skip; j++) {
         //printf("%d %d\n", i, j);
         if(B_score[j]>max_score) {
         //if(B_score[j]>0) {
            // Check the overlap
            double iw = min(A[i+2*NA], B[j+2*NB]) - max(A[i], B[j])+1;

            if(iw<=0)
               continue;
      
            double ih = min(A[i+3*NA], B[j+3*NB]) - max(A[i+NA], B[j+NB])+1;
           
            if(ih<=0) // No overlap
               continue;
 
            double ia = iw*ih;

            double Bov = ia/Barea[j];
            if(Bov<box_ov_l || Bov>box_ov_u)
               continue;

            double Aov = ia/Aarea[i];
            if(Aov<reg_ov_l || Aov>reg_ov_u)
               continue;

            double ua = Aarea[i] + Barea[j] - ia;
            double iou = ia/ua;
            if(iou<iou_l || iou>iou_u)
               continue;

            // Made it through, save score
            max_score = B_score[j]; 
            max_ind = j;
         }
      }
      // We're done, record the best index and score
      ind[i] = max_ind + 1; // +1 for 1-indexing
      score[i] = max_score;
   }      

   plhs[0] = output_score;

   if(nlhs>1) {
      plhs[1] = output_ind;
   } else {
      mxDestroyArray(output_ind);
   }

   mxFree(Aarea);
   mxFree(Barea);
}
