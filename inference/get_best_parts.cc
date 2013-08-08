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
   double Ntodo = 2; // Get top two detections
   double part_supp_ov = 0.5;
   bool skip_all = false; // Can skip large chunks of boxes by assuming they're all the same size
   // Setup variables
   switch(nrhs)
   {
      case 2:
         mexErrMsgTxt("Incorrect number of arguments");
         break;
      case 3: // Correct arguments, but nothing to do
         break;
      case 12:
         if(mxGetNumberOfElements(prhs[11])==1) {
            skip_all = mxGetScalar(prhs[11]);
         }
      case 11:
         if(mxGetNumberOfElements(prhs[10])==1) {
            part_supp_ov = mxGetScalar(prhs[10]);
         }
      case 10:
         if(mxGetNumberOfElements(prhs[9])==1) {
            Ntodo = mxGetScalar(prhs[9]);
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
   mxArray *output_score = mxCreateNumericMatrix(NA, (int)Ntodo, mxDOUBLE_CLASS, mxREAL);
   double *score = mxGetPr(output_score);

   mxArray *output_ind = mxCreateNumericMatrix(NA, (int)Ntodo, mxDOUBLE_CLASS, mxREAL);
   double *ind = mxGetPr(output_ind);

   // Precompute areas of B
   double *Aarea = (double *)mxMalloc(sizeof(double)*NA);
   double *Barea = (double *)mxMalloc(sizeof(double)*NB);

   int *relevant = (int *)mxMalloc(sizeof(int)*NB);
   bool *suppressed = (bool *)mxMalloc(sizeof(bool)*NB);

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
      int n_ok = 0;
      if(iou_l>0 && NB>0 && skip_all) {
         double ratio = Barea[0]/Aarea[i];
         skip = (ratio<iou_l || ratio>1/iou_l);
      }

      // Check to see if a part is relevant
      for(int j=0; j<NB && !skip; j++) {
         // Check the overlap
         double iw = min(A[i+2*NA], B[j+2*NB]) - max(A[i], B[j])+1;
         double ih = min(A[i+3*NA], B[j+3*NB]) - max(A[i+NA], B[j+NB])+1;
         
         if(iw<=0 || ih<=0) // No overlap
            continue;
 
         double ia = iw*ih;

         double ua = Aarea[i] + Barea[j] - ia;
         double iou = ia/ua;
         if(iou<iou_l || iou>iou_u)
            continue;

         double Aov = ia/Aarea[i];
         if(Aov<reg_ov_l || Aov>reg_ov_u)
            continue;

         double Bov = ia/Barea[j];
         if(Bov<box_ov_l || Bov>box_ov_u)
            continue;


         // Made it through, use this example
         relevant[n_ok] = j; //max_score = B_score[j]; 
         // reset suppressed
         suppressed[j] = false; // Not resetting everything for efficiency
         n_ok++;
      }

      // Now find the N highest scoring parts (ideally we'd sort, but we'll have to settle for N^2, because I'm lazy)
      int p; // We're going to need this to finish out the loop ...
      for(p=0; p<Ntodo; p++) {
         double max_score = -1*INF;
         int max_ind = -2;

         for(int j=0; j<n_ok; j++) {
            int ind_t = relevant[j];
            if(!suppressed[ind_t] && max_score < B_score[ind_t]) { // Found a new top score!
               max_score = B_score[ind_t];
               max_ind = ind_t;
            }
         }

         // Now record the max
         ind[i + p*NA] = max_ind+1; // +1 for 1-indexing
         score[i + p*NA] = max_score;

         // if nothing was found, break and continue on with the next part!
         if(max_ind==-2 || p==(Ntodo-1)) // Also skip the suppression on the last iteration
            break;

         // Finally, suppress any other boxes that overlap with that one
         for(int j=0; j<n_ok; j++) {
            int ind_t = relevant[j];

            if(!suppressed[ind_t]) {
               double iw = min(B[max_ind+2*NB], B[ind_t+2*NB]) - max(B[max_ind], B[ind_t])+1;
               double ih = min(B[max_ind+3*NB], B[ind_t+3*NB]) - max(B[max_ind+NB], B[ind_t+NB])+1;
               
               if(iw<=0 || ih<=0) // No overlap
                  continue;
            
               double ia = iw*ih;
               double ua = Barea[max_ind] + Barea[ind_t] - ia;
               double iou = ia/ua;
               
               suppressed[ind_t] = iou>=part_supp_ov; // This variable is superfluous, could actually update the relevant structure!
            }
         }
      }

      // We're done, record -1 for anything that got skipped over
      for(int p2=p+1; p2<Ntodo; p2++) { 
         ind[i + p2*NA] = -1; // +1 for 1-indexing
         score[i + p2*NA] = -1*INF;
      }
   }      

   plhs[0] = output_score;

   if(nlhs>1) {
      plhs[1] = output_ind;
   } else {
      mxDestroyArray(output_ind);
   }

   mxFree(Aarea);
   mxFree(Barea);
   mxFree(suppressed);
   mxFree(relevant);
}
