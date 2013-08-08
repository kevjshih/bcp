#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) 
{
   // get_best_part_spat(regions, boxes, scores, spat_weights, scale_weights, ...)
   // Equal spacing for spatial weights, determined from 
   // 
   int y,x,s; // for indexing
   const double INF = mxGetInf();

   double reg_ov_l =  0.;
   double reg_ov_u = INF;
   double box_ov_l = 0.;
   double box_ov_u = INF;
   double iou_l = 0.5;
   double iou_u = INF;

   // Setup variables
   switch(nrhs)
   {
      case 4:
         mexErrMsgTxt("Incorrect number of arguments");
         break;
      case 5: // Correct arguments, but nothing to do
         break;
      case 11:
         if(mxGetNumberOfElements(prhs[10])==1) {
            iou_u = mxGetScalar(prhs[10]);
         }
      case 10:
         if(mxGetNumberOfElements(prhs[9])==1) {
            iou_l = mxGetScalar(prhs[9]);
         }
      case 9:
         if(mxGetNumberOfElements(prhs[8])==1) {
            box_ov_u = mxGetScalar(prhs[8]);
         }
      case 8:
         if(mxGetNumberOfElements(prhs[7])==1) {
            box_ov_l = mxGetScalar(prhs[7]);
         }
      case 7:
         if(mxGetNumberOfElements(prhs[6])==1) {
            reg_ov_u = mxGetScalar(prhs[6]);
         }
      case 6:
         if(mxGetNumberOfElements(prhs[5])==1) {
            reg_ov_l = mxGetScalar(prhs[5]);
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

   double *spat_weights = mxGetPr(prhs[3]);
   double *scale_weights = mxGetPr(prhs[4]);

   int spat_h = mxGetM(prhs[3]);
   int spat_w = mxGetN(prhs[3]);

   // Setup output
   // index of chosen box for each window
   mxArray *output_score = mxCreateNumericMatrix(NA, 1, mxDOUBLE_CLASS, mxREAL);
   double *score = mxGetPr(output_score);

   // Allocate (ind, x, y, scale)
   mxArray *output_ind = mxCreateNumericMatrix(NA, 4, mxDOUBLE_CLASS, mxREAL);
   double *ind = mxGetPr(output_ind);

   // Precompute areas of B
   double *Aarea = (double *)mxMalloc(sizeof(double)*NA);
   double *Barea = (double *)mxMalloc(sizeof(double)*NB);

   //[0*NB NB 2*NB 3*NB]
   for(int i=0; i<NA; i++)
      Aarea[i] = (A[i+2*NA]-A[i]+1)*(A[i+3*NA]-A[i+NA]+1);

   for(int j=0; j<NB; j++) 
      Barea[j] = (B[j+2*NB]-B[j]+1)*(B[j+3*NB]-B[j+NB]+1);

   int spat_s = 3;
   // Setup spatial stuff
   double *ybins = (double *)mxMalloc(sizeof(double)*spat_h); // Indicates upper bound on each bin
   double *xbins = (double *)mxMalloc(sizeof(double)*spat_w);
   double *sbins = (double *)mxMalloc(sizeof(double)*spat_s);

   ybins[spat_h-1] = INF;

   xbins[spat_w-1] = INF;
   sbins[spat_s-1] = INF;

   // Compute bound on spatial score
   double max_spat = -1*INF;
   for(int i=0; i<spat_w*spat_h; i++)
      max_spat = max(spat_weights[i], max_spat);

   double max_scale = -1*INF;
   for(int s=0; s<spat_s; s++)
      max_scale = max(scale_weights[s], max_scale);

   max_spat = max_scale + max_spat;

   // Outer loop over regions
   for(int i=0; i<NA; i++) {
      double max_score = -1*INF;
      int max_ind[4] = {-2,-2,-2,-2};
      bool skip = false;
      if(iou_l>0 && NB>0) {
         double ratio = Barea[0]/Aarea[i];
         skip = (ratio<iou_l || ratio>1/iou_l);
      }

      // Compute quantized bins
      double bin_h = (A[i+3*NA]-A[i+NA]+1)/(spat_h);
      for(y=0; y<spat_h-1; y++) {
         ybins[y] = A[i+NA] + bin_h*(y+1);
      }
      double bin_w = (A[i+2*NA]-A[i]+1)/(spat_w);
      for(x=0; x<spat_w-1; x++) {
         xbins[x] = A[i] + bin_w*(x+1);
      }
//      printf("Spatial bins:");
      // Hard coding spatial bins for now
      for(s=0; s<spat_s-1; s++) { 
         sbins[s] = Aarea[i]/pow(2., 2.*(spat_s-s-1)); // [1/2^2 1/2^1]^2, [2 1]
//         printf("%f %f/(%f), ", sbins[s], Aarea[i], pow(2.,2.*(spat_s-s-1)));
      }

      for(int j=0; j<NB && !skip; j++) {
         //printf("%d %d\n", i, j);
         if((B_score[j]+max_spat)>max_score) { // Only consider a detection if its upper bound exceeds current max score
         //if(B_score[j]>0) {
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

            // Now check which bin it's in (we'll return this as well)
            double ycenter = 1./2*(B[i+3*NB]+B[i+NB]);
            double xcenter = 1./2*(B[i+2*NB]+B[i]);
            double scale = Barea[j];

            for(y=0; y<spat_h; y++) {
               if(ycenter<=ybins[y])
                  break;
            }
               
            for(x=0; x<spat_w; x++) {
               if(xcenter<=xbins[x])
                  break;
            }

            for(s=0; s<spat_w; s++) {
               if(scale<=sbins[s])
                  break;
            }
            if(x==spat_w || y==spat_h || s==spat_s) // This means there was a nan or something
               continue;

            double true_score = B_score[j] + spat_weights[y+spat_h*x] + scale_weights[s];
            if(true_score > max_score) {
            // Made it through, save score
               max_score = true_score; 
               max_ind[0] = j;
               max_ind[1] = x;
               max_ind[2] = y;
               max_ind[3] = s;
            }
         }
      }
      // We're done, record the best index and score
      for(int p=0; p<4; p++)
         ind[i + p*NA] = (double)max_ind[p] + 1; // +1 for 1-indexing
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
