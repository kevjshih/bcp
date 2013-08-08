#include <signal.h>
#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"
#include "blas.h"

#define IN_CACHE 2
#define WAIT 20

#define max(a,b) (a)>(b)?(a):(b)
#define min(a,b) (a)<(b)?(a):(b)
//inline double square(double x){return x*x;}
#define square(a) ((a)*(a))

inline double dot_productb(double *w, double *vec, int N)
{ 
      ptrdiff_t one=1, Ns=N, st=1; 
      return ddot(&Ns, w, &one, vec, &st);
}

// These should all be replaced with blas functions!!!
inline double dot_product1(double *w, double *vec, int N)
{  
   double accum = 0;
   for(int i=0;i<N;i++)
   {
      accum += w[i]*vec[i];
   }
   return accum;
}

double dot_product(double *w, double *vec, int step, int N)
{  
   double accum = 0;
   for(int i=0;i<N;i++)
   {
      accum += w[i]*vec[i*step];
   }
   return accum;
}

double dot_product_w(double *w, double *vec, double *Q, int step, int N)
{  
   double accum = 0;
   for(int i=0;i<N;i++)
   {
      accum += w[i]*vec[i*step]*Q[i];
   }
   return accum;
}


void compute_delta(double *gt, double *hyps, double Nw, double *feat_delta)
{
   int i;
   for(i = 0; i<Nw; i++) {
      feat_delta[i] = gt[i] - hyps[i];
   }
}


void update_w(double *w, double alpha_delta, int Nw, double *feat_delta)
{
   int i;
   for(i = 0; i<Nw; i++) {
      w[i] += alpha_delta*feat_delta[i];
   }
}


double dual_objective_l1(double *alphas, double *w, double *labels, double *feat, double *reg, int Nw, int N)
{
   double obj = 0;

   // obj = -1/2*w*Q*w
   for(int i=0; i<Nw+1; i++)
      obj += square(w[i])*reg[i];
   
   obj = -obj/2;

   // obj += sum(alphas)
   for(int i=0; i<N; i++)
      obj += alphas[i];

   return obj;
}

// OK - added sparse within cell indexing, and sparse dot product
double objective_l1(double *w, double *labels, double *feat, double* C, double *reg, int Nw, int N)
{

   double obj = 0;
   for(int i = 0; i<Nw+1; i++) {
      obj += square(w[i])*(reg[i]);
   }
   obj = obj/2;

   for(int i = 0; i<N; i++)
   {   
      double score = w[Nw]; // Start with bias
      score += dot_productb(w, feat+Nw*i, Nw);

      score = 1 - score*labels[i];

      if(score > 0)
         obj += C[i]*score;
   }

   return obj;
}

// Ok - nothing sparse here
void randperm(int *perm, int N)
{
   int i, j, t;
   for(i = 0; i<N; i++) {
      perm[i] = i;
   }
   for(i = 0; i<N; i++) {
      j = i+rand()%(N-i);
      //swap
      t = perm[i];
      perm[i] = perm[j];
      perm[j] = t;
   }

}


// Ok - cell within cell indexing, sparse norm of delta
double * allocate_etas(double *feats, double *Q, int Nw, int N)
{

   double *etas = (double *)mxMalloc(N*sizeof(double));

   for(int i=0; i<N; i++) {
      etas[i] = 0;

      for(int j=0; j<Nw; j++) {
         etas[i] += square(feats[j + Nw*i])*Q[j];
      }

      etas[i] += Q[Nw]; // For bias 
   }

   return etas;
}


// Ok - allocate sparse vector for feat_delta, compute feat_delta,  ...
void optimize_dual(double *labels, double *feat, double* C, double *reg, int N, int Nw, double *w, double *alphas)
{

   double *wl = (double *)mxMalloc((Nw+1)*sizeof(double));
   double *w_o = (double *)mxMalloc((Nw+1)*sizeof(double));

   double *Q = (double *)mxMalloc((Nw+1)*sizeof(double));
   for(int i=0; i<Nw+1; i++)
      Q[i] = 1./reg[i];

   int *counts = (int *)mxMalloc(N*sizeof(int));
   memset(counts, 0, N*sizeof(int));
   double *etas = allocate_etas(feat, Q, Nw, N);

   int *perm = (int *)mxMalloc(N*sizeof(int));

   double epsilon = 1e-8, eta1, eta2;
   double epsilon2 = 1e-3;
 
   bool terminate = 0;
   int iter = 0, i, pi;

   double eta_r, alpha_d_lab, alpha_new;
         
   double obj_p = objective_l1(w, labels, feat, C, reg, Nw, N);
   double obj_d = dual_objective_l1(alphas, w, labels, feat, reg, Nw, N);
  
   printf("Iter: %d - primal: %f, dual: %f\n", iter, obj_p, obj_d);
   int hit = 0;
   int skip = 0;
   while(!terminate) {
      iter++;

      memcpy(w_o, w, (Nw+1)*sizeof(double));
         
      randperm(perm, N);

      for(pi = 0; pi<N; pi++) {
         i = perm[pi];

         if(counts[i]>IN_CACHE) {
            --counts[i];
            ++skip;
            continue;
         }
         
         ++hit;

         eta1 = labels[i]*(dot_productb(w, feat + Nw*i, Nw) + w[Nw]) - 1;

         eta_r = eta1/etas[i];
         alpha_new = min(max(alphas[i] - eta_r, 0), C[i]);
         alpha_d_lab = labels[i]*(alpha_new - alphas[i]);
         alphas[i] = alpha_new;
        
         for(int j=0; j<Nw; ++j)
            w[j] += alpha_d_lab*feat[j+Nw*i]*Q[j];
         
         w[Nw] += alpha_d_lab*Q[Nw];

         if(alpha_new>1e-9) {
            counts[i] = 0;
         } else {
            ++counts[i];
            if(counts[i]>IN_CACHE) 
               counts[i] = WAIT;
         }
      }
      // Check for convergence
//      double delta = dot_product
//      compute_delta(w, w_o, Nw, wl);

//      double delta = sqrt(dot_product(wl, wl, 1, Nw));
//      double num = min(sqrt(dot_product(w, w, 1, Nw)), sqrt(dot_product(w_o, w_o, 1, Nw)));
   
//      if(delta/num < epsilon) {
//         printf("W update sufficiently small\n");
//         terminate = 1;
//      }
     
      if(iter>1000000)
         terminate = 1;
       
      if(iter%30 == 0 || iter<=1 || terminate) {
         obj_p = objective_l1(w, labels, feat, C, reg, Nw, N);
         obj_d = dual_objective_l1(alphas, w, labels, feat, reg, Nw, N);
 
         bool dual_gap = !terminate && (obj_p - obj_d)/obj_d < epsilon2;
         terminate = terminate || dual_gap;

         if(iter%30==0 || terminate || iter<=1) {
//         terminate = 1;
            printf("Iter: %d - primal: %f, dual: %f Touched:%f\n", iter, obj_p, obj_d, hit/((double)(hit+skip)));
         } else {
            printf(".");
            fflush(stdout);
         }

         if(dual_gap) {
            printf("Duality gap sufficiently small\n");
         }
      } else if(iter%10 ==0) {
         printf(".");
         fflush(stdout);
      }
   }

   printf("Touched %f examples\n", hit/((double)(hit+skip)));
//   printf("freeing w_o\n");
   mxFree(w_o);
   mxFree(Q);
   mxFree(counts);
   mxFree(etas);
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// Input: labels, feats, weights, C, reg, alphas
//        0     , 1    , 2      , 3, 4,   5
// Input: gt_set, l_hyp_set, losses, labels, reg, C, lb, alphas
//        0       1          2       3       4    5  6   7
   if(nrhs<4)
      mexErrMsgTxt("Wrong number of inputs"); 

   int N = mxGetNumberOfElements(prhs[0]);
   int Nw = mxGetM(prhs[1]);

   if(N!=mxGetN(prhs[1]))
      mexErrMsgTxt("Number of feature vectors doesn't match number of labels");

   mxArray *alpha_out = mxCreateNumericMatrix(N, 1, mxDOUBLE_CLASS, mxREAL);
   double *alphas = mxGetPr(alpha_out);

   // Nw+1 includes the bias bias
   mxArray *w_out = mxCreateNumericMatrix(1, Nw+1, mxDOUBLE_CLASS, mxREAL);
   double *wt = mxGetPr(w_out);

   double C0 = mxGetScalar(prhs[3]);
   double *C = (double *)mxMalloc(N*sizeof(double));
   double *weighting = mxGetPr(prhs[2]);
   for(int i=0; i<N; i++) {
      C[i] = C0*weighting[i];
   }


   double *reg;

   if(nrhs>=5 && !mxIsEmpty(prhs[4]))
      reg = mxGetPr(prhs[4]);
   else {
      reg = (double *)mxMalloc((Nw+1)*sizeof(double));

      for(int i=0; i<Nw; i++)
         reg[i] = 1.;
      
      reg[Nw] = .01;
   }

   double *feat = mxGetPr(prhs[1]);
   double *labels = mxGetPr(prhs[0]);

   if(0&&nrhs>=6 && mxGetNumberOfElements(prhs[5])==N) {
      printf("Initialzing alphas 2\n");
      double *alpha0 = mxGetPr(prhs[5]);
      for(int i=0; i<N; i++)
         alphas[i] = alpha0[i];

      // Initialize wt as well
      for(int i=0; i<N; i++) {
         for(int j=0; j<Nw; ++j)
            wt[j] += alphas[i]*labels[i]*feat[j+Nw*i];

         wt[Nw] += alphas[i]*labels[i];
      }
      printf("done!\n");
   }
//   optimize_dual(alpha_out, Nw, C, prhs[0], prhs[1], prhs[2], mxGetPr(prhs[3]), Q, N, mxGetPr(prhs[6]), wt);

   optimize_dual(labels, feat, C, reg, N, Nw, wt, alphas);

   mxFree(C);

   if(nlhs>=2) {
      plhs[1] = alpha_out;
   } else {
      mxDestroyArray(alpha_out);
   }

   if(nrhs<4) {
      mxFree(reg);
   }

   plhs[0] = w_out;

   double obj_primal = objective_l1(wt, labels, feat, C, reg, Nw, N);
   if(nlhs>=3) {
      plhs[2] = mxCreateDoubleScalar(obj_primal);
   }
//   printf("done with optimization\n");
}
