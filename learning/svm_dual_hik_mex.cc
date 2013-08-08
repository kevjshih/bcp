#include <math.h>
#include <string.h>
#include <sys/types.h>
#include "mex.h"
#include "blas.h"

#define BIASREG 0.05
//#define BIASREG 1.
#define IN_CACHE 2
#define WAIT 20

#define max(a,b) ((a)>(b)?(a):(b))
#define min(a,b) ((a)<(b)?(a):(b))
//double square(double x){return x*x;}
#define square(a) ((a)*(a))
#define abs(a) ((a)>0?(a):(-a))

typedef struct ModelHIKStruct{
   double *Ml;
   double *Mu;
   double bias;
   int Nw;
   int Nbins;
}ModelHIK;


void update_tables(ModelHIK *m, double mult, uint16_T *quant, double *feat)
{
   for(int i=0; i<m->Nw; i++) {
      for(int j=0; j<=quant[i]; j++)
         m->Ml[i + m->Nw*j]  += mult;
      

      for(int j=quant[i]+1; j<=m->Nbins; j++)
         m->Mu[i + m->Nw*j]  += mult*feat[i];
   }
}

double apply_model(ModelHIK *m, uint16_T *quant, double *feat)
{
   double val = 0;

   for(int i=0; i<m->Nw; i++) {
      val += m->Ml[i + m->Nw*quant[i]]*feat[i];
      val += m->Mu[i + m->Nw*(quant[i])];
   }

   return val;
}


void primal_dual_object_l1(ModelHIK *m, double *alphas, double *labels, double *feat, uint16_T *quant, double C, int N, double *res)
{
   double primal=0, dual=0;
   double primal_reg=0, dual_reg=0;

   for(int i=0; i<N; i++) {
      double score = apply_model(m, quant + i*m->Nw, feat + i*m->Nw);

      double reg = 1./2.*alphas[i]*labels[i]*score;
   
      // Update dual
      dual_reg -= reg;      // add regularization
      dual += alphas[i]; // add sum of alphas

      // Update primal
      primal_reg += reg;    // add regularization

      double loss = 1 - (score+m->bias)*labels[i];
      if(loss>0)
         primal += C*loss;
   }

   primal_reg += square(m->bias)*BIASREG;
   dual_reg -= square(m->bias)*BIASREG;

//   printf("Primal: %f + %f\n", primal_reg, primal);
//   printf("Dual: %f + %f\n", dual_reg, dual);

   res[0] = primal + primal_reg;
   res[1] = dual + dual_reg;
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
double * allocate_etas(double *feats, int Nw, int N)
{

   double *etas = (double *)mxMalloc(N*sizeof(double));

   for(int i=0; i<N; i++) {
      etas[i] = 0;

      for(int j=0; j<Nw; j++)
         etas[i] += feats[j+Nw*i]; //square(feats[j + Nw*i])*Q[j];
      

      etas[i] += 1/BIASREG; // For bias 
   }

   return etas;
}


// Ok - allocate sparse vector for feat_delta, compute feat_delta,  ...
void optimize_dual(ModelHIK *m, double *labels, double *feat, uint16_T *quant, double C, int N, double *alphas)
{
   int Nw = m->Nw, Nbins = m->Nbins;
   int *counts = (int *)mxMalloc(N*sizeof(int));
   memset(counts, 0, N*sizeof(int));
   double *etas = allocate_etas(feat, Nw, N);

   double delta_gap = 100000000;
   int *perm = (int *)mxMalloc(N*sizeof(int));

   double eta1, eta2;
   double epsilon2 = 5e-3;
 
   bool terminate = 0;
   int iter = 0, i, pi;

   double eta_r, alpha_d_lab, alpha_new;
   double obj_res[2];
  
   primal_dual_object_l1(m, alphas, labels, feat, quant, C, N, obj_res);        
   double obj_p, obj_d;
  
   printf("iter: %d - primal: %f, dual: %f\n", iter, obj_res[0], obj_res[1]);
   int hit = 0;
   int skip = 0;
   while(!terminate) {
      iter++;
      randperm(perm, N);

      for(pi = 0; pi<N; pi++) {
         i = perm[pi];

         if(counts[i]>IN_CACHE) {
            --counts[i];
            ++skip;
            continue;
         }
         
         ++hit;

//         eta1 = labels[i]*(dot_productb(w, feat + Nw*i, Nw) + w[Nw]) - 1;
         eta1 = labels[i]*(apply_model(m, quant+Nw*i, feat+Nw*i) + m->bias) - 1;

         eta_r = eta1/etas[i];
         alpha_new = fmin(fmax(alphas[i] - eta_r, 0), C);
         alpha_d_lab = labels[i]*(alpha_new - alphas[i]);
         alphas[i] = alpha_new;
  
         
         if(abs(alpha_d_lab)>0) {// If alpha changed, update model
            update_tables(m, alpha_d_lab, quant+Nw*i, feat+Nw*i);
            m->bias += alpha_d_lab/BIASREG;
//            printf("Updating! bias: %f, amount:%f\n", m->bias, alpha_d_lab);
         }

         if(alpha_new>1e-9) {
            counts[i] = 0;
         } else {
            ++counts[i];
            if(counts[i]>IN_CACHE) 
               counts[i] = WAIT;
         }
      }
   
            printf("-");
            fflush(stdout);
      if(iter>1000)
         terminate = 1;

      // Check for convergence
      if(iter%30 == 0 || iter<=1 || terminate) {
         primal_dual_object_l1(m, alphas, labels, feat, quant, C, N, obj_res);        
         obj_p = obj_res[0];
         obj_d = obj_res[1];

         bool dual_gap = !terminate && (obj_p - obj_d)/obj_d < epsilon2;
         bool dual_gap_approx = abs(((dual_gap) -(obj_p-obj_d)/obj_d)/dual_gap)<1e-9;
         terminate = terminate || dual_gap;// || dual_gap_approx;

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
         if(dual_gap_approx) {
            printf("Duality gap hasn't changed\n");
         }
      } else if(iter%10 ==0) {
         printf(".");
         fflush(stdout);
      }
   }

   printf("Touched %f examples\n", hit/((double)(hit+skip)));
//   printf("freeing w_o\n");
   mxFree(counts);
   mxFree(etas);
}



void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
// Input: labels, feats, quantX, Nbin, C, alphas
//        0       1      2       3     4, 5 
   if(nrhs<5)
      mexErrMsgTxt("Wrong number of inputs"); 

   int N = mxGetNumberOfElements(prhs[0]);
   int Nw = mxGetM(prhs[1]);

   if(N!=mxGetN(prhs[1]))
      mexErrMsgTxt("Number of feature vectors doesn't match number of labels");

   mxArray *alpha_out = mxCreateNumericMatrix(N, 1, mxDOUBLE_CLASS, mxREAL);
   double *alphas = mxGetPr(alpha_out);

   double *labels = mxGetPr(prhs[0]);
   double *feat = mxGetPr(prhs[1]);
   uint16_T *quantX = (uint16_T *)mxGetData(prhs[2]);
   int Nbin = (int)mxGetScalar(prhs[3]);
   double C = mxGetScalar(prhs[4]);

   // Nw includes the bias
   ModelHIK model;

   mxArray *Ml_out = mxCreateNumericMatrix(Nw, Nbin+1, mxDOUBLE_CLASS, mxREAL); // initialized to 0
   mxArray *Mu_out = mxCreateNumericMatrix(Nw, Nbin+1, mxDOUBLE_CLASS, mxREAL); // initialized to 0

   // Setup model
   model.Ml = mxGetPr(Ml_out); // Caches lower accumulation(cumsum(y*alpha*x<=bin))
   model.Mu = mxGetPr(Mu_out); // Caches upper accumulation(cumsum(y*alphas>bin)
   model.bias = 0;
   model.Nbins = Nbin;
   model.Nw = Nw;

   if(nrhs>=6 && mxGetNumberOfElements(prhs[5])==N) {
      printf("Initialzing alphas\n");
      double *alpha0 = mxGetPr(prhs[5]);
      for(int i=0; i<N; i++)
         alphas[i] = fmax(fmin(alpha0[i], C), 0);  // Reproject just to be safe

      // Initialize wt as well
      for(int i=0; i<N; i++) {
         update_tables(&model, alphas[i]*labels[i], quantX + Nw*i, feat + Nw*i);
         model.bias += alphas[i]*labels[i]/BIASREG;
      }
      printf("done!\n");
   }
//   optimize_dual(alpha_out, Nw, C, prhs[0], prhs[1], prhs[2], mxGetPr(prhs[3]), Q, N, mxGetPr(prhs[6]), wt);

   optimize_dual(&model, labels, feat, quantX, C, N, alphas);

   double obj_res[2];
   primal_dual_object_l1(&model, alphas, labels, feat, quantX, C, N, obj_res);
   double obj_primal = obj_res[0];
//   optimize_dual(alpha_out, Nw, C, prhs[0], prhs[1], prhs[2], mxGetPr(prhs[3]), Q, N, mxGetPr(prhs[6]), wt);

   plhs[0] = Ml_out;
   plhs[1] = Mu_out;
   plhs[2] = mxCreateDoubleScalar(model.bias);


   if(nlhs>=4){
      plhs[3] = alpha_out;
   } else {
      mxDestroyArray(alpha_out);
   }

//double apply_model(ModelHIK *m, uint16_T *quant, double *feat)
   printf("%f\n", apply_model(&model, quantX, feat));
   if(nlhs>=5) {
      plhs[4] = mxCreateDoubleScalar(obj_primal);
   }
//   printf("done with optimization\n");
}
