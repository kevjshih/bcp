#include "classifier.h"
#include  <emmintrin.h>

/*
float sse3_inner(const float* a, const float* b, unsigned int size)
{
        float z = 0.0f, fres = 0.0f;

        if ((size / 4) != 0) {
                const float* pa = a;
                const float* pb = b;
                __asm {
                        movss   xmm0, xmmword ptr[z]
                }
                for (unsigned int i = 0; i < size / 4; i++) {
                        __asm {
                                mov     eax, dword ptr[pa]
                                mov     ebx, dword ptr[pb]
                                movups  xmm1, [eax]
                                movups  xmm2, [ebx]
                                mulps   xmm1, xmm2
                                addps   xmm0, xmm1
                        }
                        pa += 4;
                        pb += 4;
                }
                __asm {
                        haddps  xmm0, xmm0
                        haddps  xmm0, xmm0
                        movss   dword ptr[fres], xmm0
                }
        }

        return fres;
}
*/

#define SSE
#ifdef SSE
float sse_inner(const float* a, const float* b, const float* a_end)
{
	__m128 msum = _mm_setzero_ps();
	while (a!=a_end) {
		msum = _mm_add_ps(msum, _mm_mul_ps(_mm_loadu_ps(a), _mm_loadu_ps(b)));
		a+=4;
		b+=4;
	}

#ifdef _MSC_VER
	__declspec(align(16)) float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#else
	float ftmp[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
#endif
	_mm_store_ps(ftmp, _mm_add_ps(_mm_movelh_ps(msum, msum), _mm_movehl_ps(msum, msum)));

	return ftmp[0] + ftmp[1];
}

float do_inner_product(const float* a, const float* b, size_t size) {
	if ((size % 4)==0)
		return sse_inner(a,b,a+size);

	size_t size4 = (size/4)*4;
	float score=sse_inner(a,b,a+size4);
	for (size_t i=size4; i<size; ++i)
		score+=a[i]*b[i];
	return score;
}
#else
float do_inner_product(const float* a, const float* b, size_t size) {
  float score=0;
  for (size_t i=0; i<size; ++i)
    score+=a[i]*b[i];
  return score;
}

#endif

