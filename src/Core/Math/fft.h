
#ifndef Math_fft_h
#define Math_fft_h 1

#include <SCICore/share/share.h>

#ifdef __cplusplus
extern "C" {
#endif
SCICORESHARE void fft2d_float(float* data, int res, float sign,
		 unsigned long* flops, unsigned long* memrefs);
SCICORESHARE void fft1d_float(float* data, int n, float sign,
			    unsigned long* flops, unsigned long* memrefs);
SCICORESHARE void fft1d_stride_float(float* data, int n, int stride, float sign,
			unsigned long* flops, unsigned long* memrefs);
#ifdef __cplusplus
}
#endif


#endif
