
#ifndef Math_hf_h
#define Math_hf_h 1

#include <SCICore/share/share.h>

#ifdef __cplusplus
extern "C" {
#endif
SCICORESHARE void hf_float_s6(float* data, int xres, int yres);
SCICORESHARE void hf_minmax_float_s6(float* data, int xres, int yres,
			float* pmin, float* pmax);
#ifdef __cplusplus
};
#endif


#endif
