
#include <SCICore/share/share.h>

#ifdef __cplusplus
extern "C" {
#endif

void SCICORESHARE ssmult(int beg, int end, int* rows, int* columns,
	    double* a, double* xp, double* bp);

void SCICORESHARE ssmult_upper(int beg, int end, int* rows, int* columns,
		  double* a, double* xp, double* bp);

/* This is too slow....*/
void SCICORESHARE ssmult_uppersub(int nrows, int beg, int end, int* rows, int* columns,
		     double* a, double* xp, double* bp);

#ifdef __cplusplus
}
#endif
