#include <share/share.h>

#ifdef __cplusplus
extern "C" {
#endif

void SHARE ssmult(int beg, int end, int* rows, int* columns,
	    double* a, double* xp, double* bp);

void SHARE ssmult_upper(int beg, int end, int* rows, int* columns,
		  double* a, double* xp, double* bp);

/* This is too slow....*/
void SHARE ssmult_uppersub(int nrows, int beg, int end, int* rows, int* columns,
		     double* a, double* xp, double* bp);

#ifdef __cplusplus
}
#endif
