/*
 *  LinAlg.h:  Tuned linear algebra routines
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   November 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#ifndef Math_LinAlg_h
#define Math_LinAlg_h 1

#include <share/share.h>

#ifdef __cplusplus
extern "C" {
#endif
    SHARE double linalg_norm2(int n, double* data);
    SHARE void linalg_mult(int n, double* result, double* a, double* b);
    SHARE void linalg_sub(int n, double* result, double* a, double* b);
    SHARE void linalg_add(int n, double* result, double* a, double* b);
    SHARE double linalg_dot(int n, double* a, double* b);
    SHARE void linalg_smadd(int n, double* result, double s, double* a, double* b);
    typedef double LinAlg_TriDiagRow[3];
    SHARE void linalg_tridiag(int n, LinAlg_TriDiagRow* data, double* c);
#ifdef __cplusplus
}
#endif

#endif

