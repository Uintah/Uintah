
/*
 *  Mat.h:  Simple matrix calculations
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef sci_Math_Mat_h
#define sci_Math_Mat_h 1

#ifdef __cplusplus
extern "C" {
#endif
    void matsolve3by3(double mat[3][3], double rhs[3]);
    void matsolve3by3_cond(double mat[3][3], double rhs[3], double* rcond);
#ifdef __cplusplus
};
#endif

#endif
