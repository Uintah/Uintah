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

extern "C" {
    double linalg_norm2(int n, double* data);
    double linalg_mult(int n, double* result, double* a, double* b);
    double linalg_sub(int n, double* result, double* a, double* b);
    double linalg_dot(int n, double* a, double* b);
    double linalg_smadd(int n, double* result, double s, double* a, double* b);
};

#endif

