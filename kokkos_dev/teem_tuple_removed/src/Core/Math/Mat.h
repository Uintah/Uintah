/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


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

#include <Core/share/share.h>

#ifdef __cplusplus
extern "C" {
#endif
    void SCICORESHARE matsolve3by3(double mat[3][3], double rhs[3]);
    void SCICORESHARE matsolve3by3_cond(double mat[3][3], double rhs[3], double* rcond);
    void SCICORESHARE min_norm_least_sq_3(double *A[3], double *b, double *x, double *bprime, int size);

#ifdef __cplusplus
}
#endif

#endif
