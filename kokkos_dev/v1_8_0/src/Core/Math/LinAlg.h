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

#include <Core/share/share.h>

#ifdef __cplusplus
extern "C" {
#endif
    SCICORESHARE double linalg_norm2(int n, const double* data);
    SCICORESHARE void linalg_mult(int n, double* result, double* a, double* b);
    SCICORESHARE void linalg_sub(int n, double* result, double* a, double* b);
    SCICORESHARE void linalg_add(int n, double* result, double* a, double* b);
    SCICORESHARE double linalg_dot(int n, double* a, double* b);
    SCICORESHARE void linalg_smadd(int n, double* result, double s, double* a, double* b);
    typedef double LinAlg_TriDiagRow[3];
    SCICORESHARE void linalg_tridiag(int n, LinAlg_TriDiagRow* data, double* c);
#ifdef __cplusplus
}
#endif

#endif

