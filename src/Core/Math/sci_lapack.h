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
 *  sci_lapack.h
 * 
 *  Written by:
 *   Author: Andrew Shafer
 *   Department of Computer Science
 *   University of Utah
 *   Date: Oct 21, 2003
 *
 *  Copyright (C) 2003 SCI Group
*/

#ifndef SCI_Math_sci_lapack_h
#define SCI_Math_sci_lapack_h 1

#include <Core/share/share.h>

namespace SCIRun {

bool lapackinvert(double *A, int n);  

void lapacksvd(double **A, int m, int n, double *S, double **U, double **VT);

void lapackeigen(double **A, int n, double *EigReal, double *EigImag,
		 double **EigVect=0);

} // End namespace SCIRun

#endif /* SCI_Math_sci_lapack_h */
