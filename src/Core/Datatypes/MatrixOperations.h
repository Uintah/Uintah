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
 *  MatrixOperations.h: Matrix Operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   August 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#ifndef SCI_project_MatrixOperations_h
#define SCI_project_MatrixOperations_h 1

#include <Core/Datatypes/Matrix.h>

namespace SCIRun {


MatrixHandle operator+(MatrixHandle A, MatrixHandle B);
MatrixHandle operator*(MatrixHandle A, MatrixHandle B);

} // End namespace SCIRun

#endif
