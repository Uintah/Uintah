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
 *  SurfaceLaplacian.h
 *
 *   Yesim
 *   Department of Computer Science
 *   University of Utah
 *   Nov 2003
 *  Copyright (C) 2003 SCI Institute
 */


#ifndef SurfaceLaplacian_h
#define SurfaceLaplacian_h

#include <Core/Datatypes/TriSurfMesh.h>
//#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

namespace SCIRun {

DenseMatrix *surfaceLaplacian(TriSurfMesh *mesh);

} // End namespace SCIRun
#endif // SurfaceLaplacian_h
 
