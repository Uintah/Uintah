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
 *  BuildTriFEMatrix.h:  class to build FE matrix for TriSurf mesh
 *
 *  Written by:
 *   Alexei Samsonov
 *   Department of Computer Science
 *   University of Utah
 *   March 2001   
 *  Copyright (C) 2001 SCI Group
 */

//#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/TCLstrbuff.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class BuildTriFEMatrix;
typedef LockingHandle<TriSurfField<int> >   TriSurfFieldIntHandle;
typedef LockingHandle<BuildTriFEMatrix>   BuildTriFEMatrixHandle;

class BuildTriFEMatrix: public Datatype {
  
  //! Private data members
  TriSurfFieldIntHandle                 hField_;
  TriSurfMeshHandle                hMesh_;
  MatrixHandle&                   hA_;
  SparseRowMatrix*                pA_;
  int                             np_;
  int*                            rows_;
  int*                            allCols_;
  Barrier                         barrier_;
  Array1<int>                     colIdx_;
  vector<pair<string, Tensor> >&  tens_;
  double                          unitsScale_;

  //! Private methods
  void parallel(int);
  
  void build_local_matrix(double lcl[3][3], TriSurfMesh::Face::index_type);
  
  void add_lcl_gbl(double lcl[3][3], TriSurfMesh::Face::index_type, int, int, TriSurfMesh::Node::array_type&);
  
 
public:
   //! Constructor
  BuildTriFEMatrix(TriSurfFieldIntHandle,
		vector<pair<string, Tensor> >&,
		MatrixHandle&, 
		int, double);
  static bool build_FEMatrix(TriSurfFieldIntHandle,
			     vector<pair<string, Tensor> > &,
			     MatrixHandle&, double);
  //! Destuctor
  virtual ~BuildTriFEMatrix();
  virtual void io(Piostream&);
};

} // end namespace BioPSE
