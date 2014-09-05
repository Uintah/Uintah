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
 *  BuildFEMatrix.h:  class to build FE matrix
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
#include <Core/Datatypes/FieldSet.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class BuildFEMatrix;
typedef LockingHandle<TetVol<int> >   TetVolIntHandle;
typedef LockingHandle<BuildFEMatrix>   BuildFEMatrixHandle;
typedef vector<pair<int, double> >     DirichletBC;       

class BuildFEMatrix: public Datatype {
  
  //! Private data members
  TetVolIntHandle         hField_;
  TetVolMeshHandle         hMesh_;
  DirichletBC&             dirBC_;
  MatrixHandle&            hA_;
  MatrixHandle&            hRhs_;
  SparseRowMatrix*         pA_;
  ColumnMatrix*            pRhs_;
  int                      np_;

  int*             rows_;
  int*             allCols_;
  Barrier          barrier_;
  Array1<int>      colIdx_;
  Array1<Tensor>&  tens_;
  
  //! Private methods
  void parallel(int);
  
  void build_local_matrix(double lcl[4][4], TetVolMesh::cell_index);
  
  void add_lcl_gbl(double lcl[4][4], TetVolMesh::cell_index, int, int, TetVolMesh::node_array&);
  
 
public:
   //! Constructor
  BuildFEMatrix(TetVolIntHandle,
		DirichletBC&,
		Array1<Tensor>&,
		MatrixHandle&, 
		MatrixHandle&, 
		int);
  static bool build_FEMatrix(TetVolIntHandle,
			     DirichletBC&,
			     Array1<Tensor>&,
			     MatrixHandle&, 
			     MatrixHandle&);
  //! Destuctor
  virtual ~BuildFEMatrix();
  virtual void io(Piostream&);
};

} // end namespace BioPSE
