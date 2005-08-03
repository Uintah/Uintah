/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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
 *
 *  Modified (adapted from BuildTetFEMatrix.h):
 *   Lorena Kreda, Northeastern University, October 2003
 */

//#include <Dataflow/Network/Module.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/TriSurfMesh.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Barrier.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class BuildTriFEMatrix;

typedef TriLinearLgn<int>                  btfeDatBasisi;
typedef TriLinearLgn<Tensor>               btfeDatBasist;
typedef TriSurfMesh<TriLinearLgn<Point> > TSMesh;
typedef GenericField<TSMesh, btfeDatBasisi,    vector<int> > TSFieldi;  
typedef GenericField<TSMesh, btfeDatBasist, vector<Tensor> > TSFieldt;   

typedef LockingHandle<TSFieldi>   TriSurfFieldIntHandle;
typedef LockingHandle<TSFieldt>   TriSurfFieldTensorHandle;
typedef LockingHandle<BuildTriFEMatrix>   BuildTriFEMatrixHandle;

class BuildTriFEMatrix: public Datatype {
  
  //! Private data members
  TriSurfFieldIntHandle           hFieldInt_;
  TriSurfFieldTensorHandle        hFieldTensor_;
  bool                            index_based_;
  TSMesh::handle_type             hMesh_;
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
  
  void build_local_matrix(double lcl[3][3], TSMesh::Face::index_type);
  
  void add_lcl_gbl(double lcl[3][3], TSMesh::Face::index_type, int, int, 
		   TSMesh::Node::array_type&);
  
 
public:
   //! Constructor
  BuildTriFEMatrix(TriSurfFieldIntHandle,
		   TriSurfFieldTensorHandle,
		   bool index_based,
		   vector<pair<string, Tensor> >&,
		   MatrixHandle&, 
		   int, double);
  static bool build_FEMatrix(TriSurfFieldIntHandle,
			     TriSurfFieldTensorHandle,
			     bool,
			     vector<pair<string, Tensor> > &,
			     MatrixHandle&, double,
			     int num_procs = -1);
  //! Destuctor
  virtual ~BuildTriFEMatrix();
  virtual void io(Piostream&);
};

} // end namespace BioPSE
