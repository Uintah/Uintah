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
 * file:     BuildHexFEMatrix.h
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     January 29th, 2003
 *
 * to do:    -> generalization such that 'distorted' cubic elements can be used
 *              (the mapping functions require perpendicular edges so far)
 *
 *           -> parallelization of the setup procedure
 *
 *           -> replacement of the function 'getAllNeighbors' when an equivalent
 *              method will be available in HexVolMesh / LatVolMesh
 *
 *           -> documentation
 *
 */

#ifndef BUILD_HEX_FE_MATRIX_H
#define BUILD_HEX_FE_MATRIX_H

#include <Core/Datatypes/GenericField.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Tensor.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/ReferenceElement.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

namespace BioPSE {

using namespace SCIRun;

class BuildHexFEMatrix;
typedef HexTrilinearLgn<int>                      DatBasisi;
typedef HexTrilinearLgn<Tensor>                   DatBasist;
typedef HexVolMesh<HexTrilinearLgn<Point> >       HVMesh;
typedef GenericField<HVMesh, DatBasisi,    vector<int> > HVFieldi;  
typedef GenericField<HVMesh, DatBasist, vector<Tensor> > HVFieldt;  

typedef LockingHandle<HVFieldi> HexVolFieldIntHandle;
typedef LockingHandle<HVFieldt> HexVolFieldTensorHandle;
typedef LockingHandle<BuildHexFEMatrix> BuildHexFEMatrixHandle;

class BuildHexFEMatrix {

  // private stuff
  HexVolFieldIntHandle hFieldInt_;
  HexVolFieldTensorHandle hFieldTensor_;
  bool index_based_;
  HVMesh::handle_type hMesh_;
  vector<pair<string, Tensor> >& tens_;
  MatrixHandle hA_;
  double unitsScale_;
  int *rows_;
  int *cols_;
  ReferenceElement *rE_;
  SparseRowMatrix *dA_;

  void buildLocalMatrix(double localMatrix[8][8], HVMesh::Cell::index_type ci, 
			HVMesh::Node::array_type& cell_nodes);
  void addLocal2GlobalMatrix(double localMatrix[8][8], 
			     HVMesh::Node::array_type& cell_nodes);
  double getLocalMatrixEntry(HVMesh::Cell::index_type ci, int i, int j, 
			     HVMesh::Node::array_type& cell_nodes);
  int getAllNeighbors(HVMesh::Node::index_type nii, int *index);
  void sortNodes(int *index, int length);

 public:
  
  // Constructor
  BuildHexFEMatrix(HexVolFieldIntHandle hFieldInt, 
		   HexVolFieldTensorHandle hFieldTensor, 
		   bool index_based,
		   vector<pair<string, Tensor> > &tens, double unitsScale);

  // Destructor
  virtual ~BuildHexFEMatrix();

  // access functions
  MatrixHandle buildMatrix();

};

} // end of namespace BioPSE


#endif
