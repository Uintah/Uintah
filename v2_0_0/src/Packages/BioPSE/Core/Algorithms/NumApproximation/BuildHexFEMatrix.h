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

#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Tensor.h>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/ReferenceElement.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>

namespace BioPSE {

using namespace SCIRun;

class BuildHexFEMatrix;

typedef LockingHandle<HexVolField<int> > HexVolFieldIntHandle;
typedef LockingHandle<HexVolField<Tensor> > HexVolFieldTensorHandle;
typedef LockingHandle<BuildHexFEMatrix> BuildHexFEMatrixHandle;

class BuildHexFEMatrix {

  // private stuff
  HexVolFieldIntHandle hFieldInt_;
  HexVolFieldTensorHandle hFieldTensor_;
  bool index_based_;
  HexVolMeshHandle hMesh_;
  vector<pair<string, Tensor> >& tens_;
  MatrixHandle hA_;
  double unitsScale_;
  int *rows_;
  int *cols_;
  ReferenceElement *rE_;
  SparseRowMatrix *dA_;

  void buildLocalMatrix(double localMatrix[8][8], HexVolMesh::Cell::index_type ci, HexVolMesh::Node::array_type& cell_nodes);
  void addLocal2GlobalMatrix(double localMatrix[8][8], HexVolMesh::Node::array_type& cell_nodes);
  double getLocalMatrixEntry(HexVolMesh::Cell::index_type ci, int i, int j, HexVolMesh::Node::array_type& cell_nodes);
  int getAllNeighbors(HexVolMesh::Node::index_type nii, int *index);
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
