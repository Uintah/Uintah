/*
 * file:     BuildHexFEMatrix.cc
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


#include <Packages/BioPSE/Core/Algorithms/NumApproximation/BuildHexFEMatrix.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/BBox.h>


namespace BioPSE {

using namespace SCIRun;

/*
 * BuildHexFEMatrix::BuildHexFEMatrix(...)
 * Constructor
 * @param 
 */
BuildHexFEMatrix::BuildHexFEMatrix(HexVolFieldIntHandle hFieldInt,
				   HexVolFieldTensorHandle hFieldTensor,
				   bool index_based,
				   vector<pair<string, Tensor> > &tens, 
				   double unitsScale) 
  : hFieldInt_(hFieldInt), hFieldTensor_(hFieldTensor),
    index_based_(index_based), tens_(tens), unitsScale_(unitsScale) {
  if (index_based_) hMesh_ = hFieldInt->get_typed_mesh();
  else hMesh_ = hFieldTensor->get_typed_mesh();
  rE_ = scinew ReferenceElement();
}

/*
 * BuildHexFEMatrix::~BuildHexFEMatrix()
 * Destructor
 */
BuildHexFEMatrix::~BuildHexFEMatrix() {}

/*
 * BuildHexFEMatrix::buildMatrix
 */
MatrixHandle BuildHexFEMatrix::buildMatrix() {

  int i,j;

  // create sparse matrix
  HexVolMesh::Node::size_type nsize;
  hMesh_->size(nsize);
  int numNodes = nsize;
  rows_ = scinew int[numNodes+1];
  cols_ = scinew int[numNodes*27]; // 27 entries per row only for inner nodes!!!
  int nonZeros = 0;
  HexVolMesh::Node::iterator nii, nie;
  hMesh_->begin(nii); hMesh_->end(nie);
  int index[27];
  int numNeighbors;
  j = 0;
  rows_[0] = 0;
  for(; nii!=nie; ++nii) {
	numNeighbors = getAllNeighbors(*nii, index); // get all neighbors, including the node itself (sorted)
	for(i=0; i<numNeighbors; i++) {
	  cols_[nonZeros+i] = index[i];
	}
	nonZeros += numNeighbors;
	j++;
	rows_[j] = nonZeros;
  }

  dA_  = scinew SparseRowMatrix(numNodes, numNodes, rows_, cols_, nonZeros);

  // zeroing
  double *a = dA_->a;
  for(i=0; i<nonZeros; i++) {
	a[i] = 0.0;
  }

  // fill the matrix
  HexVolMesh::Cell::iterator ii,ie;
  hMesh_->begin(ii);
  hMesh_->end(ie);
  double localMatrix[8][8];
  HexVolMesh::Node::array_type cell_nodes(8);

  // loop over cells
  for(; ii != ie; ++ii) {
	hMesh_->get_nodes(cell_nodes, *ii);
//	j = hFieldInt_->value(*ii); // get index of the cell (data location must be at cell !!!)
	buildLocalMatrix(localMatrix, *ii, cell_nodes);
	addLocal2GlobalMatrix(localMatrix, cell_nodes);
  }

  MatrixHandle hM(dA_);
  return hM;
  
}

void BuildHexFEMatrix::buildLocalMatrix(double localMatrix[8][8], HexVolMesh::Cell::index_type ci, HexVolMesh::Node::array_type& cell_nodes) {

  // compute matrix entries
  for(int i=0; i<8; i++) { // loop over nodes (basis functions)
	for(int j=0; j<8; j++) { // loop over nodes (basis functions)
	  localMatrix[i][j] = getLocalMatrixEntry(ci,i,j, cell_nodes);  
	}
  }
  
}

double BuildHexFEMatrix::getLocalMatrixEntry(HexVolMesh::Cell::index_type ci, int i, int j, 
			       HexVolMesh::Node::array_type& cell_nodes) {
  double value = 0.0;
  int w;
  double xa, xb, ya, yb, za, zb;
  Point p;
  // get global vertex positions
  hMesh_->get_point(p, cell_nodes[0]);
  xa = p.x(); ya = p.y(); za = p.z();
  hMesh_->get_point(p, cell_nodes[6]);
  xb = p.x(); yb = p.y(); zb = p.z();

  // get conductivity tensor for this cell
  double conductivity[3][3];
  if (index_based_) conductivity = tens_[hFieldInt_->value(ci)].second.mat_;
  else conductivity = hFieldTensor_->value(ci).mat_;

  for(w = 0; w < rE_->numQuadPoints; w++) { // loop over quadrature points
	value += 
	  rE_->qW[w] *
	  ((rE_->dphidx(i, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2])/rE_->dpsi1dx(xa,xb)) *
	   (conductivity[0][0]*unitsScale_/*xx*/ * rE_->dphidx(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi1dx(xa,xb) + 
		conductivity[0][1]*unitsScale_/*xy*/ * rE_->dphidy(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi2dy(ya,yb) + 
		conductivity[0][2]*unitsScale_/*xz*/ * rE_->dphidz(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi3dz(za,zb))+
	   
	   (rE_->dphidy(i, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2])/rE_->dpsi2dy(ya,yb)) *
	   (conductivity[0][1]*unitsScale_/*xy*/ * rE_->dphidx(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi1dx(xa,xb) + 
		conductivity[1][1]*unitsScale_/*yy*/ * rE_->dphidy(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi2dy(ya,yb) +
		conductivity[1][2]*unitsScale_/*yz*/ * rE_->dphidz(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi3dz(za,zb))+
	   
	   (rE_->dphidz(i, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2])/rE_->dpsi3dz(za,zb)) * 
	   (conductivity[0][2]*unitsScale_/*xz*/ * rE_->dphidx(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi1dx(xa,xb) +
		conductivity[1][2]*unitsScale_/*yz*/ * rE_->dphidy(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi2dy(ya,yb) +
		conductivity[2][2]*unitsScale_/*zz*/ * rE_->dphidz(j, rE_->qP[w][0], rE_->qP[w][1], rE_->qP[w][2]) / rE_->dpsi3dz(za,zb)));
  }

  value *= rE_->getAbsDetJacobian(xa, xb, ya, yb, za, zb);

  return value;

}

void BuildHexFEMatrix::addLocal2GlobalMatrix(double localMatrix[8][8], HexVolMesh::Node::array_type& cell_nodes) {
  // get global matrix indices
  for(int i=0; i<8; i++) {
	int row = (int)cell_nodes[i]; 
	for(int j=0; j<8; j++) {
	  int col = (int)cell_nodes[j]; 
	  dA_->get(row, col) += localMatrix[i][j];
	}
  }
}

// method will be deleted soon
int BuildHexFEMatrix::getAllNeighbors(HexVolMesh::Node::index_type nii, int *index) {
  int num = 0;
  int i,j,k;
  Point p[27];
  Point pBlack, pRed, pBlue, pGrey, pOragne, pTurquise, pPink, pGreen, pYellow;
  HexVolMesh::Node::array_type black(6);
  hMesh_->synchronize(HexVolMesh::NODE_NEIGHBORS_E);
  hMesh_->get_neighbors(black, nii);
  hMesh_->get_point(p[0], nii);
  
  // clear index
  for(i=0; i<27; i++) {
	index[i] = 0;
  }
  // set index of p0
  index[num] = (int)nii; //hField_->value(nii);
  num++;

  // find nodes
  for(i=0; i<((int)black.size()); i++) {
	hMesh_->get_point(pBlack, black[i]);
	if(pBlack.z() < p[0].z()) { // 1
	  index[num] = (int)black[i]; //hField_->value(black[i]);
	  num++;
	  hMesh_->get_point(p[1], black[i]);
	  HexVolMesh::Node::array_type red(6);
	  hMesh_->get_neighbors(red, black[i]);
	  for(j=0;j<((int)red.size());j++) {
		hMesh_->get_point(pRed, red[j]);
		if(pRed.x() < p[1].x()) { // 7
		  index[num] = (int)red[j]; //hField_->value(red[j]);
		  num++;
		  hMesh_->get_point(p[7], red[j]);
		  HexVolMesh::Node::array_type turquise(6);
		  hMesh_->get_neighbors(turquise, red[j]);
		  for(k=0; k<((int)turquise.size()); k++) {
			hMesh_->get_point(pTurquise, turquise[k]);
			if(pTurquise.y() < p[7].y()) { // 19
			  index[num] = (int)turquise[k];//hField_->value(turquise[k]);
			  num++;
			  // done
			}
			if(pTurquise.y() > p[7].y()) { // 20
			  index[num] = (int)turquise[k];//hField_->value(turquise[k]);
			  num++;
			  // done
			}
		  }
		}
		if(pRed.y() < p[1].y()) { // 8
		  index[num] = (int)red[j];//hField_->value(red[j]);
		  num++;
		  // done
		}
		if(pRed.x() > p[1].x()) { // 9
		  index[num] = (int)red[j];//hField_->value(red[j]);
		  num++;
		  hMesh_->get_point(p[9], red[j]);
		  HexVolMesh::Node::array_type pink(6);
		  hMesh_->get_neighbors(pink, red[j]);
		  for(k=0; k<((int)pink.size()); k++) {
			hMesh_->get_point(pPink, pink[k]);
			if(pPink.y() < p[9].y()) { // 21
			  index[num] = (int)pink[k];//hField_->value(pink[k]);
			  num++;
			  // done
			}
			if(pPink.y() > p[9].y()) { // 22
			  index[num] = (int)pink[k];//hField_->value(pink[k]);
			  num++;
			  // done
			}
		  }
		}
		if(pRed.y() > p[1].y()) { // 10
		  index[num] = (int)red[j];//hField_->value(red[j]);
		  num++;
		  // done
		}
	  }
	}
	if(pBlack.z() > p[0].z()) { // 2
	  index[num] = (int)black[i];//hField_->value(black[i]);
	  num++;
	  hMesh_->get_point(p[2], black[i]);
	  HexVolMesh::Node::array_type blue(6);
	  hMesh_->get_neighbors(blue, black[i]);
	  for(j=0; j<((int)blue.size());j++) {
		hMesh_->get_point(pBlue, blue[j]);
		if(pBlue.x() < p[2].x()) { // 11
		  index[num] = (int)blue[j];//hField_->value(blue[j]);
		  num++;
		  hMesh_->get_point(p[11], blue[j]);
		  HexVolMesh::Node::array_type green(6);
		  hMesh_->get_neighbors(green, blue[j]);
		  for(k=0; k<((int)green.size()); k++) {
			hMesh_->get_point(pGreen, green[k]);
			if(pGreen.y() < p[11].y()) { // 23
			  index[num] = (int)green[k];//hField_->value(green[k]);
			  num++;
			  // done
			}
			if(pGreen.y() > p[11].y()) { // 24
			  index[num] = (int)green[k];//hField_->value(green[k]);
			  num++;
			  // done
			}
		  }
		}
		if(pBlue.y() < p[2].y()) { // 12
		  index[num] = (int)blue[j];//hField_->value(blue[j]);
		  num++;
		  // done
		}
		if(pBlue.x() > p[2].x()) { // 13
		  index[num] = (int)blue[j];//hField_->value(blue[j]);
		  num++;
		  hMesh_->get_point(p[13], blue[j]);
		  HexVolMesh::Node::array_type yellow(6);
		  hMesh_->get_neighbors(yellow, blue[j]);
		  for(k=0; k<((int)yellow.size()); k++) {
			hMesh_->get_point(pYellow, yellow[k]);
			if(pYellow.y() < p[13].y()) { // 25
			  index[num] = (int)yellow[k];//hField_->value(yellow[k]);
			  num++;
			  // done
			}
			if(pYellow.y() > p[13].y()) { // 26
			  index[num] = (int)yellow[k];//hField_->value(yellow[k]);
			  num++;
			  // done
			}
		  }
		}
		if(pBlue.y() > p[2].y()) { // 14
		  index[num] = (int)blue[j];//hField_->value(blue[j]);
		  num++;
		  // done
		}
	  }
	}
	if(pBlack.y() < p[0].y()) { // 3
	  index[num] = (int)black[i];//hField_->value(black[i]);
	  num++;
	  hMesh_->get_point(p[3], black[i]);
	  HexVolMesh::Node::array_type oragne(6);
	  hMesh_->get_neighbors(oragne, black[i]);
	  for(j=0; j<((int)oragne.size()); j++) {
		hMesh_->get_point(pOragne, oragne[j]);
		if(pOragne.x() < p[3].x()) { // 15
		  index[num] = (int)oragne[j];//hField_->value(oragne[j]);
		  num++;
		  hMesh_->get_point(p[15], oragne[j]);
		  // done
		}
		if(pOragne.x() > p[3].x()) { // 16
		  index[num] = (int)oragne[j];//hField_->value(oragne[j]);
		  num++;
		  hMesh_->get_point(p[16], oragne[j]);
		  // done
		}
	  }
	}
	if(pBlack.y() > p[0].y()) { // 4
	  index[num] = (int)black[i];//hField_->value(black[i]);
	  num++;
	  hMesh_->get_point(p[4], black[i]);
	  HexVolMesh::Node::array_type grey(6);
	  hMesh_->get_neighbors(grey, black[i]);
	  for(j=0; j<((int)grey.size()); j++) {
		hMesh_->get_point(pGrey, grey[j]);
		if(pGrey.x() < p[4].x()) { // 17
		  index[num] = (int)grey[j];//hField_->value(grey[j]);
		  num++;
		  // done
		}
		if(pGrey.x() > p[4].x()) { // 18
		  index[num] = (int)grey[j];//hField_->value(grey[j]);
		  num++;
		  // done
		}
	  }
	}
	if(pBlack.x() < p[0].x()) { // 5
	  index[num] = (int)black[i];//hField_->value(black[i]);
	  num++;
	  // done
	}
	if(pBlack.x() > p[0].x()) { // 6
	  index[num] = (int)black[i];//hField_->value(black[i]);
	  num++;
	  // done
	}
  }

  sortNodes(index, num);

  return num;
}

// method will be deleted soon
void BuildHexFEMatrix::sortNodes(int *Array, int Elems) {
  int Swap, Temp;
  do { 
	Swap = 0; 
	for (int Count = 0; Count < (Elems - 1); Count++) { 
	  if (Array[Count] > Array[Count + 1]) { 
		Temp = Array[Count]; 
		Array[Count] = Array[Count + 1]; 
		Array[Count + 1] = Temp; Swap = 1; 
	  } 
	} 
  } while (Swap != 0);;
}


} // end of namespace BioPSE
