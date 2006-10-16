/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 * file:     CreateDisAnisoSpheres.cc
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     January 2003
 *
 * purpose:  -> assigns to each element a cunductivity tensor, according to the distance to the origin
 *
 * to do:    -> check computation of the conductivity tensors
 *           -> documentation   
 * 
 */

#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <Core/Datatypes/Field.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/BBox.h>
#include <Packages/BioPSE/Core/Algorithms/Forward/SphericalVolumeConductor.h>

namespace BioPSE {

using namespace SCIRun;

class CreateDisAnisoSpheres : public Module {
  typedef TetVolMesh<TetLinearLgn<Point> >                          TVMesh;
  typedef GenericField<TVMesh, ConstantBasis<int>,    vector<int> > TVField;  

  typedef HexVolMesh<HexTrilinearLgn<Point> >                       HVMesh;
  typedef GenericField<HVMesh, ConstantBasis<int>,    vector<int> > HVField; 

  // ports
  FieldIPort  *hInField;
  MatrixIPort *hInRadii;
  MatrixIPort *hInConductivities;
  FieldOPort  *hOutField;

  DenseMatrix  *conductivity;
  ColumnMatrix *radius;

  HVField *newHexField;
  TVField *newTetField;

  bool tet;
  double max;

  void assignCompartment(Point center, double distance, vector<double> &tensor);
  void getCondTensor(Point center, int compartment, vector<double> &tensor);
  void processHexField(FieldHandle field_);
  void processTetField(FieldHandle field_);

public:

  CreateDisAnisoSpheres(GuiContext *context);
  virtual ~CreateDisAnisoSpheres();
  virtual void execute();

};

DECLARE_MAKER(CreateDisAnisoSpheres)

CreateDisAnisoSpheres::CreateDisAnisoSpheres(GuiContext *context) : Module("CreateDisAnisoSpheres", context, Filter, "Forward", "BioPSE")
{
}


CreateDisAnisoSpheres::~CreateDisAnisoSpheres()
{
}


void
CreateDisAnisoSpheres::execute()
{
  // get input handles
  FieldHandle field_;
  if (!get_input_handle("Mesh", field_)) return;

  MatrixHandle radii_;
  if (!get_input_handle("SphereRadii", radii_)) return;

  MatrixHandle cond_;
  if (!get_input_handle("AnisoConductivities", cond_)) return;

  // get radii and conductivities
  int numRad = radii_->nrows();
  conductivity = scinew DenseMatrix(numRad+1, 2);
  radius = scinew ColumnMatrix(numRad+1);
  for(int i=0; i<numRad; i++) {
    radius->put(i, radii_->get(i,0));
    conductivity->put(i, RAD, cond_->get(i, RAD));
    conductivity->put(i, TAN, cond_->get(i, TAN));
  }
 
  max = radius->get(SCALP);

  //radius->put(AIR, radius->get(SCALP)*1.1); 
  //max = radius->get(AIR);

  // set conductivity of air
  conductivity->put(AIR, RAD, 0.0);
  conductivity->put(AIR, TAN, 0.0);

  // process the mesh
  const TypeDescription *mtd = field_->get_type_description(Field::MESH_TD_E);
  const TypeDescription *dtd = field_->get_type_description(Field::FDATA_TD_E);

  if(mtd->get_name().find("HexVolmesh") != string::npos) { 
    if(dtd->get_name().find("int") == string::npos) {
      error("input field was not of type 'HexVol with int data'");
      return;
    }
    tet = false;
    processHexField(field_);
  }
  else {
    if(mtd->get_name().find("TetVolMesh") != string::npos) {
      if(dtd->get_name().find("int") == string::npos) {
	error("input field was not of type 'TetVol with int data'");
	return;
      }
      tet = true;
      processTetField(field_);
    }
    else {
      error("input field is neither HexVol int nor TetVol int");
      return;
    }
  }

  // Update output.
  FieldHandle ftmp;
  if (!tet) { ftmp = newHexField; } else { ftmp = newTetField; }

  // Why don't we just do copy_properties?
  string units;
  if (radii_->get_property("units", units))
  {
    ftmp->set_property("units", units, false);
  }

  send_output_handle("Mesh", ftmp);
}


void
CreateDisAnisoSpheres::processHexField(FieldHandle field_)
{
  LockingHandle<HVField > hexField = dynamic_cast<HVField* >(field_.get_rep());
  HVMesh::handle_type mesh_ = hexField->get_typed_mesh();
  HVMesh *newMesh_   = scinew HVMesh(*mesh_->clone()); 
  newHexField = scinew HVField(newMesh_); /* cell-wise conductivity
					     tensors -> set data 
					     location to cells */
  newMesh_->synchronize(HVMesh::FACES_E);
  HVMesh::Face::iterator fii;
  newMesh_->begin(fii);
  double face_area   = newMesh_->get_area(*fii);
  double edge_length = sqrt(face_area);
  max += edge_length * radius->get(SCALP);
  //cout << "edge length: " << edge_length << endl;
  // set positions of the nodes and enumerate them
  HVMesh::Node::iterator nii, nie;
  newMesh_->begin(nii);
  newMesh_->end(nie);
  Point p;
  for(; nii != nie; ++nii) {
    mesh_->get_point(p, *nii);
    p.x(p.x()*max);
    p.y(p.y()*max);
    p.z(p.z()*max);
    newMesh_->set_point(p, *nii);
  }
  // assign conductivity tensors
  HVMesh::Cell::iterator cii, cie;
  newMesh_->begin(cii);
  newMesh_->end(cie);
  vector<double> t(6);
  HVMesh::Cell::size_type ncells;
  newMesh_->size(ncells);
  vector<pair<string, Tensor> > tensor;
  tensor.resize(ncells);
  int i = 0;
  Point c;
  for(; cii != cie; ++cii) {
    newMesh_->get_center(c, *cii);
    Vector d = c.vector();
    assignCompartment(c, d.length(), t);
    Tensor ten(t);
    tensor[i] = pair<string, Tensor>(to_string((int)i), ten);
    newHexField->set_value(i, *cii);
    i++;
  }
  newHexField->set_property("conductivity_table", tensor, false);
}


void
CreateDisAnisoSpheres::processTetField(FieldHandle field_)
{
  LockingHandle<TVField > tetField = dynamic_cast<TVField* >(field_.get_rep());
  TVMesh::handle_type mesh_ = tetField->get_typed_mesh();
  TVMesh *newMesh_   = scinew TVMesh(*mesh_->clone());
  newTetField = scinew TVField(newMesh_);
  // set positions of the nodes and enumerate them
  TVMesh::Node::iterator nii, nie;
  newMesh_->begin(nii);
  newMesh_->end(nie);
  Point p;
  for(; nii != nie; ++nii) {
    mesh_->get_point(p, *nii);
    p.x(p.x()*max);
    p.y(p.y()*max);
    p.z(p.z()*max);
    newMesh_->set_point(p, *nii);
  }
  // assign conductivity tensors
  TVMesh::Cell::iterator cii, cie;
  newMesh_->begin(cii);
  newMesh_->end(cie);
  vector<double> t(6);
  TVMesh::Cell::size_type ncells;
  newMesh_->size(ncells);
  vector<pair<string, Tensor> > tensor;
  tensor.resize(ncells);
  int i = 0;
  Point c;
  for(; cii != cie; ++cii) {
    newMesh_->get_center(c, *cii);
    Vector d = c.vector();
    assignCompartment(c, d.length(), t);
    Tensor ten(t);
    tensor[i] = pair<string, Tensor>(to_string((int)i), ten);
    newTetField->set_value(i, *cii);
    i++;
  }
  newTetField->set_property("conductivity_table", tensor, false);
}


void
CreateDisAnisoSpheres::assignCompartment(Point center, double distance,
                                         vector<double> &tensor)
{
  if(distance <= radius->get(BRAIN)) { // brain
    getCondTensor(center, BRAIN, tensor);
  }
  else {
    if(distance <= radius->get(CBSF)) { // cbsf
      getCondTensor(center, CBSF, tensor);
    }
    else {
      if(distance <= radius->get(SKULL)) { // skull
	getCondTensor(center, SKULL, tensor);
      }
      else {
	getCondTensor(center, SCALP, tensor);
	//if(distance <= radius->get(SCALP)) { // scalp
	//getCondTensor(center, SCALP, tensor);
	//}
	//else {
	//getCondTensor(center, AIR, tensor); // air 
	//}
      }
    }
  }
}


void
CreateDisAnisoSpheres::getCondTensor(Point center, int comp,
                                     vector<double> &tensor)
{
  // radial vector
  Vector radial = center.vector();

  radial.x(fabs(radial.x()));
  radial.y(fabs(radial.y()));
  radial.z(fabs(radial.z()));
  
  double eps = 1e-10;
  if(radial.x() <= eps) {
    radial.x(0.0); 
  }
  if(radial.y() <= eps) {
    radial.y(0.0); 
  }
  if(radial.z() <= eps) {
    radial.z(0.0); 
  }

  // tangential vector 1
  Vector tangential1;
  if(radial.z() >= eps) {
    radial.safe_normalize();
    tangential1.Set(radial.x(), radial.z(), -radial.y());
  }
  else { 
    if(radial.x() >= eps) {
      radial.safe_normalize();
      tangential1.Set(radial.z(), radial.y(), -radial.x());
    }
    else {
      if(radial.y() >= eps) {
	radial.safe_normalize();
	tangential1.Set(radial.y(), -radial.x(), radial.z());
      }
      else {
	// should happen only to the central element !!!
	radial.x(0.0); radial.y(0.0); radial.z(1.0);
	tangential1.Set(0.0, 1.0, 0.0);
      }
    }
  }

  tangential1.safe_normalize();
  // tangential vector2
  Vector tangential2 = Cross(tangential1, radial);
  tangential2.safe_normalize();
  
  // set conductivities
  eps = 1e-8;
  // xx
  //tensor[0] = conductivity->get(comp, RAD);
  tensor[0] = conductivity->get(comp, RAD) * radial.x() * radial.x() +
    conductivity->get(comp, TAN) * tangential1.x() * tangential1.x() +
    conductivity->get(comp, TAN) * tangential2.x() * tangential2.x();
  if(fabs(tensor[0]) < eps) tensor[0] = 0.0;
  // xy
  //tensor[1] = 0.0;
  tensor[1] = conductivity->get(comp, RAD) * radial.x() * radial.y() +
    conductivity->get(comp, TAN) * tangential1.x() * tangential1.y() +
    conductivity->get(comp, TAN) * tangential2.x() * tangential2.y();
  if(fabs(tensor[1]) < eps) tensor[1] = 0.0;
  // xz
  //tensor[2] = 0.0;
  tensor[2] = conductivity->get(comp, RAD) * radial.x() * radial.z() +
    conductivity->get(comp, TAN) * tangential1.x() * tangential1.z() +
    conductivity->get(comp, TAN) * tangential2.x() * tangential2.z();
  if(fabs(tensor[2]) < eps) tensor[2] = 0.0;
  // yy
  //tensor[3] = conductivity->get(comp, RAD);
  tensor[3] = conductivity->get(comp, RAD) * radial.y() * radial.y() +
    conductivity->get(comp, TAN) * tangential1.y() * tangential1.y() +
    conductivity->get(comp, TAN) * tangential2.y() * tangential2.y();
  if(fabs(tensor[3]) < eps) tensor[3] = 0.0;
  // yz
  //tensor[4] = 0.0;
  tensor[4] = conductivity->get(comp, RAD) * radial.y() * radial.z() +
    conductivity->get(comp, TAN) * tangential1.y() * tangential1.z() +
    conductivity->get(comp, TAN) * tangential2.y() * tangential2.z();
  if(fabs(tensor[4]) < eps) tensor[4] = 0.0;
  // zz
  //tensor[5] = conductivity->get(comp, RAD);
  tensor[5] = conductivity->get(comp, RAD) * radial.z() * radial.z() +
    conductivity->get(comp, TAN) * tangential1.z() * tangential1.z() +
    conductivity->get(comp, TAN) * tangential2.z() * tangential2.z();
  if(fabs(tensor[5]) < eps) tensor[5] = 0.0;
}


} // end of namespace BioPSE
