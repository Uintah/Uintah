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

#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/Field.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/BBox.h>
#include <Packages/BioPSE/Core/Algorithms/Forward/SphericalVolumeConductor.h>

namespace BioPSE {

using namespace SCIRun;

class CreateDisAnisoSpheres : public Module {

  // ports
  FieldIPort  *hInField;
  MatrixIPort *hInRadii;
  MatrixIPort *hInConductivities;
  FieldOPort  *hOutField;

  FieldHandle  field_;

  DenseMatrix  *conductivity;
  ColumnMatrix *radius;

  HexVolField<int> *newHexField;
  TetVolField<int> *newTetField;

  bool tet;
  double max;

  void assignCompartment(Point center, double distance, vector<double> &tensor);
  void getCondTensor(Point center, int compartment, vector<double> &tensor);
  void processHexField();
  void processTetField();

public:

  CreateDisAnisoSpheres(GuiContext *context);
  virtual ~CreateDisAnisoSpheres();
  virtual void execute();

};

DECLARE_MAKER(CreateDisAnisoSpheres)

CreateDisAnisoSpheres::CreateDisAnisoSpheres(GuiContext *context) : Module("CreateDisAnisoSpheres", context, Filter, "Forward", "BioPSE") {}

CreateDisAnisoSpheres::~CreateDisAnisoSpheres() {}

void CreateDisAnisoSpheres::execute() {

  // get input ports
  hInField = (FieldIPort*)get_iport("Mesh");
  if(!hInField) {
	error("impossible to initialize input port 'Mesh'");
	return;
  }
  hInRadii = (MatrixIPort*)get_iport("SphereRadii");
  if(!hInRadii) {
	error("impossible to initialize input port 'SphereRadii'");
	return;
  }
  hInConductivities = (MatrixIPort*)get_iport("AnisoConductivities");
  if(!hInConductivities) {
	error("impossible to initialize input port 'AnisoConductivities'");
	return;
  }

  // get output port
  hOutField = (FieldOPort*)get_oport("Mesh");
  if(!hOutField) {
	error("impossible to initialize output port 'Mesh'");
	return;
  }

  // get input handles
  if(!hInField->get(field_) || !field_.get_rep()) {
	error("impossible to get input field handle.");
	return;
  }
  MatrixHandle radii_;
  if(!hInRadii->get(radii_) || !radii_.get_rep()) {
	error("impossible to get radii handle.");
	return;
  }
  MatrixHandle cond_;
  if(!hInConductivities->get(cond_) || !cond_.get_rep()) {
	error("impossible to get conductivity handle.");
	return;
  }

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

  // get units
  string units;
  radii_->get_property("units", units);

  // process the mesh
  if(field_->get_type_name(0) == "HexVolField") { 
	if(field_->get_type_name(1) != "int") {
	  error("input field was not of type 'HexVolField<int>'");
	  return;
	}
	tet = false;
	processHexField();
  }
  else {
	if(field_->get_type_name(0) == "TetVolField") {
	  if(field_->get_type_name(1) != "int") {
		error("input field was not of type 'TetVolField<int>'");
		return;
	  }
	  tet = true;
	  processTetField();
	}
	else {
	  error("input field is neither HexVolField<int> nor TetVolField<int>");
	  return;
	}
  }

  // update output
  if(!tet) {
	newHexField->set_property("units", units, false);
	hOutField->send(newHexField);
  }
  else {
	newTetField->set_property("units", units, false);
	hOutField->send(newTetField);
  }

}

void CreateDisAnisoSpheres::processHexField() {
  LockingHandle<HexVolField<int> > hexField = dynamic_cast<HexVolField<int>* >(field_.get_rep());
  HexVolMeshHandle mesh_ = hexField->get_typed_mesh();
  HexVolMesh *newMesh_   = scinew HexVolMesh(*mesh_->clone()); 
  newHexField = scinew HexVolField<int>(newMesh_, Field::CELL);  // cell-wise conductivity tensors -> set data location to cells
  newMesh_->synchronize(HexVolMesh::FACES_E);
  HexVolMesh::Face::iterator fii;
  newMesh_->begin(fii);
  double face_area   = newMesh_->get_area(*fii);
  double edge_length = sqrt(face_area);
  max += edge_length * radius->get(SCALP);
  //cout << "edge length: " << edge_length << endl;
  // set positions of the nodes and enumerate them
  HexVolMesh::Node::iterator nii, nie;
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
  HexVolMesh::Cell::iterator cii, cie;
  newMesh_->begin(cii);
  newMesh_->end(cie);
  vector<double> t(6);
  HexVolMesh::Cell::size_type ncells;
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

void CreateDisAnisoSpheres::processTetField() {
  LockingHandle<TetVolField<int> > tetField = dynamic_cast<TetVolField<int>* >(field_.get_rep());
  TetVolMeshHandle mesh_ = tetField->get_typed_mesh();
  TetVolMesh *newMesh_   = scinew TetVolMesh(*mesh_->clone());
  newTetField = scinew TetVolField<int>(newMesh_, Field::CELL);
  // set positions of the nodes and enumerate them
  TetVolMesh::Node::iterator nii, nie;
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
  TetVolMesh::Cell::iterator cii, cie;
  newMesh_->begin(cii);
  newMesh_->end(cie);
  vector<double> t(6);
  TetVolMesh::Cell::size_type ncells;
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

void CreateDisAnisoSpheres::assignCompartment(Point center, double distance, vector<double> &tensor) {
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

void CreateDisAnisoSpheres::getCondTensor(Point center, int comp, vector<double> &tensor) {
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
