/**
 * file:     SphereModel.cc
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     February 2003
 *
 * purpose:  selects those elementes from the unit volume which belong to the unit sphere
 *           and creates a new mesh containing only those elements;
 *
 * to do:    -> documentation
 * 
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>

namespace BioPSE {

using namespace SCIRun;

class SphereModel : public Module {

  // ports
  FieldIPort *hInField;
  FieldOPort *hOutField;

  HexVolField<int> *newField;
  double radius;

  void assignNodes(double distance, HexVolMesh::Node::array_type &node);
  bool isInside(double distance);

public:

  SphereModel(GuiContext *context);
  virtual ~SphereModel();
  virtual void execute();

};

DECLARE_MAKER(SphereModel)

SphereModel::SphereModel(GuiContext *context) : Module("SphereModel", context, Filter, "Forward", "BioPSE") {}

SphereModel::~SphereModel() {}

void SphereModel::execute() {
  
  // get ports
  hInField = (FieldIPort*)get_iport("HexVolField");
  if(!hInField) {
	error("impossible to initialize input port 'HexVolField'");
	return;
  }
  hOutField = (FieldOPort*)get_oport("HexVolField");
  if(!hOutField) {
	error("impossible to initialize output port 'HexVolField'");
	return;
  }

  // get input handle
  FieldHandle field_;
  if(!hInField->get(field_) || !field_.get_rep()) {
	error("impossible to get input field handle");
	return;
  }

  // get the mesh
  if(!(field_->get_type_name(0) == "HexVolField") || !(field_->get_type_name(1) == "int")) {
	error("input field is not of type 'HexVolField<int>'");
	return;
  }
  LockingHandle<HexVolField<int> > field = dynamic_cast<HexVolField<int>* >(field_.get_rep()); 
  HexVolMeshHandle mesh_ = field->get_typed_mesh();
  

  // process the mesh
  HexVolMesh *newMesh_   = scinew HexVolMesh(*mesh_->clone()); // get copy of the mesh
  newField = scinew HexVolField<int>(newMesh_, Field::NODE);   // create new field
  // set field values to default value
  HexVolMesh::Node::iterator nii, nie;
  newMesh_->begin(nii);
  newMesh_->end(nie);
  for(; nii != nie; ++nii) {
	newField->set_value(0, *nii);
  }
  // get cells which belong to the sphere
  HexVolMesh::Cell::iterator cii, cie;
  HexVolMesh::Node::array_type cell_nodes(8);
  newMesh_->begin(cii);
  newMesh_->end(cie);
  Point c;
  radius = 1.0;
  for(; cii != cie; ++cii) { 
	newMesh_->get_nodes(cell_nodes, *cii);
	newMesh_->get_center(c, *cii);
	Vector d = c.vector();
	assignNodes(d.length(), cell_nodes);
  }
  // create new mesh containing only the necessary nodes
  HexVolMesh *newMesh2_ = scinew HexVolMesh();
  HexVolMesh::Node::size_type nnodes;
  newMesh_->size(nnodes);
  HexVolMesh::Node::array_type all_nodes(nnodes);
  newMesh_->begin(nii);
  newMesh_->end(nie);
  int i=0;
  Point p;
  for(; nii != nie; ++nii) {
	if(newField->value(*nii) == 1) {
	  newMesh_->get_point(p, *nii);
	  all_nodes[i] = newMesh2_->add_point(p);
	  newField->set_value(i, *nii); // enumerate nodes for mapping
	  i++;
	}
  }
  // create elements in the new mesh
  HexVolMesh::Node::array_type new_nodes(8);
  newMesh_->begin(cii);
  newMesh_->end(cie);
  for(; cii != cie; ++cii) { 
	newMesh_->get_center(c, *cii);
	Vector d = c.vector();
	if(isInside(d.length())) {
	  newMesh_->get_nodes(cell_nodes, *cii);
	  for(i=0; i<8; i++) {
		new_nodes[i] = all_nodes[newField->value(cell_nodes[i])];
	  }
	  newMesh2_->add_elem(new_nodes);
	}
  }

  HexVolField<int> *newField2 = scinew HexVolField<int>(newMesh2_, Field::NODE);
  
  hOutField->send(newField2);

}

void SphereModel::assignNodes(double distance, HexVolMesh::Node::array_type &node) {
  if(distance <= radius) {
	for(int i=0; i<8; i++) {
	  newField->set_value(1, node[i]);
	}
  }
}

bool SphereModel::isInside(double distance) {
  if(distance <= radius)
	return true;
  else
	return false;
}

} // end namespace BioPSE
