// sam //

/** 
 * file:     AnisoSphereModel.cc
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     February 2003
 *
 * purpose:  input module for all specifications for a 4-layer anisotropic
 *           spherical volume conductor:
 *   
 *           -> radii for scalp, skull, cerebrospinal fulid and brain
 *           -> radial and tangential conductivities for the various compartments
 *           -> electrode positions (will be projected on the scalp) where the 
 *              potential should be measured
 * 
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Packages/BioPSE/Core/Algorithms/Forward/SphericalVolumeConductor.h>
#include <Packages/BioPSE/share/share.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE AnisoSphereModel : public Module {

  // ports
  FieldIPort  *hInElectrodes;
  MatrixIPort *hInConductivities;

  FieldOPort  *hOutElectrodes;
  MatrixOPort *hOutConductivities;
  MatrixOPort *hOutRadii;

  // gui data
  GuiDouble r_scalp;
  GuiDouble r_skull;
  GuiDouble r_cbsf;
  GuiDouble r_brain;
  GuiString units;
  GuiDouble rc_scalp;
  GuiDouble rc_skull;
  GuiDouble rc_cbsf;
  GuiDouble rc_brain;
  GuiDouble tc_scalp;
  GuiDouble tc_skull;
  GuiDouble tc_cbsf;
  GuiDouble tc_brain;

  bool condMatrix;

  ColumnMatrix *radii;
  DenseMatrix  *cond;

public:
  
  AnisoSphereModel(GuiContext *context);
  virtual ~AnisoSphereModel();
  virtual void execute();

};

DECLARE_MAKER(AnisoSphereModel)

AnisoSphereModel::AnisoSphereModel(GuiContext *context) : 
  Module("AnisoSphereModel", context, Filter, "Forward", "BioPSE"),
  r_scalp(context->subVar("r_scalp")),
  r_skull(context->subVar("r_skull")),
  r_cbsf(context->subVar("r_cbsf")),
  r_brain(context->subVar("r_brain")),
  units(context->subVar("units")),
  rc_scalp(context->subVar("rc_scalp")),
  rc_skull(context->subVar("rc_skull")),
  rc_cbsf(context->subVar("rc_cbsf")),
  rc_brain(context->subVar("rc_brain")),
  tc_scalp(context->subVar("tc_scalp")),
  tc_skull(context->subVar("tc_skull")),
  tc_cbsf(context->subVar("tc_cbsf")),
  tc_brain(context->subVar("tc_brain")) {}

AnisoSphereModel::~AnisoSphereModel() {}

void AnisoSphereModel::execute() {

  condMatrix = false;
  
  // get input ports
  hInElectrodes = (FieldIPort*)get_iport("ElectrodePositions");
  if(!hInElectrodes) {
	error("DipoleInAnisoSpheres::execute() -> impossible to initialize input port 'ElectrodePositions'");
	return;
  }
  hInConductivities = (MatrixIPort *)get_iport("AnisoConductivities");

  // get output ports
  hOutElectrodes = (FieldOPort*)get_oport("ElectrodePositions");
  if(!hOutElectrodes) {
	error("impossible to initialize output port 'ElectrodePositions'");
	return;
  }

  hOutRadii = (MatrixOPort*)get_oport("SphereRadii");
  hOutConductivities = (MatrixOPort*)get_oport("AnisoConductivities");

  // get electrode handle
  FieldHandle hElectrodes;
  hInElectrodes->get(hElectrodes);
  if(!hElectrodes.get_rep() ||
	 !(hElectrodes->get_type_name(0) == "PointCloudField") ||
	 !(hElectrodes->get_type_name(1) == "double")) {
	error("input electrode field is not of type 'PointCloudField<double>'");
	return;
  }
  
  // if possible, get matrix handle for conductivities
  MatrixHandle hConductivities;
  if(!hInConductivities->get(hConductivities) || !hConductivities.get_rep())
	condMatrix = false;
  else
	condMatrix = true;


  // get electrode positions from input port
  PointCloudField<double> *pElectrodes = dynamic_cast<PointCloudField<double>* >(hElectrodes.get_rep());
  if(!pElectrodes) {
	error("input field ElectrodePositions is not of type 'PointCloudField<double>'");
	return;
  }
  PointCloudMeshHandle mesh_ = pElectrodes->get_typed_mesh();
  PointCloudMesh::Node::iterator nii, nie;
  mesh_->begin(nii);
  mesh_->end(nie);
  vector<Point> electrodePositions;
  Point p;
  for(; nii != nie; ++nii) {
	mesh_->get_point(p, *nii);
	electrodePositions.push_back(p);
  }
  int numElectrodes = electrodePositions.size();

  // get size of the spheres
  radii = scinew ColumnMatrix(4);
  radii->put(SCALP, r_scalp.get());
  radii->put(SKULL, r_skull.get());
  radii->put(CBSF,  r_cbsf.get());
  radii->put(BRAIN, r_brain.get());

  // get conductivity information: if there is a matrix at the input port, take those values,
  // otherwise take the values from the gui
  cond = scinew DenseMatrix(4,2);
  if(condMatrix) {
	for(int i=0; i<4; i++) {
	  cond->put(i, RAD, hConductivities->get(i, RAD)); // radial
	  cond->put(i, TAN, hConductivities->get(i, TAN)); // tangential
	}
	// update gui
	rc_scalp.set(cond->get(SCALP, RAD)); tc_scalp.set(cond->get(SCALP, TAN));
	rc_skull.set(cond->get(SKULL, RAD)); tc_skull.set(cond->get(SKULL, TAN));
	rc_cbsf.set(cond->get(CBSF, RAD)); tc_cbsf.set(cond->get(CBSF, TAN));
	rc_brain.set(cond->get(BRAIN, RAD)); tc_brain.set(cond->get(BRAIN, TAN));
  }
  else {
	cond->put(SCALP, RAD, rc_scalp.get());
	cond->put(SKULL, RAD, rc_skull.get()); 
	cond->put(CBSF,  RAD, rc_cbsf.get());  
	cond->put(BRAIN, RAD, rc_brain.get()); 
	cond->put(SCALP, TAN, tc_scalp.get());
	cond->put(SKULL, TAN, tc_skull.get()); 
	cond->put(CBSF,  TAN, tc_cbsf.get()); 
	cond->put(BRAIN, TAN, tc_brain.get());
  }

  // get unit of sphere radii
  string unitss = units.get();

  // project electrodes on outer sphere (scalp) and create new PointCloudField
  double rad;
  PointCloudMesh *electrodeMesh = scinew PointCloudMesh();
  for(int i=0; i<numElectrodes; i++) {
	p = electrodePositions[i];
	rad = sqrt(p.x()*p.x() + p.y()*p.y() + p.z()*p.z());
	p *= radii->get(SCALP) / rad;
	electrodeMesh->add_point(p);
  }
  PointCloudMeshHandle hElectrodeMesh(electrodeMesh);
  PointCloudField<int> *newElectrodePositions = scinew PointCloudField<int>(hElectrodeMesh, Field::NODE);
  // enumerate the nodes
  electrodeMesh->begin(nii);
  electrodeMesh->end(nie);
  int i=0;
  for(; nii != nie; ++nii) {
	newElectrodePositions->set_value(i, *nii);
	i++;
  }

  // set units
  newElectrodePositions->set_property("units", unitss, false);
  radii->set_property("units", unitss, false);

  // send out results
  hOutElectrodes->send(newElectrodePositions);
  hOutConductivities->send(cond);
  hOutRadii->send(radii);

}

} // end namespace BioPSE

// ~sam //
