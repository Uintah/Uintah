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

/**
 * file:     DipoleInAnisoSpheres.cc 
 * @version: 1.0
 * @author:  Sascha Moehrs
 * email:    sascha@sci.utah.edu
 * date:     January 2003
 * purpose:  Computes the potential on the outer sphere surface
 *           of a four layer spherical volume conductor with a dipole
 *           source in the innermost sphere
 *
 * to do:    -> correct handling of multiple dipoles
 *           -> review conversion of dipole positions
 *           -> documentation
 *           -> check for units (length scales)
 */

#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>    
#include <Dataflow/Ports/FieldPort.h>   
#include <Dataflow/Ports/MatrixPort.h> 
#include <Core/Datatypes/PointCloudField.h>  
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/Field.h>
#include <Packages/BioPSE/Core/Algorithms/Forward/SphericalVolumeConductor.h>

namespace BioPSE {
  
using namespace SCIRun;

class DipoleInAnisoSpheres : public Module { 
	
  // input
  FieldIPort  *hInSource;          // dipole
  FieldIPort  *hInElectrodes;      // electrode positions 
  MatrixIPort *hInConductivities; // conductivity information 
  MatrixIPort *hInRadii;  // radii

  // output
  FieldOPort  *hOutPotentials;  // potential on outer sphere surface at the electrode positions

  // some field handles
  FieldHandle hSource; 
  FieldHandle hElectrodes; 
  MatrixHandle hConductivities;
  MatrixHandle hRadii;

  int numDipoles, numElectrodes;
  bool condMatrix, condOut, radiiOut;

  PointCloudField<Vector> *pDipoles; 
  PointCloudField<int> *pElectrodes;
  PointCloudField<double> *nElectrodes;
  SphericalVolumeConductor *svc;
  PointCloudMesh *newElectrodeMesh;

  GuiDouble accuracy;
  GuiDouble expTerms;

public:
  
  DipoleInAnisoSpheres(GuiContext *context);
  virtual ~DipoleInAnisoSpheres();
  virtual void execute();
  //virtual void tcl_command(GuiArgs& args, void *userdata);

}; 

DECLARE_MAKER(DipoleInAnisoSpheres)

DipoleInAnisoSpheres::DipoleInAnisoSpheres(GuiContext *context) :
  Module("DipoleInAnisoSpheres", context, Filter, "Forward", "BioPSE"),
  accuracy(context->subVar("accuracy")),
  expTerms(context->subVar("expTerms"))
{}

DipoleInAnisoSpheres::~DipoleInAnisoSpheres() {}

void DipoleInAnisoSpheres::execute() {

  // get input ports
  hInSource = (FieldIPort*)get_iport("Dipole Sources");
  hInElectrodes = (FieldIPort*)get_iport("Electrodes");
  hInConductivities = (MatrixIPort*)get_iport("AnisoConductivities");
  hInRadii = (MatrixIPort*)get_iport("Radii");

  // get output ports
  hOutPotentials = (FieldOPort*)get_oport("ElectrodePotentials");

  update_state(NeedData);

  // get dipole handle
  hInSource->get(hSource);
  if(!hSource.get_rep() ||
	 !(hSource->get_type_name(0) == "PointCloudField") ||
	 !(hSource->get_type_name(1) == "Vector")) {
	error("dipole field needed");
	return;
  }  

  // get electrode handle
  hInElectrodes->get(hElectrodes);
  if(!hElectrodes.get_rep() ||
	 !(hElectrodes->get_type_name(0) == "PointCloudField") ||
	 !(hElectrodes->get_type_name(1) == "int")) {
	error("electrode field needed");
	return;
  }

  // get conductivity handle
  if(!hInConductivities->get(hConductivities) || (!hConductivities.get_rep())) {
	error("conductivity matrix needed");
	return;
  }

  // get radii handle
  if(!hInRadii->get(hRadii) || (!hRadii.get_rep())) {
	error("radii matrix needed");
	return;
  }
  
  
  // get dipole info from input port
  pDipoles  = dynamic_cast<PointCloudField<Vector>*>(hSource.get_rep());
  if(!pDipoles) {
	error("dipoles were not or type PointCloudField<Vector>");
	return;
  }
  PointCloudMeshHandle hMeshD = pDipoles->get_typed_mesh();
  PointCloudMesh::Node::iterator iter;
  PointCloudMesh::Node::iterator iter_end;
  hMeshD->begin(iter);
  hMeshD->end(iter_end);
  Point p; 
  Vector m; 
  vector<Vector> dipoles;
  vector<Point> dipolePositions;
  for(; iter != iter_end; ++iter) { // for all dipoles get ...
	pDipoles->value(m, *iter);      // ... moment and 
	dipoles.push_back(m); 
	hMeshD->get_point(p, *iter);     // ... position
	dipolePositions.push_back(p);
  }
  numDipoles = (int)dipolePositions.size();
  DenseMatrix dipoleMatrix(numDipoles, 6);
  int i;
  for(i=0; i<numDipoles; i++) {
	m = dipoles[i];
	p = dipolePositions[i];
	dipoleMatrix[i][XC] = p.x(); dipoleMatrix[i][YC] = p.y(); dipoleMatrix[i][ZC] = p.z();
	dipoleMatrix[i][XM] = m.x(); dipoleMatrix[i][YM] = m.y(); dipoleMatrix[i][ZM] = m.z();
  }

  // get electrode position info from input port 
  pElectrodes = dynamic_cast<PointCloudField<int>*>(hElectrodes.get_rep());
  if(!pElectrodes) {
	error("electrodes were not of type PointCloudField<double>");
	return;
  }
  PointCloudMeshHandle hMeshE = pElectrodes->get_typed_mesh();
  hMeshE->begin(iter);
  hMeshE->end(iter_end);
  vector<Point> electrodePositions;
  for(; iter != iter_end; ++iter) { // for all electrodes get ...
	hMeshE->get_point(p, *iter);     // ... position
	electrodePositions.push_back(p);
  }
  numElectrodes = (int)electrodePositions.size();
  DenseMatrix electrodeMatrix(numElectrodes, 3);
  for(i=0; i < numElectrodes; i++) {
	p = electrodePositions[i];
	electrodeMatrix[i][XC] = p.x(); electrodeMatrix[i][YC] = p.y(); electrodeMatrix[i][ZC] = p.z(); 
  }

  // size of the spheres (radii)
  ColumnMatrix radii(4);
  radii[SCALP] = hRadii->get(SCALP,0);
  radii[SKULL] = hRadii->get(SKULL,0);
  radii[CBSF]  = hRadii->get(CBSF,0);
  radii[BRAIN] = hRadii->get(BRAIN,0);

  // get units of the radii
  string units;
  double unitsScale = 1;
  hRadii->get_property("units", units);
  if (units == "mm") unitsScale = 1./1000;
  else if (units == "cm") unitsScale = 1./100;
  else if (units == "dm") unitsScale = 1./10;
  else if (units == "m") unitsScale = 1./1;

  // get conductivities
  ColumnMatrix radCond(4);
  ColumnMatrix tanCond(4);
  for(i=0; i<4; i++) {
	radCond[i] = hConductivities->get(i,0);
	tanCond[i] = hConductivities->get(i,1);
  }

  // show state
  update_state(JustStarted);

  // compute potential at the selected points (i.e. at the electrodes)
  ColumnMatrix result(numElectrodes);
  svc = scinew SphericalVolumeConductor(dipoleMatrix, electrodeMatrix, radii, radCond, tanCond, result, numDipoles, numElectrodes, accuracy.get(), unitsScale);
  svc->computePotential();
  expTerms.set(svc->getNumberOfSeriesTerms());

  // create new output field containing the potential values
  newElectrodeMesh = scinew PointCloudMesh(*hMeshE->clone());
  PointCloudMeshHandle hNewMesh(newElectrodeMesh);
  nElectrodes = scinew PointCloudField<double>(hNewMesh, 0);
  
  // set new electrode values
  vector<double>& newElectrodeValues = nElectrodes->fdata();
  for(i = 0; i < numElectrodes; i++) {
	newElectrodeValues[i] = result[i];
  }

  // send result to output port
  hOutPotentials->send(nElectrodes);

}

//void DipoleInAnisoSpheres::tcl_command(GuiArgs& args, void *userdata) {
//cout << args[1] << endl;
//Module::tcl_command(args, userdata);
//}

} // end namespace BioPSE
