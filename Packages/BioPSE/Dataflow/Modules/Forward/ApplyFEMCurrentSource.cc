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
 *  ApplyFEMCurrentSource.cc:  Builds the RHS of the FE matrix for current sources
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   May 1999
 *  Modified by:
 *   Alexei Samsonov
 *   March 2001
 *  Copyright (C) 1999, 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <Packages/BioPSE/Core/Algorithms/NumApproximation/ReferenceElement.h>

namespace BioPSE {

using namespace SCIRun;

class ApplyFEMCurrentSource : public Module {

  //! Private data

  //! Input ports
  FieldIPort*   iportField_;
  FieldIPort*   iportSource_;
  FieldIPort*   iportInterp_;
  MatrixIPort*  iportRhs_;

  //! Output ports
  MatrixOPort*  oportRhs_;
  MatrixOPort*  oportWeights_;

  int gen_;
  TetVolMesh::Cell::index_type loc;

  bool tet;

public:
  GuiInt sourceNodeTCL_;
  GuiInt sinkNodeTCL_;
  GuiString modeTCL_; //"dipole" or "electrodes" (if electrodes use interp map)

  //! Constructor/Destructor
  ApplyFEMCurrentSource(GuiContext *context);
  virtual ~ApplyFEMCurrentSource();
  
  //! Public methods
  virtual void execute();
};

DECLARE_MAKER(ApplyFEMCurrentSource)


ApplyFEMCurrentSource::ApplyFEMCurrentSource(GuiContext *context)
  : Module("ApplyFEMCurrentSource", context, Filter, "Forward", "BioPSE"),
    sourceNodeTCL_(context->subVar("sourceNodeTCL")),
    sinkNodeTCL_(context->subVar("sinkNodeTCL")),
    modeTCL_(context->subVar("modeTCL"))
{
}

ApplyFEMCurrentSource::~ApplyFEMCurrentSource()
{
}

void ApplyFEMCurrentSource::execute()
{
  iportField_ = (FieldIPort *)get_iport("Mesh");
  iportSource_ = (FieldIPort *)get_iport("Dipole Sources");
  iportRhs_ = (MatrixIPort *)get_iport("Input RHS");
  iportInterp_ = (FieldIPort *)get_iport("Interpolant");
  oportRhs_ = (MatrixOPort *)get_oport("Output RHS");
  oportWeights_ = (MatrixOPort *)get_oport("Output Weights");

  if (!iportField_) {
    error("Unable to initialize iport 'Mesh'.");
    return;
  }
  if (!iportSource_) {
    error("Unable to initialize iport 'Dipole Sources'.");
    return;
  }
  if (!iportRhs_) {
    error("Unable to initialize iport 'Input RHS'.");
    return;
  }
  if (!iportInterp_) {
    error("Unable to initialize iport 'Interpolant'.");
    return;
  }
  if (!oportRhs_) {
    error("Unable to initialize oport 'Output RHS'.");
    return;
  }
  if (!oportWeights_) {
    error("Unable to initialize oport 'Output Weights'.");
    return;
  }
  
  //! Obtaining handles to computation objects
  FieldHandle hField;
  
  if (!iportField_->get(hField) || !hField.get_rep()) {
    error("Can't get handle to input mesh.");
    return;
  }

  TetVolMeshHandle hMesh;
  HexVolMeshHandle hHexMesh;

  if (hField->get_type_name(0) == "TetVolField") {
    remark("Input is a 'TetVolField'");
    tet = true;
    hMesh = dynamic_cast<TetVolMesh*>(hField->mesh().get_rep());
  } else if (hField->get_type_name(0) == "HexVolField") {
    remark("Input is a 'HexVolField'");
    tet = false;
    hHexMesh = dynamic_cast<HexVolMesh*>(hField->mesh().get_rep());
  } else {
    error("Supplied field is not 'TetVolField' or 'HexVolField'");
    return;
  }
  
  MatrixHandle  hRhsIn;
  ColumnMatrix* rhsIn;
  ColumnMatrix* rhs;
  int nsize;

  if(tet) {
	TetVolMesh::Node::size_type nsizeTet; hMesh->size(nsizeTet);
	nsize = nsizeTet;
  }
  else {
	HexVolMesh::Node::size_type nsizeHex; hHexMesh->size(nsizeHex);
	nsize = nsizeHex;
  }

  rhs = scinew ColumnMatrix(nsize);

  // -- if the user passed in a vector the right size, copy it into ours 
  if (iportRhs_->get(hRhsIn) && 
      (rhsIn=dynamic_cast<ColumnMatrix*>(hRhsIn.get_rep())) && 
      (rhsIn->nrows() == nsize))
  {
    string units;
    if (rhsIn->get_property("units", units))
      rhs->set_property("units", units, false);

    for (int i=0; i < nsize; i++) 
      (*rhs)[i]=(*rhsIn)[i];
  }
  else{
    rhs->set_property("units", string("volts"), false);
 //   msgStream_ << "The supplied RHS doesn't correspond to the mesh in size. Creating own one..." << endl;
    rhs->zero();
  }

  // process tet mesh
  if(tet) {
  
	if (modeTCL_.get() == "dipole") {
	  FieldHandle hSource;
	  if (!iportSource_->get(hSource) || !hSource.get_rep()) {
		error("Can't get handle to source field.");
		return;
	  }
	  
	  LockingHandle<PointCloudField<Vector> > hDipField;
	  
	  if (hSource->get_type_name(0)!="PointCloudField" || hSource->get_type_name(1)!="Vector"){
		error("Supplied field is not of type PointCloudField<Vector>.");
		return;
	  }
	  else {
		hDipField = dynamic_cast<PointCloudField<Vector>*> (hSource.get_rep());
	  }
	  
	  hMesh->synchronize(Mesh::LOCATE_E);
	  
	  //! Computing contributions of dipoles to RHS
	  PointCloudMesh::Node::iterator ii;
	  PointCloudMesh::Node::iterator ii_end;
	  Array1<double> weights;
	  hDipField->get_typed_mesh()->begin(ii);
	  hDipField->get_typed_mesh()->end(ii_end);
	  for (; ii != ii_end; ++ii) {
		
		Vector dir = hDipField->value(*ii); // correct unit ???
		Point p;
		hDipField->get_typed_mesh()->get_point(p, *ii);
		
		if (hMesh->locate(loc, p)) {
		  msgStream_ << "Source p="<<p<<" dir="<<dir<<" found in elem "<<loc<<endl;
		  if (fabs(dir.x()) > 0.000001) {
			weights.add(loc*3);
			weights.add(dir.x());
		  }
		  if (fabs(dir.y()) > 0.000001) {
			weights.add(loc*3+1);
			weights.add(dir.y());
		  }
		  if (fabs(dir.z()) > 0.000001) {
			weights.add(loc*3+2);
			weights.add(dir.z());
		  }
		  
		  double s1, s2, s3, s4;
		  Vector g1, g2, g3, g4;
		  hMesh->get_gradient_basis(loc, g1, g2, g3, g4);
		  
		  s1=Dot(g1,dir);
		  s2=Dot(g2,dir);
		  s3=Dot(g3,dir);
		  s4=Dot(g4,dir);
		  
		  TetVolMesh::Node::array_type cell_nodes;
		  hMesh->get_nodes(cell_nodes, loc);
		  (*rhs)[cell_nodes[0]]+=s1;
		  (*rhs)[cell_nodes[1]]+=s2;
		  (*rhs)[cell_nodes[2]]+=s3;
		  (*rhs)[cell_nodes[3]]+=s4;
		} else {
		  msgStream_ << "Dipole: "<< p <<" not located within mesh!"<<endl;
		}
	  }
	  gen_=hSource->generation;
	  ColumnMatrix* w = scinew ColumnMatrix(weights.size());
	  for (int i=0; i<weights.size(); i++) (*w)[i]=weights[i];
	  oportWeights_->send(MatrixHandle(w));
	  oportRhs_->send(MatrixHandle(rhs)); 
	} else {  // electrode sources
	  FieldHandle hInterp;
	  iportInterp_->get(hInterp);
	  FieldHandle hSource;
	  iportSource_->get(hSource);
	  
	  unsigned int sourceNode = Max(sourceNodeTCL_.get(), 0);
	  unsigned int sinkNode = Max(sinkNodeTCL_.get(), 0);
	  
	  // if we have an Interp field and its type is good, hInterpTetToPC will
	  //  be valid after this block
	  LockingHandle<PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > > > hInterpTetToPC;
	  if (hInterp.get_rep()) {
		hInterpTetToPC = dynamic_cast<PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > > *>(hInterp.get_rep());
		if (!hInterpTetToPC.get_rep()) {
		  error("Input interp field wasn't interp'ing PointCloudField from a TetVolMesh::Node.");
		  return;
		}
	  }
	  
	  // if we have an Interp field and a Source field and all types are good,
	  //  hCurField will be valid after this block
	  LockingHandle<PointCloudField<double> > hCurField;
	  if (hInterpTetToPC.get_rep() && hSource.get_rep()) {
		if (hSource->get_type_name(0)=="PointCloudField") {
		  if (hSource->get_type_name(1)!="double") {
			error("Can only use a PointCloudField<double> when using an Interp field and a source field -- this mode is for specifying current densities");
			return;
		  }
		  hCurField = dynamic_cast<PointCloudField<double>*> (hSource.get_rep());
		  if (hInterpTetToPC->get_typed_mesh().get_rep() !=
			  hCurField->get_typed_mesh().get_rep()) {
			error("Can't have different meshes for the Source and Interp field");
			return;
		  } 
		} else {
		  error("Can only use a PointCloudField<double> for the current sources");
		  return;
		}
	  }
	  
	  // if we have don't have an Interp field, use the source/sink indices 
	  //  directly as TetVol nodes
	  
	  // if we do have an Interp field, but we don't have a Source field, 
	  //  then the source/sink indices refer to the PointCloud, so use the
	  //  InterpField to get their corresponding TetVol node indices
	  
	  // if we have an Interp field AND a Source field, then ignore the
	  //  source/sink indices.  The Source field and the Interp field
	  //  will have the same mesh, where the Interp field speifies the
	  //  TetVol node index for each source, and the Source field gives a
	  //  scalar quantity (current) for each source
	  
	  if (!hInterpTetToPC.get_rep()) {
		if ((int)sourceNode >= nsize || (int)sinkNode >= nsize)
		{
		  error("SourceNode or SinkNode was out of mesh range.");
		  return;
		}
		(*rhs)[sourceNode] += -1;
		(*rhs)[sinkNode] += 1;
		oportRhs_->send(MatrixHandle(rhs)); 
		return;
	  }
	  
	  if (!hCurField.get_rep()) {
		if (sourceNode < hInterpTetToPC->fdata().size() &&
			sinkNode < hInterpTetToPC->fdata().size()) {
		  sourceNode = hInterpTetToPC->fdata()[sourceNode].begin()->first;
		  sinkNode = hInterpTetToPC->fdata()[sinkNode].begin()->first;
		} else {
		  error("SourceNode or SinkNode was out of interp range.");
		  return;
		}
		(*rhs)[sourceNode] += -1;
		(*rhs)[sinkNode] += 1;
		oportRhs_->send(MatrixHandle(rhs)); 
		return;
	  }
	  
	  PointCloudMesh::Node::iterator ii;
	  PointCloudMesh::Node::iterator ii_end;
	  Array1<double> weights;
	  hInterpTetToPC->get_typed_mesh()->begin(ii);
	  hInterpTetToPC->get_typed_mesh()->end(ii_end);
	  for (; ii != ii_end; ++ii) {
		vector<pair<TetVolMesh::Node::index_type, double> > vp;
		hInterpTetToPC->value(vp, *ii);
		double currentDensity;
		hCurField->value(currentDensity, *ii);
		for (unsigned int vv=0; vv<vp.size(); vv++) {
		  unsigned int rhsIdx = (unsigned int)(vp[vv].first);
		  double rhsVal = vp[vv].second * currentDensity;
		  (*rhs)[rhsIdx] += rhsVal;
		}
	  }
	  oportRhs_->send(MatrixHandle(rhs)); 
	}
  }
  else { // process hex mesh
	if (modeTCL_.get() == "electrodes") { // electrodes -> has to be implemented
	  error("source/sink modelling is not yet available for HexFEM");
	  return;
	}
	// dipoles
	FieldHandle hSource;
	if (!iportSource_->get(hSource) || !hSource.get_rep()) {
	  error("Can't get handle to source field.");
	  return;
	}
	LockingHandle<PointCloudField<Vector> > hDipField;
	if (hSource->get_type_name(0)!="PointCloudField" || hSource->get_type_name(1)!="Vector"){
	  error("Supplied field is not of type PointCloudField<Vector>.");
	  return;
	}
	else {
	  hDipField = dynamic_cast<PointCloudField<Vector>*> (hSource.get_rep());
	}

	// get unit of the mesh -> is it in sync with the dipole moment
	//string units;
	//double unitsScale = 1.;
	//field->get_property("units", units);
	//if (units == "mm") unitsScale = 1./1000;
	//else if (units == "cm") unitsScale = 1./100;
	//else if (units == "dm") unitsScale = 1./10;
	//else if (units == "m") unitsScale = 1./1;
	
	hHexMesh->synchronize(Mesh::LOCATE_E);
	
	//! Computing contributions of dipoles to RHS
	PointCloudMesh::Node::iterator ii;
	PointCloudMesh::Node::iterator ii_end;
	Array1<double> weights;
	hDipField->get_typed_mesh()->begin(ii);
	hDipField->get_typed_mesh()->end(ii_end);
	HexVolMesh::Cell::index_type ci;
	ReferenceElement *rE_ = scinew ReferenceElement();
	for (; ii != ii_end; ++ii) { // loop over dipoles
	  Vector moment = hDipField->value(*ii); // correct unit of dipole moment -> should be checked
	  Point dipole;
	  hDipField->get_typed_mesh()->get_point(dipole, *ii); // position of the dipole
	  if (hHexMesh->locate(ci, dipole)) {
		msgStream_ << "Source p="<<dipole<<" dir="<< moment <<" found in elem "<< loc <<endl;
	  }
	  else {
		msgStream_ << "Dipole: "<< dipole <<" not located within mesh!"<<endl;
	  }

	  // get dipole in reference element
	  double xa, xb, ya, yb, za, zb;
	  Point p;
	  HexVolMesh::Node::array_type n_array;
	  hHexMesh->get_nodes(n_array, ci);
	  hHexMesh->get_point(p, n_array[0]);
	  xa = p.x(); ya = p.y(); za = p.z();
	  hHexMesh->get_point(p, n_array[6]);
	  xb = p.x(); yb = p.y(); zb = p.z();
	  Point diRef(rE_->isp1(dipole.x(), xa, xb), rE_->isp2(dipole.y(), ya, yb), rE_->isp3(dipole.z(), za, zb));

	  // update rhs
	  for(int i=0; i <8; i++) {
		/*int node = field->value(n_array[i]);*/
		double val = moment[0] /* * (1/unitsScale)*/ * rE_->dphidx(i, diRef.x(), diRef.y(), diRef.z()) / rE_->dpsi1dx(xa, xb)
		  + moment[1] /* * (1/unitsScale)*/ * rE_->dphidy(i, diRef.x(), diRef.y(), diRef.z()) / rE_->dpsi2dy(ya, yb)
		  + moment[2] /* * (1/unitsScale)*/ * rE_->dphidz(i, diRef.x(), diRef.y(), diRef.z()) / rE_->dpsi3dz(za,zb);
		rhs->put((int)n_array[i], val);
	  }
	  
	}
	oportRhs_->send(MatrixHandle(rhs));
  }

}
} // End namespace BioPSE
