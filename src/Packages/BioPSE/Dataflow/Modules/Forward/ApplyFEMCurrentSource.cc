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
 *
 *   Lorena Kreda, Northeastern University, November 2003
 */

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Modules/Fields/FieldInfo.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/BoxWidget.h>
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
  MatrixIPort*  iportCurrentPattern_;
  MatrixIPort*  iportCurrentPatternIndex_;
  MatrixIPort*  iportElectrodeParams_;

  //! Output ports
  MatrixOPort*  oportRhs_;
  MatrixOPort*  oportWeights_;

  int gen_;
  TetVolMesh::Cell::index_type loc;
  TetVolMesh::Face::index_type locTri;

  bool tet;
  bool hex;
  bool tri;

private:
  virtual double CalculateCurrent(double theta, double arclength, int index, int numElectrodes);

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
  int numParams=3;
  iportField_ = (FieldIPort *)get_iport("Mesh");
  iportSource_ = (FieldIPort *)get_iport("Dipole Sources");
  iportRhs_ = (MatrixIPort *)get_iport("Input RHS");
  iportInterp_ = (FieldIPort *)get_iport("Interpolant");
  iportCurrentPattern_ = (MatrixIPort *)get_iport("Current Pattern");
  iportCurrentPatternIndex_ = (MatrixIPort *)get_iport("CurrentPatternIndex");
  iportElectrodeParams_ = (MatrixIPort *)get_iport("Electrode Parameters");
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
  if (!iportCurrentPattern_) {
    error("Unable to initialize iport 'Current Pattern'.");
    return;
  }
  if (!iportCurrentPatternIndex_) {
    error("Unable to initialize iport 'CurrentPatternIndex'.");
    return;
  }
  if (!iportElectrodeParams_) {
    error("Unable to initialize iport 'Electrode Params'.");
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

  // Handle TriSurfField inconsistently for now - fix in 1.20.2
  TriSurfMeshHandle hTriMesh;
  LockingHandle<TriSurfField<int> > hTriCondField;


  /* put this with the TriSurf code 

  // Check for valid data type - must be <int>
  if (hField->get_type_name(1) != "int") {
    error("Data in supplied field is not of type int");
    return;
  }
  */

  tet = false;
  hex = false;
  tri = false;

  if (hField->get_type_name(0) == "TetVolField") {
    remark("Input is a 'TetVolField'");
    tet = true;
    hMesh = dynamic_cast<TetVolMesh*>(hField->mesh().get_rep());
  } 
  else if (hField->get_type_name(0) == "HexVolField") {
    remark("Input is a 'HexVolField'");
    hex = true;
    hHexMesh = dynamic_cast<HexVolMesh*>(hField->mesh().get_rep());
  } 
  else if (hField->get_type_name(0) == "TriSurfField") {
    remark("Input is a 'TriSurfField<int>'");
    hTriCondField = dynamic_cast<TriSurfField<int>*> (hField.get_rep());
    hTriMesh = hTriCondField->get_typed_mesh();
    tri = true;
  } else {
    error("Supplied field is not 'TetVolField' or 'HexVolField' or 'TriSurfField'");
    return;
  }

  if ((modeTCL_.get() == "electrode set") && (tri != true)) {
    error("Only TriSurfField type is supported in electrode set mode");
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
  else if (hex) {
	HexVolMesh::Node::size_type nsizeHex; hHexMesh->size(nsizeHex);
	nsize = nsizeHex;
  }
  else if(tri) {
        TriSurfMesh::Node::size_type nsizeTri; hTriMesh->size(nsizeTri);
	nsize = nsizeTri;
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
        // TET + DIPOLE
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
	} // TET + ELECTRODE PAIR 
        else if(modeTCL_.get() == "electrode pair") {  // electrode pair source
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
        // TET + ELECTRODE SET (not implemented yet)
        else if (modeTCL_.get() == "electrode set"){ 
	  error("electrode set modelling is not yet available for TetVolFEM");
	  return;
	}

  }
  else if (hex) { // process hex mesh
        // HEX + ELECTRODE PAIR (not implemented yet)
	if (modeTCL_.get() == "electrodes") { // electrodes -> has to be implemented
	  error("source/sink modelling is not yet available for HexFEM");
	  return;
	}
	// HEX + DIPOLE
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
    // The following code supporting TriSurf's will be improved in 1.20.2
    // It's here as a place holder, just to get it in, but it's really not very good
  else if(tri) {
    // TRI + DIPOLE

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
  
    //! Computing contributions of dipoles to RHS
    PointCloudMesh::Node::iterator ii;
    PointCloudMesh::Node::iterator ii_end;
    Array1<double> weights;
    hDipField->get_typed_mesh()->begin(ii);
    hDipField->get_typed_mesh()->end(ii_end);
    for (; ii != ii_end; ++ii) {
      
      Vector dir = hDipField->value(*ii);
      Point p;
      hDipField->get_typed_mesh()->get_point(p, *ii);

        if (hTriMesh->locate(locTri, p)) {
          msgStream_ << "Source p="<<p<<" dir="<<dir<<" found in elem "<< locTri <<endl;

  	  if (fabs(dir.x()) > 0.000001) {
	    weights.add(locTri*3);
	    weights.add(dir.x());
	  }
	  if (fabs(dir.y()) > 0.000001) {
	    weights.add(locTri*3+1);
	    weights.add(dir.y());
	  }
	  if (fabs(dir.z()) > 0.000001) {
	    weights.add(locTri*3+2);
	    weights.add(dir.z());
	  }
	
	  double s1, s2, s3;
	  Vector g1, g2, g3;
   	  hTriMesh->get_gradient_basis(locTri, g1, g2, g3);

	  s1=Dot(g1,dir);
	  s2=Dot(g2,dir);
	  s3=Dot(g3,dir);
		
	  TriSurfMesh::Node::array_type face_nodes;
	  hTriMesh->get_nodes(face_nodes, locTri);
	  (*rhs)[face_nodes[0]]+=s1;
	  (*rhs)[face_nodes[1]]+=s2;
	  (*rhs)[face_nodes[2]]+=s3;
        } else {
	  msgStream_ << "Dipole: "<< p <<" not located within mesh!"<<endl;
        }
    } // end for loop
    gen_=hSource->generation;
    ColumnMatrix* w = scinew ColumnMatrix(weights.size());
    for (int i=0; i<weights.size(); i++) (*w)[i]=weights[i];
    oportWeights_->send(MatrixHandle(w));
  }
  // TRI + ELECTRODE PAIR 
  else if (modeTCL_.get() == "electrode pair") {

    FieldHandle hInterp;
    iportInterp_->get(hInterp);
    unsigned int sourceNode = Max(sourceNodeTCL_.get(), 0);
    unsigned int sinkNode = Max(sinkNodeTCL_.get(), 0);
      
    if (hInterp.get_rep()) {
   
        PointCloudField<vector<pair<TriSurfMesh::Node::index_type, double> > >* interp;
        interp = dynamic_cast<PointCloudField<vector<pair<TriSurfMesh::Node::index_type, double> > > *>(hInterp.get_rep());
        if (!interp) {
	  error("Input interp field wasn't interp'ing PointCloudField from a TriSurfMesh::Node.");
	  return;
        } else if (sourceNode < interp->fdata().size() &&
	  	   sinkNode < interp->fdata().size()) {
  	  sourceNode = interp->fdata()[sourceNode].begin()->first;
	  sinkNode = interp->fdata()[sinkNode].begin()->first;
        } else {
	  error("SourceNode or SinkNode was out of interp range.");
	  return;
        }
    }
    if (sourceNode >= nsize || sinkNode >= nsize)
    {
      error("SourceNode or SinkNode was out of mesh range.");
      return;
    }
    msgStream_ << "sourceNode="<<sourceNode<<" sinkNode="<<sinkNode<<"\n";
    (*rhs)[sourceNode] += -1;
    (*rhs)[sinkNode] += 1;
    }
  else // TRI + ELECTRODE SET
    {

    // Get the current pattern input - these are the input currents at each 
    // electrode - later we will combine with electrode information 
    // to produce an electrode density. In 2D, we assume the electrode height is
    // 1 because we assume no variation in the z direction, hence we set h=1
    // so that it doesn't influence the computation. I think this input is used
    // only for models other than the continuum model
    MatrixHandle  hCurrentPattern;
    if (!iportCurrentPattern_->get(hCurrentPattern) || !hCurrentPattern.get_rep()) {
      error("Can't get handle to current pattern matrix.");
      return;
    }

    // Get the current pattern index - this is used for calculating the current value
    // if the continuum model is used 
    MatrixHandle  hCurrentPatternIndex;
    ColumnMatrix* currPatIdx;
    int           k;

    // Copy the input current index into local variable, k 
    if (iportCurrentPatternIndex_->get(hCurrentPatternIndex) && 
        (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) && 
        (currPatIdx->nrows() == 1))
    {
      k=static_cast<int>((*currPatIdx)[0]);
    }
    else{
      msgStream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
    }

    // Get the interp field - this is the location of the electrodes interpolated onto the 
    // body mesh
    FieldHandle hInterp;
    iportInterp_->get(hInterp);

    // determine the dimension (number of electrodes) in the interp'd field of electrodes

    int numElectrodes;

    PointCloudMesh::Node::size_type nsize; 
    LockingHandle<PointCloudField<vector<pair<NodeIndex<int>,double> > > >hInterpField;
    PointCloudMeshHandle hPtCldMesh;
    PointCloudField<vector<pair<NodeIndex<int>,double> > >* interp = dynamic_cast<PointCloudField<vector<pair<NodeIndex<int>,double> > > *> (hInterp.get_rep());
    hPtCldMesh = interp->get_typed_mesh();
    hPtCldMesh->size(nsize);
    numElectrodes = (int) nsize;

    ColumnMatrix* currentPattern = scinew ColumnMatrix(numElectrodes);
    currentPattern=dynamic_cast<ColumnMatrix*>(hCurrentPattern.get_rep()); 

    // Get the electrode length from the parameters matrix
    MatrixHandle  hElectrodeParams;
    if (!iportElectrodeParams_->get(hElectrodeParams) || !hElectrodeParams.get_rep()) {
      error("Can't get handle to electrode parameters matrix.");
      return;
    }
    ColumnMatrix* electrodeParams = scinew ColumnMatrix(numParams);
    electrodeParams=dynamic_cast<ColumnMatrix*>(hElectrodeParams.get_rep());

    unsigned int electrodeModel = (*electrodeParams)[0];
    double electrodeLen = (*electrodeParams)[2];
    cout << "electrode model = " << electrodeModel << "  length= " << electrodeLen << endl;

    if (hInterp.get_rep()) {
      PointCloudField<vector<pair<TriSurfMesh::Node::index_type, double> > >* interp = dynamic_cast<PointCloudField<vector<pair<TriSurfMesh::Node::index_type, double> > > *>(hInterp.get_rep());
      if (!interp) {
	error("Input interp field wasn't interp'ing PointCloudField from a TriSurfMesh::Node.");
	return;
      }
 
      // TRI + ELECTRODE SET + GAP MODEL
      if (electrodeModel == 1) // gap model
      {
        // Convert the input current to a current density by dividing by the area of the 
        // electrodes. For 2D, assume the electrode height is equal to one.        
        for (int i=0; i<numElectrodes; i++) {
          (*currentPattern)[i] = (*currentPattern)[i] / electrodeLen;  // (units = amps/m)
        }
        for (int i=0; i<numElectrodes; i++) {
          cout << "***** output for electrode: " << i << " *** 0.5*electrodeLen = "<< electrodeLen/2 << endl;
          // find neighbors of this node
          int centerNode = interp->fdata()[i].begin()->first;
          int nextNodes[2];
          TriSurfMesh::Node::array_type neib_nodes;
          neib_nodes.clear();
          hTriMesh->synchronize(Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E);
          hTriMesh->get_neighbors(neib_nodes, TriSurfMesh::Node::index_type(centerNode));
          // find the neighbors that are on the boundary
          Point coord,centerCoord, nextPt[2];
          hTriMesh->get_point(centerCoord, centerNode);
          double boundaryRadius = sqrt(pow(centerCoord.x(),2) + pow(centerCoord.y(),2));          
          double radius;
          int boundaryNeighborCount=0;
          for (unsigned int jj=0; jj<neib_nodes.size(); jj++)
	  {
            hTriMesh->get_point(coord,neib_nodes[jj]);
            radius = sqrt(pow(coord.x(),2) + pow(coord.y(),2));  
            // if the radial distance of this point from the center is within 
            // 3% of the boundary radius, conclude this is a boundary node   
            if (abs(boundaryRadius-radius) < (0.03)*boundaryRadius)
	    {
              nextPt[boundaryNeighborCount]=coord;
              nextNodes[boundaryNeighborCount]=neib_nodes[jj];
              boundaryNeighborCount++;
	    }
            if (boundaryNeighborCount == 2) break;
	  }
          // traverse boundary in direction of nextNodes[jj] and update rhs
          // current vector as we go
          double s, d;

          for (unsigned int jj=0; jj<2; jj++)
	  {
            int lastNode, currNode, nextNode;
            Point currCoord, nextCoord;
            double cumulativeDistance=0.0;
            currCoord = centerCoord;
            nextCoord = nextPt[jj];
            currNode = centerNode;
            nextNode = nextNodes[jj];
  
            cout << "center node: " << centerNode << " next node: " << nextNode << endl;

            while (cumulativeDistance < electrodeLen/2)
	    {
              s = sqrt(pow((currCoord.x()-nextCoord.x()),2) + pow((currCoord.y()-nextCoord.y()),2));
              if ((cumulativeDistance + s) < electrodeLen/2)
	      {
                // this segment lies completely under the electrode
                cumulativeDistance = cumulativeDistance + s;
		//cout << "cumulativeDistance: " << cumulativeDistance << endl;
                (*rhs)[currNode] += 0.5*s/electrodeLen*(*currentPattern)[i]; // contribution to current node
		//cout << (0.5)*(s/electrodeLen)*(*currentPattern)[i] << " added to node " << currNode << endl;
                cout << currNode << "  " << (*rhs)[currNode] << endl;
                (*rhs)[nextNode] += (0.5)*s/electrodeLen*(*currentPattern)[i]; // contribution to next node
                //cout << (0.5)*(s/electrodeLen)*(*currentPattern)[i] << " added to node " << nextNode << endl;
                cout << nextNode << "  " << (*rhs)[nextNode] << endl;
              
                lastNode = currNode;
                currNode = nextNode;
                // find new nextNode, ie. the neighbor of the new currNode that does not equal lastNode
                hTriMesh->get_neighbors(neib_nodes, TriSurfMesh::Node::index_type(currNode));
                for (unsigned int jj=0; jj<neib_nodes.size(); jj++)
	        {
                  hTriMesh->get_point(coord,neib_nodes[jj]);
                  radius = sqrt(pow(coord.x(),2) + pow(coord.y(),2));  
                  // if the radial distance of this point from the center is within 
                  // 3% of the boundary radius, conclude this is a boundary node   
                  if (abs(boundaryRadius-radius) < (0.03)*boundaryRadius)
      	          {
                    if (neib_nodes[jj] != lastNode) 
                    {
                      nextNode = neib_nodes[jj];           
                      break;
                    }
                  }
	        }
                hTriMesh->get_point(currCoord,currNode);
                hTriMesh->get_point(nextCoord,nextNode);

                //cout << "  " << nextNode << endl;
                     
	      }
              else
	      {
                // this segment is only partially covered by the electrode
                d = (electrodeLen/2) - cumulativeDistance;
                // find area of trapezoidal region
                double area = d*(s-d)/s + (0.5)*d*(1-(s-d)/s);

		cout << "s = "<< s<< " d = " << d << " area = " << area << " currDens = " << (*currentPattern)[i] << endl;

                (*rhs)[currNode] += d/electrodeLen*(*currentPattern)[i]; // contribution to current node

                cout << (d/electrodeLen)*(*currentPattern)[i] << " added to node " << currNode << endl;
                cout << currNode << "  " << (*rhs)[currNode] << endl;
                
                break; // we are done with this half of the electrode.
	      }

	    } // end while
	  } // end for loop over two neighbors of central node
        } // end for loop over electrodes
      }

      // TRI + ELECTRODE SET + CONTINUUM MODEL
      else if (electrodeModel == 0) // continuum model
      {
	int count = 0;
	// start with the first electrode node
          hTriMesh->synchronize(Mesh::EDGES_E | Mesh::NODE_NEIGHBORS_E);
          int startNode = interp->fdata()[0].begin()->first;
          Point startCoord;
          hTriMesh->get_point(startCoord, startNode);
          Point coord,currCoord,nextCoord;
          int currNode = startNode;
          currCoord = startCoord;
          int nextNode, lastNode;
          double radius, nextTheta, s;
          double boundaryRadius = sqrt(pow(startCoord.x(),2) + pow(startCoord.y(),2));
          double theta = Atan(startCoord.y()/startCoord.x());
	  //          cout << "x: " << startCoord.x() << "  y: " << startCoord.y() << "  theta : " << theta << endl;

            // find the next node in order to determine value for arclength
            TriSurfMesh::Node::array_type neib_nodes;
            neib_nodes.clear();
            hTriMesh->get_neighbors(neib_nodes, TriSurfMesh::Node::index_type(currNode));
            // find the neighbor that is on the boundary and of increasing angle
            for (unsigned int jj=0; jj<neib_nodes.size(); jj++)
  	    {
              hTriMesh->get_point(coord,neib_nodes[jj]);
              //cout << "x " << coord.x() << "  y " << coord.y();
              radius = sqrt(pow(coord.x(),2) + pow(coord.y(),2));
              if ((coord.x() < 0) && (coord.y() > 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 3.14159;
              if ((coord.x() < 0) && (coord.y() < 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 3.14159;
              if ((coord.x() > 0) && (coord.y() < 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 2*3.14159;
              if ((coord.x() > 0) && (coord.y() > 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001));

              //cout << " theta " << theta << " nextTheta " << nextTheta << " radius " << radius << endl;
              // if the radial distance of this point from the center is within 
              // 3% of the boundary radius and the angle is increasing
              if ((abs(boundaryRadius-radius) < (0.03)*boundaryRadius) && (nextTheta > theta))
	      {
                nextNode=neib_nodes[jj];
                break;
	      }
	    }
            hTriMesh->get_point(nextCoord,nextNode);

            s = sqrt(pow((currCoord.x()-nextCoord.x()),2) + pow((currCoord.y()-nextCoord.y()),2));

            double current =  CalculateCurrent(theta, s, k, numElectrodes); // contribution to current node
            cout << current << endl;
          // compute current for this node
	    (*rhs)[startNode] = current; //CalculateCurrent(theta,s,k,numElectrodes); // contribution to current node

          cout << startNode << "  "<<(*rhs)[startNode] << endl;

          lastNode = startNode;
          // initialize cumulativeDistance and lastNode
          double cumulativeDistance = 0.0;
          double cumulativeTheta = theta;
          while ((cumulativeDistance < 2*3.14159*boundaryRadius) && (count < 128))
 	  {
            count++;
            cout << count << endl;
            // find the next node
            TriSurfMesh::Node::array_type neib_nodes;
            neib_nodes.clear();
            hTriMesh->get_neighbors(neib_nodes, TriSurfMesh::Node::index_type(currNode));
            // find the neighbor that is on the boundary and of increasing angle
            for (unsigned int jj=0; jj<neib_nodes.size(); jj++)
  	    {
              hTriMesh->get_point(coord,neib_nodes[jj]);
              //cout << "x " << coord.x() << "  y " << coord.y();
              radius = sqrt(pow(coord.x(),2) + pow(coord.y(),2));
              if ((coord.x() < 0) && (coord.y() > 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 3.14159;
              if ((coord.x() < 0) && (coord.y() < 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 3.14159;
              if ((coord.x() > 0) && (coord.y() < 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001)) + 2*3.14159;
              if ((coord.x() > 0) && (coord.y() > 0)) nextTheta = Atan(coord.y()/(coord.x() + 0.0000000001));

              //cout << " theta " << theta << " nextTheta " << nextTheta << " radius " << radius << endl;
              // if the radial distance of this point from the center is within 
              // 3% of the boundary radius and the angle is increasing
              if ((abs(boundaryRadius-radius) < (0.03)*boundaryRadius) && (neib_nodes[jj] != lastNode) && (count < 128))
	      {
                nextNode=neib_nodes[jj];
                break;
	      }
	    }
            hTriMesh->get_point(nextCoord,nextNode);

            s = sqrt(pow((currCoord.x()-nextCoord.x()),2) + pow((currCoord.y()-nextCoord.y()),2));
            cumulativeDistance = cumulativeDistance + s;
            cumulativeTheta = cumulativeTheta + nextTheta;

            if ((cumulativeDistance > 2*3.14159*boundaryRadius)|| (count > 127)) break;

            //cout << " cumulativeDistance = " << cumulativeDistance << " theta = " << theta << " nextTheta = " << nextTheta <<  endl;
           
            double current =  CalculateCurrent(nextTheta, s, k, numElectrodes); // contribution to current node
            cout << current << endl;

            (*rhs)[nextNode] = current; //CalculateCurrent(nextTheta, s, k, numElectrodes); // contribution to current node
   
            cout << nextNode << "  "<< (*rhs)[nextNode] << endl;

            lastNode = currNode;
            currNode = nextNode;
            currCoord = nextCoord;
            theta = nextTheta;

	  } // end while cumulativeDistance
      } // end else (if model == 0) 
    } // end if interp.get_rep()
  } // end electrode set else clause

  //! Sending result
  oportRhs_->send(MatrixHandle(rhs)); 
 }

}

double ApplyFEMCurrentSource::CalculateCurrent(double theta,double arclength, int index, int numElectrodes)
{
  cout << theta << "  " << arclength << "  " << index << "  " << numElectrodes << endl;
  double current;
  if ((index+1) <(numElectrodes/2)+1)
    {
      current = arclength * cos((index+1)*theta);
    }
  else
    {
    current = arclength * sin(((index+1)-numElectrodes/2)*theta);
    }

  return current;
}


} // End namespace BioPSE
