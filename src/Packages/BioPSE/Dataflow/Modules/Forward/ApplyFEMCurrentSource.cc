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

#include <Core/Containers/Array1.h>  
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/CurveField.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/Math/MiscMath.h>
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
  FieldIPort*   iportFieldBoundary_;
  MatrixIPort*  iportBoundaryToMesh_;

  //! Output ports
  MatrixOPort*  oportRhs_;
  MatrixOPort*  oportWeights_;
  MatrixOPort*  oportMeshToElectrodeMap_;

  int gen_;
  TetVolMesh::Cell::index_type loc;
  TetVolMesh::Face::index_type locTri;

  bool tet;
  bool hex;
  bool tri;

  enum ElectrodeModelType
  {
      CONTINUUM_MODEL = 0,
      GAP_MODEL
  };

private:
  virtual double CalculateCurrent(double theta, double arclength, int index, int numElectrodes);
  virtual double CalcContinuumTrigCurrent(Point p, int index, int numBoundaryNodes);
  virtual double ComputeTheta(Point);
  void ProcessTriElectrodeSet( ColumnMatrix* rhs, TriSurfMeshHandle hTriMesh );

public:
  GuiInt sourceNodeTCL_;
  GuiInt sinkNodeTCL_;
  GuiString modeTCL_; 
  // modeTCL_ (above) can be either: "dipole", "sources and sinks", or "electrode set" 
  // The iportInter_ input is only used if mode is "sources and sinks" or "electrode set". 

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
  iportCurrentPattern_ = (MatrixIPort *)get_iport("Current Pattern");
  iportCurrentPatternIndex_ = (MatrixIPort *)get_iport("CurrentPatternIndex");
  iportElectrodeParams_ = (MatrixIPort *)get_iport("Electrode Parameters");
  iportFieldBoundary_ = (FieldIPort *)get_iport("Boundary");
  iportBoundaryToMesh_ = (MatrixIPort *)get_iport("Boundary Transfer Matrix");

  oportRhs_ = (MatrixOPort *)get_oport("Output RHS");
  oportWeights_ = (MatrixOPort *)get_oport("Output Weights");

  // The following output is only produced in TriSurf + ElectrodeSet mode
  oportMeshToElectrodeMap_ = (MatrixOPort *)get_oport("Mesh to Electrode Map"); 

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
  // FieldBoundary and BoundaryInterp inputs are only utilized in the
  // TriSurf + ElectrodeSet configuration  
  if (!iportFieldBoundary_) {
    error("Unable to initialize iport 'Boundary'.");
    return;
  }
  if (!iportBoundaryToMesh_) {
    error("Unable to initialize iport 'Boundary Transfer Matrix'.");
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
  if (!oportMeshToElectrodeMap_) {
    error("Unable to initialize oport 'Mesh to Electrode Map'.");
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
  int nsize = 0;

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
  
  if (nsize > 0)
  {
      rhs = scinew ColumnMatrix(nsize);
  }
  else
  {
    error("Input mesh has zero size");
    return;
  }

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

  // process mesh
  if (tet) {
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
	} // TET + SOURCES AND SINKS 
        else if(modeTCL_.get() == "sources and sinks") {
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

    // HEX + DIPOLE
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
    // HEX + SOURCES AND SINKS (not implemented yet)
    else if (modeTCL_.get() == "sources and sinks") { 
	error("source/sink modelling is not yet available for HexFEM");
	return;
    }

    // HEX + ELECTRODE SET (not implemented yet)
    else if (modeTCL_.get() == "electrode set") { 
	error("electrode set modelling is not yet available for HexFEM");
	return;
    }

  }
  else if (tri) {
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
    // TRI + SOURCES AND SINKS 
    else if (modeTCL_.get() == "sources and sinks") {

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
        if (sourceNode >= (unsigned int) nsize || sinkNode >= (unsigned int) nsize)
        {
          error("SourceNode or SinkNode was out of mesh range.");
          return;
        }
        msgStream_ << "sourceNode="<<sourceNode<<" sinkNode="<<sinkNode<<"\n";
        (*rhs)[sourceNode] += -1;
        (*rhs)[sinkNode] += 1;
    }
    // TRI + ELECTRODE SET
    else if (modeTCL_.get() == "electrode set") {

      ProcessTriElectrodeSet( rhs, hTriMesh );

    } // end electrode set else clause

    //! Sending result
    oportRhs_->send(MatrixHandle(rhs)); 
  
  } // end of 'if' statement over mesh type

}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMCurrentSource::ProcessTriElectrodeSet
//
// Description: This method isolates a specialized block of code that handles the TriSurfMesh
//              and 'Electrode Set' mode.
//
// Inputs:
//
// Returns: 
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void
ApplyFEMCurrentSource::ProcessTriElectrodeSet( ColumnMatrix* rhs, TriSurfMeshHandle hTriMesh )
{
  int numParams=4;

    // Get the electrode parameters input vector
    // -----------------------------------------
  MatrixHandle  hElectrodeParams;

  if (!iportElectrodeParams_->get(hElectrodeParams) || !hElectrodeParams.get_rep()) 
    {
      error("Can't get handle to electrode parameters matrix.");
      return;
    }

  ColumnMatrix* electrodeParams = scinew ColumnMatrix(numParams);
  electrodeParams=dynamic_cast<ColumnMatrix*>(hElectrodeParams.get_rep());

  unsigned int electrodeModel = (unsigned int)((*electrodeParams)[0]);
  int numElectrodes           = (int) ( (*electrodeParams)[1]);
  double electrodeLen         = (*electrodeParams)[2];
  int startNodeIndex          = (int) ( (*electrodeParams)[3]);

  cout << "electrode model = " << electrodeModel << "  length= " << electrodeLen << endl;


    // Get the current pattern input
    // -----------------------------
    // These are the input currents at each electrode - later we will combine with electrode 
    // information to produce an electrode density. In 2D, we assume the electrode height is
    // 1 because we assume no variation in the z direction, hence we set h=1
    // so that it doesn't influence the computation. 
    // This input is used only for models other than the continuum model

  MatrixHandle  hCurrentPattern;
  if ((!iportCurrentPattern_->get(hCurrentPattern) || !hCurrentPattern.get_rep()) && (electrodeModel != CONTINUUM_MODEL)) 
    {
      error("Can't get handle to current pattern matrix.");
      return;
    }

  // Get the current pattern index
  // -----------------------------
  // This is used for calculating the current value if the continuum model is used 
  MatrixHandle  hCurrentPatternIndex;
  ColumnMatrix* currPatIdx;
  int           k = 0;

  // Copy the input current index into local variable, k 
  // ---------------------------------------------------
  if (iportCurrentPatternIndex_->get(hCurrentPatternIndex) && 
      (currPatIdx=dynamic_cast<ColumnMatrix*>(hCurrentPatternIndex.get_rep())) && 
      (currPatIdx->nrows() == 1))
    {
      k=static_cast<int>((*currPatIdx)[0]);
    }
  else
    {
      msgStream_ << "The supplied current pattern index is not a 1x1 matrix" << endl;
    }

  // Get the FieldBoundary input
  // ---------------------------
  FieldHandle      hFieldBoundary;
  CurveMeshHandle  hBoundaryMesh;
  LockingHandle<CurveField<double> > hCurveBoundary;
  bool boundary = false;
  if ( iportFieldBoundary_->get(hFieldBoundary) )
    { 
      if (hFieldBoundary.get_rep())
        {
          // Check field type - this only works for CurveFields<double> extracted from a TriSurf
          // -----------------------------------------------------------------------------------
          if ( (hFieldBoundary->get_type_name(0) == "CurveField") &&
               (hFieldBoundary->get_type_name(1) == "double") )
            {
              remark("Field boundary input is a CurveField<double>");
              hCurveBoundary = dynamic_cast<CurveField<double>*> ( hFieldBoundary.get_rep() );
              hBoundaryMesh = hCurveBoundary->get_typed_mesh();

              CurveMesh::Node::size_type nsize;
              hBoundaryMesh->size(nsize);
              int numCurveNodes = (int) nsize;
              cout << "There are " << numCurveNodes << " nodes in the boundary field." << endl;

              boundary = true;
	    }
          else
	    {
              remark("Supplied boundary field is not of type CurveField<double>");
	    }
	}
    }
  else
    {
      msgStream_ << "There is an error in the supplied boundary field" << endl;
    }

  // If a boundary field was supplied, check for the matrix that maps boundary
  // nodes to mesh nodes. This matrix is the output of the module: InterpolantToTransferMatrix
  // -----------------------------------------------------------------------------------------

  MatrixHandle      hBoundaryToMesh;

  if (boundary)
    {
      if ( !(iportBoundaryToMesh_->get(hBoundaryToMesh) &&
             hBoundaryToMesh.get_rep()) )
	{
          // disable susequent boundary-related code if we had a problem here
          boundary = false;
	}

    }
    
    
  // Get the interp field 
  // --------------------
  // This is the location of the electrodes interpolated onto the body mesh. The 
  // presence of this input means the user is electing to specify electrode locations
  // manually rather than use an automatic placement scheme selected through the 
  // electrode manager.
  // --------------------------------------------------------------------------------
  FieldHandle hInterp;
  //int numInterpFieldElectrodes;

  if ( iportInterp_->get(hInterp) && hInterp.get_rep()) 
    {
      // determine the dimension (number of electrodes) in the interp'd field of electrodes
      // ----------------------------------------------------------------------------------
      PointCloudMesh::Node::size_type nsize; 
      LockingHandle<PointCloudField<vector<pair<NodeIndex<int>,double> > > >hInterpField;
      PointCloudMeshHandle hPtCldMesh;
      PointCloudField<vector<pair<NodeIndex<int>,double> > >* interp = 
        dynamic_cast<PointCloudField<vector<pair<NodeIndex<int>,double> > > *> (hInterp.get_rep());
      hPtCldMesh = interp->get_typed_mesh();
      hPtCldMesh->size(nsize);
      //numInterpFieldElectrodes = (int) nsize;
    }
  // if electrode interp field is not supplied, determine electrode centers using 
  // number of electrodes, spacing from the electrode manager and extracted field boundary
  // -------------------------------------------------------------------------------------
  else
    {

    }
    
  // Make a local copy of the input current pattern 
  // Hold off on copying the current pattern until after we check if there's an interpolated 
  // electrode field as this could influence the value of numElectrodes
  // Also, this input is not needed for the continuum case and may not be present in this case.
  // ------------------------------------------------------------------------------------------
  ColumnMatrix* currentPattern = scinew ColumnMatrix(numElectrodes);
  currentPattern=dynamic_cast<ColumnMatrix*>(hCurrentPattern.get_rep()); 


    // Allocate vector for the mesh-to-electrode-map
  ColumnMatrix* meshToElectrodeMap;
  TriSurfMesh::Node::size_type msize;
  hTriMesh->size(msize);
  int numMeshNodes = (int) msize;

  meshToElectrodeMap = scinew ColumnMatrix(msize);

  // Initialize meshToElectrodeMap to all -1s. -1 indicates a non-electrode node; later we will 
  // identify the electrode nodes.
  for (int i = 0; i < numMeshNodes; i++)
    {
      (*meshToElectrodeMap)[i] = -1;
    }


  // TRI + ELECTRODE SET + CONTINUUM MODEL
  // -------------------------------------
  if (electrodeModel == CONTINUUM_MODEL)
    {
      if (boundary)
  	{
          // Visit each node on the boundary mesh.
          CurveMesh::Node::iterator nodeItr;
          CurveMesh::Node::iterator nodeItrEnd;
  	  
          hBoundaryMesh->begin(nodeItr);
          hBoundaryMesh->end(nodeItrEnd);
  
          Array1<int>       meshNodeIndex;
          Array1<double>    weight;

          int numBoundaryNodes = hBoundaryToMesh->nrows();

          for (; nodeItr != nodeItrEnd; ++nodeItr) 
            {		  
              Point p;
              hBoundaryMesh->get_point(p, *nodeItr);

              cout << "Cont------------ *nodeItr  = " << *nodeItr << " -----------------------------" << endl;
		                                                
              cout << " p=" << p << endl;

              // Find the corresponding node index in the body (TriSurf) mesh.
              hBoundaryToMesh->getRowNonzeros(*nodeItr, meshNodeIndex, weight);
		
              cout << "Boundary node " << (int) (*nodeItr) << " maps to mesh node " << meshNodeIndex[0] << endl; 
                
              int rhsIndex = meshNodeIndex[0];

              // Get the value for the current at this node and store this value in the RHS output vector
              (*rhs)[rhsIndex] = CalcContinuumTrigCurrent(p, k, numBoundaryNodes);

              // Tag this node as an "electrode" node
              (*meshToElectrodeMap)[rhsIndex] = (*nodeItr);
	    }
    
	} // end if (boundary)

    } // end else (if model == CONTINUUM_MODEL) 

  // TRI + ELECTRODE SET + GAP MODEL
  // -------------------------------
  else if (electrodeModel == GAP_MODEL ) 
    {
      // print out the current pattern
      // -----------------------------
      for (int i = 0; i < numElectrodes; i++)
        {
      	  cout << i << "  " << (*currentPattern)[i] << endl;
        }

      // print out number of electrodes according to electrode manager
      cout << "Creating " << numElectrodes << " electrodes of length " << electrodeLen << endl;

      // Note:
      // Originally, we didn't execute if an electrode interp field was not supplied because
      // this is the only way we know where the electrodes are on the input mesh.
      // Supplying a point cloud field of electrode positions could still be an option, but it is not supported now.
      // The equivalent effect can be obtained using the ElectrodeManager module.
      // The hInterp input is ignored by this part of ApplyFEMCurrentSource.
      // -----------------------------------------------------------------------------------------------------------

      // The code below places electrodes on the boundary of the input field.
      // --------------------------------------------------------------------

      // Traverse the boundary (curve) field and determine its length
      if (!boundary)
        {
	  error("Cannot proceed without a field boundary");
	  return;
        }
                              
      // Iterate over edges in the boundary and build a look-up-table that maps each
      // node index to its neighbor node indices
      CurveMesh::Node::size_type nsize;
      hBoundaryMesh->size(nsize);
      int numBoundaryNodes = (int) nsize;

      Array1<Array1<CurveMesh::Node::index_type> > neighborNodes;
      neighborNodes.resize(numBoundaryNodes);

      Array1<Array1<CurveMesh::Edge::index_type> > neighborEdges;
      neighborEdges.resize(numBoundaryNodes);

      Array1<double> edgeLength;
      edgeLength.resize(numBoundaryNodes);

      CurveMesh::Node::array_type childNodes;

      CurveMesh::Edge::iterator edgeItr;
      CurveMesh::Edge::iterator edgeItrEnd;

      hBoundaryMesh->begin(edgeItr);
      hBoundaryMesh->end(edgeItrEnd);

      double boundaryLength = 0.0;

      for (; edgeItr != edgeItrEnd; ++edgeItr) 
        {		  
          hBoundaryMesh->get_nodes(childNodes, *edgeItr);
          unsigned int nodeIndex0 = (unsigned int) childNodes[0];
          unsigned int nodeIndex1 = (unsigned int) childNodes[1];

          neighborNodes[nodeIndex0].add(nodeIndex1);
          neighborNodes[nodeIndex1].add(nodeIndex0);   
          neighborEdges[nodeIndex0].add(*edgeItr);
          neighborEdges[nodeIndex1].add(*edgeItr);

          // Store the edge length for future reference
          edgeLength[(unsigned int) *edgeItr] = hBoundaryMesh->get_size(*edgeItr);

          // Accumulate the total boundary length
          boundaryLength += edgeLength[(unsigned int) *edgeItr];

        }

      cout << "Boundary length: " << boundaryLength << endl;

      double electrodeSeparation = boundaryLength / numElectrodes;

      cout << "Electrode separation = " << electrodeSeparation << endl;

      // Using the map we just created (neighborNodes), traverse the boundary and assign electrode nodes
      // Create an array that maps boundary node index to electrode index. Initialize this array to -1's
      // meaning each boundary node is not assigned to an electrode. A boundary node may only belong to 
      // one electrode.

      Array1<int> nodeElectrodeMap;
      nodeElectrodeMap.resize(numBoundaryNodes);
      for (int i = 0; i < numBoundaryNodes; i++)
        {
          nodeElectrodeMap[i] = -1;
        }

      Array1<Array1<bool> > nodeFlags;
      nodeFlags.resize(numBoundaryNodes);
      for (int i = 0; i < numBoundaryNodes; i++)
        {
	  nodeFlags[i].resize(2);
          for (int j = 0; j < 2; j++)
            {
              nodeFlags[i][j] = false;
            }
        }

      Array1<Array1<double> > adjacentEdgeLengths;
      adjacentEdgeLengths.resize(numBoundaryNodes);
      for (int i = 0; i < numBoundaryNodes; i++)
        {
	  adjacentEdgeLengths[i].resize(2);
          for (int j = 0; j < 2; j++)
            {
              adjacentEdgeLengths[i][j] = 0.0;
            }
        }

      // Let the node in the boundary mesh given by startNodeIndex (in the electrodeParams input) be the 
      // first node in the first electrode.
      int prevNode = -1;
      int currNode = startNodeIndex;
      int nextNode = neighborNodes[currNode][1];  // selecting element [0] or [1] influences the direction
      // in which we traverse the boundary (this should be investigated;
      // [1] seems to work well relative to the analytic solution.

      double cumulativeElectrodeLength = 0.0;
      double cumulativeElectrodeSeparation = 0.0;

      bool done = false;

      double maxError = boundaryLength/numBoundaryNodes/2;  // maximum error we can accept = 1/2 avg. edge length
      double currError = 0.0;  // abs difference between a desired length and a current cumulative length

      int currEdgeIndex = 0;   // index of the boundary edge currently being considered

      bool firstNode = true;  // flag to indicate this is the first node in an electrode

      for (int i = 0; i < numElectrodes; i++)
        {
          //int lastElectrodeNode = -1;  // for use in determining first node in next electrode

	  cout << "Gap=================== i = " << i << " =============================" << endl;
          while (!done)
            {
	      // Label the current node with the current electrode ID 
	      if (nodeElectrodeMap[currNode] == -1) 
                {
                  nodeElectrodeMap[currNode] = i;
                }

              if (firstNode) 
                {
		  cout << "First node is: " << currNode << endl;
                  nodeFlags[currNode][0] = true;
                  firstNode = false;
                }

  	      // Traverse the boundary until distance closest to the desired electrode length is achieved.
              // -----------------------------------------------------------------------------------------

              // First, determine if this is the degenerate 1-node electrode case
              // ----------------------------------------------------------------
              if (electrodeLen <= maxError)
                {
                  nodeFlags[currNode][1] = true;  // the current node is the last node
                  done = true;
                  cumulativeElectrodeLength = 0.0;
                  cout << "done with this electrode (aa) " << currNode << " is last node in this electrode" << endl;
                  //lastElectrodeNode = currNode;
                }
             
              // Find the index of the edge between currNode and nextNode
              // --------------------------------------------------------
  	      int candidateEdgeIndex0 = neighborEdges[currNode][0];
	      int candidateEdgeIndex1 = neighborEdges[currNode][1];

              if ((int) neighborEdges[nextNode][0] == candidateEdgeIndex0 )
                {
  		  currEdgeIndex = candidateEdgeIndex0;
                }
              else if ((int) neighborEdges[nextNode][1] == candidateEdgeIndex0 )
                {
		  currEdgeIndex = candidateEdgeIndex0;
                }
              else if ((int) neighborEdges[nextNode][0] == candidateEdgeIndex1 )
                {
	          currEdgeIndex = candidateEdgeIndex1;
                }
              else if ((int) neighborEdges[nextNode][1] == candidateEdgeIndex1 )
                {
	          currEdgeIndex = candidateEdgeIndex1;
                }

	      // For first nodes that are not also last nodes, store the forward direction adjacent edge length
              if (nodeFlags[currNode][1] != true)
                {
                  adjacentEdgeLengths[currNode][1] = edgeLength[currEdgeIndex];
                }
              
              // Handle case where electrode covers more than one node
              if (!done)  
                {
		  // Determine if it is better to include the next node or the next two nodes 
                  // (If the effective electrode length will be closer to the desired electrode length.)

	          double testLength1 = cumulativeElectrodeLength + edgeLength[currEdgeIndex];
                  double testError1 = Abs(electrodeLen - testLength1);

                  // advance along boundary to test addition of the next node
                  int tempPrevNode = currNode;
                  int tempCurrNode = nextNode;
                  int tempNextNode = -1;
                  if ((int) neighborNodes[tempCurrNode][1] != tempPrevNode)
                    {
                      tempNextNode = (int) neighborNodes[tempCurrNode][1];
                    }
	          else
                    {
                      tempNextNode = (int) neighborNodes[tempCurrNode][0];
                    }
  
                  // Find the index of the edge between tempCurrNode and tempNextNode
                  // ----------------------------------------------------------------
  	          int candidateEdgeIndex0 = neighborEdges[tempCurrNode][0];
	          int candidateEdgeIndex1 = neighborEdges[tempCurrNode][1];

                  int tempEdgeIndex = -1;

                  if ((int) neighborEdges[tempNextNode][0] == candidateEdgeIndex0 )
                    {
  		      tempEdgeIndex = candidateEdgeIndex0;
                    }
                  else if ((int) neighborEdges[tempNextNode][1] == candidateEdgeIndex0 )
                    {
		      tempEdgeIndex = candidateEdgeIndex0;
                    }
                  else if ((int) neighborEdges[tempNextNode][0] == candidateEdgeIndex1 )
                    {
	              tempEdgeIndex = candidateEdgeIndex1;
                    }
                  else if ((int) neighborEdges[tempNextNode][1] == candidateEdgeIndex1 )
                    {
	              tempEdgeIndex = candidateEdgeIndex1;
                    }

	          double testLength2 = testLength1 + edgeLength[tempEdgeIndex];
                  double testError2 = Abs(electrodeLen - testLength2);

                  if (testError1 < testError2)
                    {
  		      // this means the nearer node achieves an electrode length closer to that desired
                      // and that this node is the last node in the electrode
  		      nodeElectrodeMap[nextNode] = i;
                      nodeFlags[nextNode][1] = true;
                      cumulativeElectrodeLength = testLength1;
  	             
                      // We also need to store the backward direction adjacent edge length for nextNode
                      adjacentEdgeLengths[nextNode][0] = edgeLength[currEdgeIndex];

                      done = true;
		      cout << "Last node is: " << nextNode << endl;
                      //lastElectrodeNode = nextNode;

                    }
                  else
                    {
                      // this means the further node achieves an electrode length closer to that desired
  		      nodeElectrodeMap[nextNode] = i;
                      cumulativeElectrodeLength = testLength1;

                      // For middle nodes, we need to store both the backward and forward adjacent edge lengths for nextNode
                      adjacentEdgeLengths[nextNode][0] = edgeLength[currEdgeIndex];
                      adjacentEdgeLengths[nextNode][1] = edgeLength[tempEdgeIndex];

		      cout << "Node " << nextNode << " is a middle node" << endl;

                    }

                  // advance node pointers whether the electrode stops or continues
                  prevNode = tempPrevNode;
                  currNode = tempCurrNode;
                  nextNode = tempNextNode;

                } // end if (!done)

            }  // end while (!done)

	  // At this point, we've finished with the current electrode.
          // Now we need to find the first node in the next electrode - this will be based on the value of
          // cumulativeElectrodeSeparation which we can initialize here to the value of cumulativeElectrodeLength.
	  cumulativeElectrodeSeparation = cumulativeElectrodeLength;
          cout << "cumulativeElectrodeSeparation = " << cumulativeElectrodeSeparation << endl;

          bool startNewElectrode = false;

          while (!startNewElectrode)
            {
	      cumulativeElectrodeSeparation += edgeLength[currEdgeIndex];

              currError = Abs(electrodeSeparation - cumulativeElectrodeSeparation);

              if (currError <= maxError)
                {
		  // We're within 1/2 an edge segment of the ideal electrode separation.
		  prevNode = currNode;
                  currNode = nextNode;

                  // Initialize nextNode
                  if ((int) neighborNodes[currNode][1] != prevNode)
                    {
                      nextNode = neighborNodes[currNode][1];
                    }
	          else
                    {
                      nextNode = neighborNodes[currNode][0];
                    }

                  startNewElectrode = true;
                      

		  cout << "First node in next electrode : " << currNode << endl;
                }
              else if (cumulativeElectrodeSeparation > electrodeSeparation)
                {
		  // The current error is greater than we allow, and we've exceeded the separation we want.
		  // We're trying to make the first node in the next electrode equal to the last node 
                  // in the last electrode - this is not allowed
	          error("Electrodes cannot overlap.");
	          return;                  
                }
              // Otherwise,
              // The current error is greater than 1/2 an edge segment, and the cumulativeElectrodeSeparation
              // is still less than what we want. This happens when we have more than one non-electrode
              // node between electrodes.
              // do nothing in this case... 
                 
	      if (!startNewElectrode)
                {
                  prevNode = currNode;
                  currNode = nextNode;
                  if ((int)neighborNodes[currNode][1] != prevNode)
                    {
                      nextNode = neighborNodes[currNode][1];
                    }
	          else
                    {
                      nextNode = neighborNodes[currNode][0];
                    }
                  cout << "prev: " << prevNode << " curr: " << currNode << " next: " << nextNode << endl;        
                }

            }  // end while (!startNewElectrode)

	  done = false;
          firstNode = true;
          cumulativeElectrodeLength = 0.0;
          cumulativeElectrodeSeparation = 0.0;

        } 

      // The following code was used for testing while under development - commented out for now
     
      for (int i = 0; i < numBoundaryNodes; i++)
        {
          cout << i << "  " << nodeElectrodeMap[i] << endl;
        }

      for (int i = 0; i < numBoundaryNodes; i++)
        {
      	  cout << i << "  " << (int) nodeFlags[i][0] << " " << (int) nodeFlags[i][1] << endl;
        }

      for (int i = 0; i < numBoundaryNodes; i++)
        {
      	  cout << i << "  " << adjacentEdgeLengths[i][0] << "  " << adjacentEdgeLengths[i][1] << endl;
        }
      

      cout << "Currents assigned to electrode nodes" << endl;
      cout << "------------------------------------" << endl;

      // Determine the currents for the RHS vector
      // -----------------------------------------
      for (int i = 0; i < numBoundaryNodes; i++)
        {
          // Note: size of the currentPattern vector must be equal to the number of electrodes!!
          // test this above
          // -----------------------------------------------------------------------------------
	  if (nodeElectrodeMap[i] != -1 )  // this is an electrode node
            {
	      double basisInt = 0.0;
              double current = 0.0;
              if ( (nodeFlags[i][0] == 1) && (nodeFlags[i][1] == 1) )  // special case: single node electrode
                {
		  current = (*currentPattern)[ nodeElectrodeMap[i] ];
                }
  	      else if (nodeFlags[i][0] == 1)  // this is the first node in an electrode
                {
		  basisInt = 0.5 * adjacentEdgeLengths[i][1];
                  current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];
                }
              else if (nodeFlags[i][1] == 1)  // this is the last node in an electrode
                {
		  basisInt = 0.5 * adjacentEdgeLengths[i][0];
                  current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];

                }
	      else  // this is a middle node in an electrode
                {
	          basisInt = 0.5 * adjacentEdgeLengths[i][0] + 0.5 * adjacentEdgeLengths[i][1];
                  current = basisInt * (*currentPattern)[ nodeElectrodeMap[i] ];
                }

              Array1<int>       meshNodeIndex;
              Array1<double>    weight;

              // Find the corresponding TriSurfMesh node index
	      hBoundaryToMesh->getRowNonzeros(i, meshNodeIndex, weight);
	
              //cout << "Gap Model " << i << "  equivalent node meshNodeIndex[0] = " << meshNodeIndex[0] << endl; 
              
              int rhsIndex = meshNodeIndex[0];

              (*rhs)[rhsIndex] = current;

              cout << "Bound. Idx: " << i << " Mesh Idx: " << rhsIndex << " current: " << current << endl;

              // Tag this node as an "electrode" node using the electrode index
              (*meshToElectrodeMap)[rhsIndex] = nodeElectrodeMap[i];

            }
        }

      // The following code was used for testing while under development - commented out for now
      
      //TriSurfMesh::Node::size_type msize;
      //hTriMesh->size(msize);
      //int numMeshNodes = (int) msize;


      //cout << "There are " << numMeshNodes << " nodes in the TriSurf field" << endl;
     
      //for (int i = 0; i < numMeshNodes; i++)
      //{
      //          cout << (*rhs)[i] << endl;
      //}

    } // end if GAP model
   
  //  Debugging statements - leave commented out for now
  //  TriSurfMesh::Node::size_type msize;
  //  hTriMesh->size(msize);
  //  int numMeshNodes = (int) msize;

  //  for (int i = 0; i < numMeshNodes; i++)
  //  {
  //	if ( abs( (*rhs)[i] ) > 0 )  cout << "mesh node index: " << i << " "<< (*rhs)[i] << endl;
  //  }

  //! Send the meshToElectrodeMap
  oportMeshToElectrodeMap_->send(MatrixHandle(meshToElectrodeMap)); 
    
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMCurrentSource::CalcContinuumTrigCurrent
//
// Description:
//
// Inputs:
//
// Returns: 
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double ApplyFEMCurrentSource::CalcContinuumTrigCurrent(Point p, int k, int numBoundaryNodes)
{
  double current;

  double theta = ComputeTheta(p);

  cout << "p = " << p.x() << "," << p.y() << "  theta = " << theta*180/PI << " k = " << k ;
  cout << " numBoundaryNodes = " << numBoundaryNodes << endl;  
    
  if ( k < (numBoundaryNodes/2) + 1 )
    {
      cout << "using cos" << endl;
      current = cos(k*theta);
    }
  else
    {
      cout << "using sin" << endl;
      current = sin((k-numBoundaryNodes/2)*theta);
    }

  cout << "current = " << current << endl;

  return current;
}

// ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
// ApplyFEMCurrentSource::ComputeTheta
//
// Description: Find the angle, theta, the input point makes with the positive x axis.
//              This is a helper method for CalcContinuumTrigCurrent.
//
// Inputs:  Point p
//
// Returns: double theta, ( 0 <= theta < 2*PI )
//
// +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
double ApplyFEMCurrentSource::ComputeTheta(Point p)
{

    double theta = 0.0;

    if ((p.x() <= 0) && (p.y() >= 0)) 
    {
        theta = Atan(p.y()/(p.x() + 0.0000000001)) + PI;
    }

    if ((p.x() <= 0) && (p.y() <= 0)) 
    {
        theta = Atan(p.y()/(p.x() + 0.0000000001)) + PI;
    }

    if ((p.x() >= 0) && (p.y() <= 0)) 
    {
        theta = Atan(p.y()/(p.x() + 0.0000000001)) + 2*PI;
    }

    if ((p.x() >= 0) && (p.y() >= 0))
    {
        theta = Atan(p.y()/(p.x() + 0.0000000001));
    }

    return theta;
}

// old version

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
