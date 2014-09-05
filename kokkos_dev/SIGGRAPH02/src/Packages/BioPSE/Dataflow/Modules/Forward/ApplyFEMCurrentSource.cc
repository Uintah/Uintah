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
#include <Core/Datatypes/PointCloudField.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/Trig.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

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
  LockingHandle<TetVolField<int> > hCondField;

  if (hField->get_type_name(0)!="TetVolField" && hField->get_type_name(1)!="int"){
    error("Supplied field is not of type TetVolField<int>.");
    return;
  }
  else {
    hCondField = dynamic_cast<TetVolField<int>*> (hField.get_rep());
    hMesh = hCondField->get_typed_mesh();
  }
  
  MatrixHandle  hRhsIn;
  ColumnMatrix* rhsIn;
  
  TetVolMesh::Node::size_type nsize; hMesh->size(nsize);
  ColumnMatrix* rhs = scinew ColumnMatrix(nsize);
 
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
  } else {  // electrode sources
    FieldHandle hInterp;
    iportInterp_->get(hInterp);
    unsigned int sourceNode = Max(sourceNodeTCL_.get(), 0);
    unsigned int sinkNode = Max(sinkNodeTCL_.get(), 0);
      
    if (hInterp.get_rep()) {
      PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > >* interp = dynamic_cast<PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > > *>(hInterp.get_rep());
      if (!interp) {
	error("Input interp field wasn't interp'ing PointCloudField from a TetVolMesh::Node.");
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
  //! Sending result
  oportRhs_->send(MatrixHandle(rhs)); 
}
} // End namespace BioPSE
