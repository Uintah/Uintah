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
 *  Coregister.cc:  Coregister two point clouds
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Algorithms/Geometry/CoregPts.h>
#include <Core/Containers/Array1.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

class Coregister : public Module
{
private:
  GuiInt allowScale_;
  GuiInt allowRotate_;
  GuiInt allowTranslate_;
  GuiInt seed_;
  GuiInt iters_;
  GuiDouble misfitTol_;
  GuiString method_;
  int abort_;
  MusilRNG *mr_;
public:
  Coregister(GuiContext* ctx);
  virtual ~Coregister();
  virtual void execute();
  void tcl_command( GuiArgs&, void * );
};


DECLARE_MAKER(Coregister)

Coregister::Coregister(GuiContext* ctx)
  : Module("Coregister", ctx, Filter, "FieldsOther", "SCIRun"),
    allowScale_(ctx->subVar("allowScale")),
    allowRotate_(ctx->subVar("allowRotate")),
    allowTranslate_(ctx->subVar("allowTranslate")), seed_(ctx->subVar("seed")),
    iters_(ctx->subVar("iters")), misfitTol_(ctx->subVar("misfitTol")),
    method_(ctx->subVar("method"))
{
}

Coregister::~Coregister()
{
}

void
Coregister::execute()
{
  FieldIPort *fixed = (FieldIPort *) get_iport("Fixed PointCloudField");
  FieldIPort *mobile = (FieldIPort *) get_iport("Mobile PointCloudField");
  FieldIPort *dfield = (FieldIPort *) get_iport("DistanceField From Fixed");

  FieldHandle fixedH, mobileH;
  PointCloudField<double> *fixedPC, *mobilePC;
  PointCloudMeshHandle fixedM, mobileM;
  PointCloudMesh::Node::size_type nnodes;
  
  if (!fixed) {
    error("Unable to initialize iport 'Fixed PointCloudField'.");
    return;
  }
  if (!mobile) {
    error("Unable to initialize iport 'Mobile PointCloudField'.");
    return;
  }
  if (!dfield) {
    error("Unable to initialize iport 'DistanceField From Fixed'.");
    return;
  }

  if (!fixed->get(fixedH)) return;
  fixedPC = dynamic_cast<PointCloudField<double> *>(fixedH.get_rep());
  if (!fixedPC) return;

  fixedM = fixedPC->get_typed_mesh();

  if (!mobile->get(mobileH)) return;
  mobilePC = dynamic_cast<PointCloudField<double> *>(mobileH.get_rep());
  if (!mobilePC) return;

  mobileM = mobilePC->get_typed_mesh();

  
  fixedM->size(nnodes);
  if (nnodes < 3) {
    error("Fixed PointCloudField needs at least 3 input points.");
    return;
  }
  mobileM->size(nnodes);
  if (nnodes < 3) {
    error("Mobile PointCloudField needs at least 3 input points.");
    return;
  }

  MatrixOPort *omat = (MatrixOPort *)get_oport("Transform");
  if (!omat) {
    error("Unable to initialize oport 'Transform'.");
    return;
  }

  Array1<Point> fixedPts, mobilePts;
  Transform trans;

  PointCloudMesh::Node::iterator fni, fne, mni, mne;
  fixedM->begin(fni); fixedM->end(fne);
  mobileM->begin(mni); mobileM->end(mne);
  Point p;
  while (fni != fne) {
    fixedM->get_center(p, *fni);
    fixedPts.add(p);
    ++fni;
  }
  while (mni != mne) {
    mobileM->get_center(p, *mni);
    mobilePts.add(p);
    ++mni;
  }

  int allowScale = allowScale_.get();
  int allowRotate = allowRotate_.get();
  int allowTranslate = allowTranslate_.get();
  string method = method_.get();
  
  CoregPts* coreg;
  if (method == "Analytic") {
    coreg = scinew CoregPtsAnalytic(allowScale, allowRotate, allowTranslate);
  } else if (method == "Procrustes") {
    coreg = scinew CoregPtsProcrustes(allowScale, allowRotate, allowTranslate);
  } else { // method == "Simplex"
    FieldHandle dfieldH;
    if (!dfield->get(dfieldH)) {
      error("Simplex needs a distance field.");
      return;
    }
    ScalarFieldInterfaceHandle dfieldP = dfieldH->query_scalar_interface(this);
    if (!dfieldP.get_rep()) {
      error("Simplex needs a distance field.");
      return;
    }
    int seed = seed_.get();
    seed_.set(seed+1);
    mr_ = scinew MusilRNG(seed);
    (*mr_)();
    abort_=0;
    coreg = scinew 
      CoregPtsSimplexSearch(iters_.get(), misfitTol_.get(), abort_, dfieldP, 
			    *mr_, allowScale, allowRotate, allowTranslate);
  }
  coreg->setOrigPtsA(mobilePts);
  coreg->setOrigPtsP(fixedPts);
  coreg->getTrans(trans);
  double misfit;
  coreg->getMisfit(misfit);
//  Array1<Point> transformedPts;
//  coreg->getTransPtsA(transformedPts);
  //cerr << "Here's the misfit: "<<misfit<<"\n";
  DenseMatrix *dm = scinew DenseMatrix(trans);
  omat->send(MatrixHandle(dm));
}
//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void Coregister::tcl_command(GuiArgs& args, void* userdata) {
  if (args[1] == "stop") {
    abort_=1;
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace SCIRun
