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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Algorithms/Geometry/CoregPts.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Dataflow/GuiInterface/GuiVar.h>
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
    allowScale_(get_ctx()->subVar("allowScale"), 1),
    allowRotate_(get_ctx()->subVar("allowRotate"), 1),
    allowTranslate_(get_ctx()->subVar("allowTranslate"), 1),
    seed_(get_ctx()->subVar("seed"), 1),
    iters_(get_ctx()->subVar("iters"), 1000),
    misfitTol_(get_ctx()->subVar("misfitTol"), 0.001),
    method_(get_ctx()->subVar("method")),
    mr_(0)
{
}


Coregister::~Coregister()
{
  if (mr_) { delete mr_; mr_ = 0; }
}


void
Coregister::execute()
{
  FieldIPort *fixed = (FieldIPort *) get_iport("Fixed PointCloudField");
  FieldIPort *mobile = (FieldIPort *) get_iport("Mobile PointCloudField");
  FieldIPort *dfield = (FieldIPort *) get_iport("DistanceField From Fixed");

  typedef ConstantBasis<double>                FDCdoubleBasis;
  typedef PointCloudMesh<ConstantBasis<Point> > PCMesh;
  typedef GenericField<PCMesh, FDCdoubleBasis, vector<double> >PCField;

  FieldHandle fixedH, mobileH;
  PCField *fixedPC, *mobilePC;
  PCMesh::handle_type fixedM, mobileM;
  PCMesh::Node::size_type nnodes;
  
  if (!fixed->get(fixedH)) return;
  fixedPC = dynamic_cast<PCField *>(fixedH.get_rep());
  if (!fixedPC) return;

  fixedM = fixedPC->get_typed_mesh();

  if (!mobile->get(mobileH)) return;
  mobilePC = dynamic_cast<PCField *>(mobileH.get_rep());
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

  vector<Point> fixedPts, mobilePts;
  Transform trans;

  PCMesh::Node::iterator fni, fne, mni, mne;
  fixedM->begin(fni); fixedM->end(fne);
  mobileM->begin(mni); mobileM->end(mne);
  Point p;
  while (fni != fne) {
    fixedM->get_center(p, *fni);
    fixedPts.push_back(p);
    ++fni;
  }
  while (mni != mne) {
    mobileM->get_center(p, *mni);
    mobilePts.push_back(p);
    ++mni;
  }

  int allowScale = allowScale_.get();
  int allowRotate = allowRotate_.get();
  int allowTranslate = allowTranslate_.get();
  string method = method_.get();
  
  CoregPts* coreg = 0;
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
    if (mr_) { delete mr_; }
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

  MatrixHandle dm(scinew DenseMatrix(trans));
  send_output_handle("Transform", dm);
}


//! Commands invoked from the Gui.  Pause/unpause/stop the search.

void
Coregister::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "stop") {
    abort_ = 1;
  } else {
    Module::tcl_command(args, userdata);
  }
}
} // End namespace SCIRun
