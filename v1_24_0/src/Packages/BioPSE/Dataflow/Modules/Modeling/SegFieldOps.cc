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
 *  SegFieldOps.cc: Erosion/dilation/open/close/absorption
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/BioPSE/Core/Datatypes/SegLatVolField.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace BioPSE {

using namespace SCIRun;

class SegFieldOps : public Module
{
  string tclCmd_;
  FieldHandle origFldH_;
  SegLatVolField* origFld_;
  FieldHandle currFldH_;
  SegLatVolField* currFld_;
  GuiInt minCompSize_;
  int lastGen_;
public:
  SegFieldOps(GuiContext* ctx);
  virtual ~SegFieldOps();
  virtual void execute();
  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SegFieldOps)

SegFieldOps::SegFieldOps(GuiContext* ctx)
  : Module("SegFieldOps", ctx, Filter, "Modeling", "BioPSE"), tclCmd_(""),
    currFld_(0), minCompSize_(ctx->subVar("min_comp_size")), lastGen_(0)
{
}



SegFieldOps::~SegFieldOps()
{
}

void
SegFieldOps::execute()
{
  // Make sure the ports exist.
  FieldIPort *ifp = (FieldIPort *)get_iport("LatVol or SegField");
  FieldOPort *ofp = (FieldOPort *)get_oport("SegField");

  // Make sure the input data exists.
  FieldHandle ifieldH;
  if (!ifp->get(ifieldH) || !ifieldH.get_rep()) {
    error("No input data");
    return;
  }

  if (ifieldH->generation != lastGen_) 
    // new field -- doesn't matter what user requested, we're just gonna load
    //   this one
    tclCmd_ = "";
  else if (tclCmd_ == "") {
    // same field as last time, no new command -- just send same data (if we
    //   have any)
    if (currFldH_.get_rep())
      ofp->send(currFldH_);
    return;
  }

  // for caching
  lastGen_ = ifieldH->generation;

  if (tclCmd_ != "" && !currFld_) {
    // user wants us to do something, but we have no field... error and return
    error("Don't yet have a valid field for tcl command");
    tclCmd_ = "";
    return;
  }

  if (tclCmd_ != "") {
    // process user command
    if (tclCmd_ == "reset") {
      cerr << "reseting to original field\n";
      currFldH_ = origFldH_;
      currFld_ = origFld_;
      ofp->send(currFldH_);
    } else if (tclCmd_ == "absorb") {
      currFldH_.detach();
      currFld_ = dynamic_cast<SegLatVolField *>(currFldH_.get_rep());
      currFld_->absorbSmallComponents(minCompSize_.get());
      ofp->send(currFldH_);
    } else {
      error("Unknown command");
    }
    tclCmd_ = "";
    return;
  }

  LatVolField<int> *lvf;
  SegLatVolField *slvf = dynamic_cast<SegLatVolField *>(ifieldH.get_rep());
  if (!slvf) {
    lvf = dynamic_cast<LatVolField<int> *>(ifieldH.get_rep());
    if (!lvf) {
      error("SegFieldOps requires either a SegLatVolField or a LatVolField<int> (data at cells) as input.");
      return;
    }
    if (lvf->basis_order() != 0) {
      error("Please move data_at to cells before invoking SegFieldOps");
      return;
    }
    slvf = new SegLatVolField(lvf->get_typed_mesh());
    slvf->setData(lvf->fdata());
  } else {
    ifieldH.detach();
    slvf = dynamic_cast<SegLatVolField *>(ifieldH.get_rep());
  }

  // save local copies of data
  currFldH_ = slvf;
  currFld_ = slvf;
  origFldH_ = slvf;
  origFld_ = slvf;
  ofp->send(currFldH_);
}


void
SegFieldOps::tcl_command(GuiArgs& args, void* userdata)
{
  if (args.count() < 2) {
    args.error("SegFieldOps needs a minor command");
    return;
  }
  if (args[1] == "print") {
    if (currFld_) {
      currFld_->printComponents();
    }
  } else if (args[1] == "audit") {
    if (currFld_) {
      currFld_->audit();
    }
  } else if (args[1] == "absorb") {
    tclCmd_ = "absorb";
    want_to_execute();
  } else if (args[1] == "reset") {
    tclCmd_ = "reset";
    want_to_execute();
  } else Module::tcl_command(args, userdata);
}
} // End namespace BioPSE

