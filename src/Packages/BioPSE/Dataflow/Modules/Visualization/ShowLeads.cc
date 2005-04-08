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
 *  ShowLeads.cc:
 *
 *  Written by:
 *   mcole
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/GuiInterface/GuiVar.h>

namespace BioPSE {

using namespace SCIRun;

class ShowLeads : public Module {
public:
  ShowLeads(GuiContext *context);

  virtual ~ShowLeads();

  virtual void execute();

private:
  MatrixIPort *iport_;

  GuiString units_;
  GuiDouble tmin_;
  GuiDouble tmax_;
  
  int gen_;
};


DECLARE_MAKER(ShowLeads)


ShowLeads::ShowLeads(GuiContext *context) : 
  Module("ShowLeads", context, Source, "Visualization", "BioPSE"),
  iport_(0),
  units_(context->subVar("time-units")),
  tmin_(context->subVar("time-min")),
  tmax_(context->subVar("time-max")),
  gen_(-1)
{
}

ShowLeads::~ShowLeads(){
}

void ShowLeads::execute(){

  iport_ = (MatrixIPort *)get_iport("Potentials");
  MatrixHandle mh;
  if (! iport_->get(mh)) {
    error("Cannot get matrix from input port.");
    return;
  }
  if (mh->generation == gen_) { 
    remark("Same input matrix, nothing changed.");
    return; 
  }
  gen_ = mh->generation;

  ostringstream clr;
  clr << id << " clear";
  gui->execute(clr.str().c_str());
  

  int rows = mh->nrows();
  int cols = mh->ncols();
  ostringstream set_mm;
  set_mm << id << " set_min_max_index 0 " << rows - 1; 
  gui->execute(set_mm.str().c_str());

  for(int i = 0; i < rows; i++) {
    ostringstream cmmd;
    ostringstream xstr;
    ostringstream ystr;
    xstr << "{";
    ystr << " {";
    cmmd << id << " add_lead " << i << " ";
    for (int j = 0; j < cols; j++) {
      xstr << j/(float)cols << " "; // the time inc we are at
      ystr << mh->get(i, j) << " "; // the value at i
    }
    xstr << "}";
    ystr << "}";
    cmmd << xstr.str();
    cmmd << ystr.str();
    gui->execute(cmmd.str().c_str());
  }

  // set params from properties before drawing leads
  string units;
  double start;
  double end;
  if (mh.get_rep() && mh->get_property(string("time-units"), units)) {
    units_.set(units.c_str());
  }  
  if (mh.get_rep() && mh->get_property(string("time-start"), start)) {
    tmin_.set(start);
  }  
  if (mh.get_rep() && mh->get_property(string("time-end"), end)) {
    tmax_.set(end);
  }  

  ostringstream cmmd;
  cmmd << id << " draw_leads";
  gui->execute(cmmd.str().c_str());
}

} // End namespace BioPSE


