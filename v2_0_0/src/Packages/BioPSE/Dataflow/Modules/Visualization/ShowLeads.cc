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
#include <Packages/BioPSE/share/share.h>

namespace BioPSE {

using namespace SCIRun;

class BioPSESHARE ShowLeads : public Module {
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
  if (!iport_) {
    error("Unable to initialize iport 'Potentials'.");
    return;
  }
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
  PropertyManager *pm = mh.get_rep();
  string units;
  double start;
  double end;
  if (pm && pm->get_property(string("time-units"), units)) {
    units_.set(units.c_str());
  }  
  if (pm && pm->get_property(string("time-start"), start)) {
    tmin_.set(start);
  }  
  if (pm && pm->get_property(string("time-end"), end)) {
    tmax_.set(end);
  }  

  ostringstream cmmd;
  cmmd << id << " draw_leads";
  gui->execute(cmmd.str().c_str());
}

} // End namespace BioPSE


