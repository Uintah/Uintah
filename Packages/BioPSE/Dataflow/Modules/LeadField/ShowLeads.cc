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
  ShowLeads(const string& id);

  virtual ~ShowLeads();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
private:
  MatrixIPort *iport_;

  GuiString units_;
  GuiDouble tmin_;
  GuiDouble tmax_;
  
  int gen_;
};

extern "C" BioPSESHARE Module* make_ShowLeads(const string& id) {
  return scinew ShowLeads(id);
}

ShowLeads::ShowLeads(const string& id) : 
  Module("ShowLeads", id, Source, "LeadField", "BioPSE"),
  iport_(0),
  units_("time-units", id, this),
  tmin_("time-min", id, this),
  tmax_("time-max", id, this),
  gen_(-1)
{
}

ShowLeads::~ShowLeads(){
}

void ShowLeads::execute(){

  iport_ = (MatrixIPort *)get_iport("Matrix");
  if (!iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  MatrixHandle mh;
  if (! iport_->get(mh)) {
    return;
  }
  if (mh->generation == gen_) { return; }
  gen_ = mh->generation;

  DenseMatrix *dm = dynamic_cast<DenseMatrix*>(mh.get_rep());
  if (!dm) return;

  int rows = dm->nrows();
  int cols = dm->ncols();
  
  // test GUI first 
  
  for(int i = 0; i < rows; i++) {
    ostringstream cmmd;
    ostringstream xstr;
    ostringstream ystr;
    xstr << "{";
    ystr << " {";
    cmmd << id << " add_lead " << i << " ";
    for (int j = 0; j < cols; j++) {
      xstr << j/(float)cols << " "; // the time inc we are at
      ystr << dm->get(i, j) << " "; // the value at i
    }
    xstr << "}";
    ystr << "}";
    cmmd << xstr.str();
    cmmd << ystr.str();
    TCL::execute(cmmd.str().c_str());
  }

  // set params from properties before drawing leads
  PropertyManager *pm = mh.get_rep();
  string units;
  double start;
  double end;
  if (pm && pm->get(string("time-units"), units)) {
    units_.set(units.c_str());
  }  
  if (pm && pm->get(string("time-start"), start)) {
    tmin_.set(start);
  }  
  if (pm && pm->get(string("time-end"), end)) {
    tmax_.set(end);
  }  

  ostringstream cmmd;
  cmmd << id << " draw_leads";
  TCL::execute(cmmd.str().c_str());
}

void ShowLeads::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace BioPSE


