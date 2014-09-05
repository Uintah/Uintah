/*
 *  TetVolCellToNode.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVol.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE TetVolCellToNode : public Module {
public:
  
  Field          *infield_;
  FieldHandle    inhandle_;
  FieldIPort     *inport_;
  TetVol<Vector> *vf_;

  Field          *outfield_;
  FieldHandle    outhandle_;
  FieldOPort     *outport_;

public:
  TetVolCellToNode(const clString& id);

  virtual ~TetVolCellToNode();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECORESHARE Module* make_TetVolCellToNode(const clString& id) {
  return scinew TetVolCellToNode(id);
}

TetVolCellToNode::TetVolCellToNode(const clString& id)
  : Module("TetVolCellToNode", id, Source, "Fields", "SCIRun")
{
}

TetVolCellToNode::~TetVolCellToNode(){
}

void TetVolCellToNode::execute()
{
  typedef TetVolMesh::cell_iterator cell_iterator;
  typedef TetVolMesh::node_array    node_array;
  typedef node_array::iterator      node_iterator;

  vector<Vector> vector_sums;
  vector<double> mag_sums;
  vector<int>    ref_counts;

  // must find ports and have valid data on inputs
  if (!(inport_=(FieldIPort*)get_iport("Cell centered volume")) ||
      !inport_->get(inhandle_) || 
      !(infield_ = inhandle_.get_rep()))
    return;

  if (!(outport_=(FieldOPort*)get_oport("Node centered volume")))
    return;

  // we expect that the input field is a TetVol<Vector>
  if (infield_->get_type_name() != "TetVol<Vector>") {
    postMessage("TetVolCellToNode: ERROR: Cell centered volume is not a "
		"TetVol of Vectors.  Exiting.");
    return;
  }                     

  if (infield_->data_at() != Field::CELL) {
    postMessage("TetVolCellToNode: ERROR: Cell centered volume is not "
		"cell centered.  Exiting.");
    return;
  }                         

  vf_ = (TetVol<Vector>*)infield_;

  TetVolMesh *mesh = 
    dynamic_cast<TetVolMesh*>(vf_->get_typed_mesh().get_rep());

  vf_->finish_mesh();

  int cells_size = mesh->cells_size();
  int nodes_size = mesh->nodes_size();

  vector_sums.resize(nodes_size,0);
  mag_sums.resize(nodes_size,0);
  ref_counts.resize(nodes_size,0);

  TCL::execute(id + " set_state Executing 0");

  cell_iterator ci;
  node_iterator ni;
  node_array    na;
  int           index;
  float         count = 0;

  for (ci = mesh->cell_begin();
       ci != mesh->cell_end();
       ++ci,++count) {
    mesh->get_nodes(na,*ci);
    for (ni = na.begin();
	 ni != na.end();
	 ++ni) {

      index = *ni;

      ref_counts[index] += 1;
      vector_sums[index] += vf_->value(*ci);
      mag_sums[index] += (vf_->value(*ci)).length2();

    }

    TCL::execute(id + " set_progress " + to_string(count/cells_size) + " 0");
  }

  TetVol<Vector> *newfield = scinew TetVol<Vector>(mesh,Field::NODE);

  TetVol<Vector>::fdata_type &fdata = newfield->fdata();

  Vector curvec;
  for (int loop=0;loop<nodes_size;++loop) {
    curvec = vector_sums[loop];
    double h = 1./(curvec.length2()/(mag_sums[loop]/ref_counts[loop]));
    fdata[loop] = curvec * h;
  }

  outport_->send(newfield);
}

void TetVolCellToNode::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


