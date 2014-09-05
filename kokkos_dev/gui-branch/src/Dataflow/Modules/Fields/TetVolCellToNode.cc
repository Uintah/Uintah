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
#include <Core/Containers/StringUtil.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE TetVolCellToNode : public Module {
private:
  
  Field          *infield_;
  FieldHandle    inhandle_;
  FieldIPort     *inport_;
  TetVol<Vector> *vf_;

  Field          *outfield_;
  FieldHandle    outhandle_;
  FieldOPort     *outport_;

public:
  TetVolCellToNode(const string& id);

  virtual ~TetVolCellToNode();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" PSECORESHARE Module* make_TetVolCellToNode(const string& id) {
  return scinew TetVolCellToNode(id);
}

TetVolCellToNode::TetVolCellToNode(const string& id)
  : Module("TetVolCellToNode", id, Source, "Fields", "SCIRun")
{
}

TetVolCellToNode::~TetVolCellToNode(){
}

void TetVolCellToNode::execute()
{
  vector<Vector> vector_sums;
  vector<double> mag_sums;
  vector<int>    ref_counts;

  // must find ports and have valid data on inputs
  inport_ = (FieldIPort*)get_iport("Cell centered volume");

  if (!inport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!inport_->get(inhandle_) || 
      !(infield_ = inhandle_.get_rep()))
    return;

  outport_ = (FieldOPort*)get_oport("Node centered volume");
  if (!outport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  // we expect that the input field is a TetVol<Vector>
  if (infield_->get_type_description()->get_name() !=
      get_type_description((TetVol<Vector> *)0)->get_name())
  {
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

  TetVolMesh::Node::size_type nsize;
  TetVolMesh::Cell::size_type csize;
  mesh->size(nsize);
  mesh->size(csize);
  const unsigned int nodes_size = csize;
  const unsigned int cells_size = nsize;

  vector_sums.resize(nodes_size,0);
  mag_sums.resize(nodes_size,0);
  ref_counts.resize(nodes_size,0);

  tcl_execute(id + " set_state Executing 0");

  TetVolMesh::Cell::iterator ci, cie;
  TetVolMesh::Node::array_type::iterator ni;
  TetVolMesh::Node::array_type na;
  int           index;
  float         count = 0;

  mesh->begin(ci); mesh->end(cie);
  for (; ci != cie; ++ci,++count)
  {
    mesh->get_nodes(na,*ci);
    for (ni = na.begin();
	 ni != na.end();
	 ++ni) {

      index = *ni;

      ref_counts[index] += 1;
      vector_sums[index] += vf_->value(*ci);
      mag_sums[index] += (vf_->value(*ci)).length2();

    }

    tcl_execute(id + " set_progress " + to_string(count/cells_size) + " 0");
  }

  TetVol<Vector> *newfield = scinew TetVol<Vector>(mesh,Field::NODE);

  TetVol<Vector>::fdata_type &fdata = newfield->fdata();

  Vector curvec;
  for (unsigned int loop=0;loop<nodes_size;++loop) {
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


