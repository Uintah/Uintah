/*
 *  TetVolFieldCellToNode.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Containers/StringUtil.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <math.h>

#include <Core/share/share.h>

#include <vector>
#include <iostream>

namespace SCIRun {

using namespace std;

class PSECORESHARE TetVolFieldCellToNode : public Module {
private:
  
  Field          *infield_;
  FieldHandle    inhandle_;
  FieldIPort     *inport_;

  Field          *outfield_;
  FieldHandle    outhandle_;
  FieldOPort     *outport_;

public:
  TetVolFieldCellToNode(GuiContext* ctx);

  virtual ~TetVolFieldCellToNode();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};

  DECLARE_MAKER(TetVolFieldCellToNode)

TetVolFieldCellToNode::TetVolFieldCellToNode(GuiContext* ctx)
  : Module("TetVolFieldCellToNode", ctx, Source, "Fields", "SCIRun")
{
}

TetVolFieldCellToNode::~TetVolFieldCellToNode(){
}

void TetVolFieldCellToNode::execute()
{
  vector<Vector> vector_sums;
  vector<double> mag_sums;
  vector<int>    ref_counts;

  // must find ports and have valid data on inputs
  inport_ = (FieldIPort*)get_iport("Cell centered volume");

  if (!inport_)
  {
    error("Unable to initialize iport 'Cell centered volume'.");
    return;
  }
  if (!inport_->get(inhandle_) || 
      !(infield_ = inhandle_.get_rep()))
  {
    return;
  }

  outport_ = (FieldOPort*)get_oport("Node centered volume");
  if (!outport_)
  {
    error("Unable to initialize oport 'Node centered volume'.");
    return;
  }

  // we expect that the input field is a TetVolField<Vector>
  if (inhandle_.get_rep()->get_type_description()->get_name() !=
      "TetVolField<Vector>" )
  {
    error("Cell centered volume is not a TetVolField of vectors.");
    return;
  }                     

  if (infield_->data_at() != Field::CELL)
  {
    error("Cell centered volume is not cell centered.");
    return;
  }
                        
  TetVolField<Vector> *tvf = (TetVolField<Vector>*)(inhandle_.get_rep());
  TetVolMesh *mesh = (TetVolMesh*)(tvf->get_typed_mesh().get_rep());

  TetVolMesh::Node::size_type nsize;
  TetVolMesh::Cell::size_type csize;
  mesh->size(nsize);
  mesh->size(csize);
  const unsigned int nodes_size = csize;
  const unsigned int cells_size = nsize;

  vector_sums.resize(nodes_size,0);
  mag_sums.resize(nodes_size,0);
  ref_counts.resize(nodes_size,0);

  gui->execute(id + " set_state Executing 0");

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
      vector_sums[index] += tvf->value(*ci);
      mag_sums[index] += (tvf->value(*ci)).length2();

    }

    gui->execute(id + " set_progress " + to_string(count/cells_size) + " 0");
  }

  TetVolField<Vector> *newfield = scinew TetVolField<Vector>(mesh,Field::NODE);

  TetVolField<Vector>::fdata_type &fdata = newfield->fdata();

  Vector curvec;
  for (unsigned int loop=0;loop<nodes_size;++loop) {
    curvec = vector_sums[loop];
    double h = 1./(curvec.length2()/(mag_sums[loop]/ref_counts[loop]));
    fdata[loop] = curvec * h;
  }

  outport_->send(newfield);
}

void TetVolFieldCellToNode::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


