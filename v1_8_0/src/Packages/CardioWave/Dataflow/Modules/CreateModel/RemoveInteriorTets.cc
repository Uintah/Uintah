/*
 *  RemoveInteriorTets.cc:
 *
 *   Written by:
 *   Joe Tranquillo
 *   Duke University 
 *   Biomedical Engineering Department
 *   August 2001
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Containers/StringUtil.h>

extern "C" {
#include <Packages/CardioWave/Core/Algorithms/Vulcan.h>
}

#include <Packages/CardioWave/share/share.h>

namespace CardioWave {

using namespace SCIRun;

class CardioWaveSHARE RemoveInteriorTets : public Module {
  GuiString	threshold_;
public:
  RemoveInteriorTets(GuiContext *context);
  virtual ~RemoveInteriorTets();
  virtual void execute();
};


DECLARE_MAKER(RemoveInteriorTets)


RemoveInteriorTets::RemoveInteriorTets(GuiContext *context)
  : Module("RemoveInteriorTets", context, Source, "CreateModel", "CardioWave"),
    threshold_(context->subVar("threshold"))
{
}

RemoveInteriorTets::~RemoveInteriorTets(){
}

void RemoveInteriorTets::execute()
{
  double threshold = atof(threshold_.get().c_str());

  // must find ports and have valid data on inputs
  FieldIPort *imesh = (FieldIPort*)get_iport("TetsIn");
  if (!imesh) {
    error("Unable to initialize iport 'TetsIn'.");
    return;
  }
  
  FieldOPort *omesh = (FieldOPort*)get_oport("TetsOut");
  if (!omesh) {
    error("Unable to initialize oport 'TetsOut'.");
    return;
  }

  FieldHandle meshH;
  if (!imesh->get(meshH) || 
      !meshH.get_rep())
    return;

  TetVolField<Vector> *tv_old = dynamic_cast<TetVolField<Vector> *>(meshH.get_rep());
  if (!tv_old)
  {
    error("Input field wasn't a TetVolField<Vector>.");
    return;
  }
  
  if (tv_old->data_at() != Field::NODE && tv_old->data_at() != Field::NONE)
  {
    error("Must be at the nodes in order for us to remove elements.");
    return;
  }

  TetVolMeshHandle old_mesh = tv_old->get_typed_mesh();

  TetVolMesh::Node::array_type nodes;
  Point centroid, p;
  TetVolMesh::Node::size_type nnodes;
  old_mesh->size(nnodes);
  TetVolMesh::Cell::size_type ncells;
  old_mesh->size(ncells);

  vector<bool> cell_valid(ncells, true);
  vector<bool> node_valid(nnodes, false);

  // find the tets with centroid far from their nodes
  TetVolMesh::Cell::iterator cb, ce; old_mesh->begin(cb); old_mesh->end(ce);
  int i;
  while (cb!=ce) {
    old_mesh->get_nodes(nodes, *cb);
    old_mesh->get_center(centroid, *cb);
    for (i=0; i<4; i++) {
      old_mesh->get_center(p, nodes[i]);
      if ((p-centroid).length() > threshold) {
	cell_valid[*cb]=false;
      }
    }
    ++cb;
  }

  // node which nodes are now unconnected
  old_mesh->begin(cb);
  while(cb!=ce) {
    if (cell_valid[*cb]) {
      old_mesh->get_nodes(nodes, *cb);
      for (i=0; i<4; i++) {
	node_valid[nodes[i]]=true;
      }
    }
    ++cb;
  }

  TetVolMesh *new_mesh = scinew TetVolMesh;

  // add the remaining nodes to a new mesh and make a map of old-to-new 
  //   node indices
  int count=0;
  vector<int> node_map(nnodes, -1);
  TetVolMesh::Node::iterator nb, ne; old_mesh->begin(nb); old_mesh->end(ne);
  while(nb!=ne) {
    if (node_valid[*nb]) {
      old_mesh->get_center(p, *nb);
      new_mesh->add_point(p);
      node_map[*nb]=count;
      count++;
    }
    ++nb;
  }

  if (count != nnodes) {
    warning("Threshold was too large -- " + to_string(nnodes-count) + 
	    " of " + to_string(nnodes) + " were removed.");
  }

  // add the valid tets to the new mesh
  count=0;
  old_mesh->begin(cb);
  while(cb!=ce) {
    if (cell_valid[*cb]) {
      old_mesh->get_nodes(nodes, *cb);
      new_mesh->add_tet(node_map[nodes[0]], node_map[nodes[1]],
			node_map[nodes[2]], node_map[nodes[3]]);
      count++;
    }
    ++cb;
  }
  msgStream_ << "RemoveInteriorTets: ncells="<<count<<"\n";

  // copy the fdata for valid nodes
  TetVolField<Vector> *tv_new = scinew TetVolField<Vector>(new_mesh, Field::NODE);
  count=0;
  old_mesh->begin(nb);
  while(nb!=ne) {
    if (node_valid[*nb]) {
      tv_new->fdata()[count] = tv_old->fdata()[*nb];
      count++;
    }
    ++nb;
  }
  omesh->send(tv_new);
}
} // End namespace CardioWave
