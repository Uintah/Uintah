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
 *  HexMeshCuthillMcKee.cc:  Clip out the portions of a HexVol with specified values
 *
 *  Written by:
 *   Michael Callahan
 *   University of Utah
 *   May 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/HexVolField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <vector>
#include <algorithm>
#include <deque>
#include <queue>

namespace CardioWave {

using std::priority_queue;

using namespace SCIRun;

class HexMeshCuthillMcKee : public Module {
private:
  unsigned int last_generation_;
  FieldHandle bwfieldH;

public:

  //! Constructor/Destructor
  HexMeshCuthillMcKee(GuiContext *context);
  virtual ~HexMeshCuthillMcKee();

  //! Public methods
  virtual void execute();
};


class pair_greater
{
public:
  bool operator()(const pair<int, HexVolMesh::Node::index_type> &a,
		  const pair<int, HexVolMesh::Node::index_type> &b)
  {
    return a.first > b.first;
  }
};

DECLARE_MAKER(HexMeshCuthillMcKee)


HexMeshCuthillMcKee::HexMeshCuthillMcKee(GuiContext *context) : 
  Module("HexMeshCuthillMcKee", context, Filter, "CreateModel", "CardioWave"),
  last_generation_(0)
{
}


HexMeshCuthillMcKee::~HexMeshCuthillMcKee()
{
}



void
HexMeshCuthillMcKee::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("HexVol");
  if (!ifieldport) {
    error("Unable to initialize iport 'HexVol'.");
    return;
  }
  FieldHandle ifieldhandle;
  if(!(ifieldport->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Can't get field.");
    return;
  }

  HexVolField<int> *hvfield =
    dynamic_cast<HexVolField<int> *>(ifieldhandle.get_rep());

  if (hvfield == 0)
  {
    error("'" + ifieldhandle->get_type_description()->get_name() + "'" +
	  " field type is unsupported.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Bandwidth Minimized HexVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }

  // cache generation
  if (hvfield->generation == last_generation_)
  {
    ofp->send(bwfieldH);
    return;
  }
  last_generation_ = hvfield->generation;

  HexVolMeshHandle hvmesh = hvfield->get_typed_mesh();
  HexVolMeshHandle bwmesh = scinew HexVolMesh();

  HexVolMesh::Node::size_type nnodes;
  hvmesh->size(nnodes);

  vector<int> node_get_new_idx(nnodes); // get new node idx from old node idx
  vector<int> node_get_old_idx(nnodes); // get old node idx from new node idx
  int curr_new_idx=nnodes-1;            // reverse Cuthill McKee
  vector<vector<int> > node_nbrs(nnodes); // nbr list for each old node idx
  vector<int> visited(nnodes, 0);         // set to 1 when we visit a node

  // in our priority queue, we store a priority and an (old node) index
  typedef pair<int, HexVolMesh::Node::index_type> np_type;

  // internally store the queue as a vector, and pop lower priority nodes 1st
  typedef std::priority_queue<np_type, vector<np_type>, pair_greater> pq_type;

  pq_type *curr_queue = scinew pq_type;
  pq_type *next_queue = scinew pq_type;
  pq_type *swap_queue;

  HexVolMesh::Node::iterator nbi, nei;
  hvmesh->begin(nbi); hvmesh->end(nei);

  hvmesh->synchronize(Mesh::NODE_NEIGHBORS_E);

  int max_half_bw=0;
  int bw;

  // pre-process -- build up neighbor lists
  while (nbi != nei) {
    HexVolMesh::Node::array_type neighbors;
    hvmesh->get_neighbors(neighbors, *nbi);
    for (unsigned int i = 0; i < neighbors.size(); i++) {
      bw = abs(*nbi - (int)neighbors[i]);
      if (bw > max_half_bw) max_half_bw = bw;
      node_nbrs[*nbi].push_back((int)neighbors[i]);
    }
    ++nbi;
  }

  msgStream_ << "Bandwidth before re-ordering: "<<max_half_bw*2+1<<"\n";

  // visit each separate component
  while(curr_new_idx != -1) {

    // find the least-connected, non-visited node
    HexVolMesh::Node::index_type fewest_nbrs_idx;
    int fewest_nbrs_num=-1;
    hvmesh->begin(nbi);
    while (nbi != nei) {
      if (!visited[*nbi] && 
	  (fewest_nbrs_num == -1 || node_nbrs[*nbi].size()<fewest_nbrs_num)) {
	fewest_nbrs_num = (int)(node_nbrs[*nbi].size());
	fewest_nbrs_idx = *nbi;
      }
      ++nbi;
    }
    ASSERT(fewest_nbrs_num != -1);

    // breadth-first-search to assign re-ordering of nodes
    visited[fewest_nbrs_idx]=1;
    next_queue->push(np_type(fewest_nbrs_num, fewest_nbrs_idx));

    // if there's another level, swap queues and visit new nodes
    while (!next_queue->empty()) {
      swap_queue = next_queue;
      next_queue = curr_queue;
      curr_queue = swap_queue;
      
      // for all the nodes on this level, push their unvisited neighbors onto 
      //   the next_queue
      while (!curr_queue->empty()) {
	HexVolMesh::Node::index_type curr_old_idx = curr_queue->top().second;
	node_get_new_idx[curr_old_idx] = curr_new_idx;
	node_get_old_idx[curr_new_idx] = curr_old_idx;
	curr_new_idx--;
	curr_queue->pop();

	// update progress for each percent
	if ((int)(((nnodes-curr_new_idx*1.)/nnodes)*100) !=
	    (int)(((nnodes-curr_new_idx+1.)/nnodes)*100))
	  update_progress((nnodes-curr_new_idx*1.)/nnodes);

	for (unsigned int i = 0; i < node_nbrs[curr_old_idx].size(); i++) {
	  unsigned int nbr_idx = (unsigned int)node_nbrs[curr_old_idx][i];
	  if (!visited[nbr_idx]) {
	    visited[nbr_idx] = 1;
	    next_queue->push(np_type((int)(node_nbrs[nbr_idx].size()),
				 (HexVolMesh::Node::index_type)nbr_idx));
	  }
	}
      }
    }
  }
  delete curr_queue;
  delete next_queue;

  int ii=0;
  for (ii=0; ii<nnodes; ii++) {
    if (!visited[ii]) {
      error("Somehow not all of the nodes were visited!");
      ASSERTFAIL("not all nodes were visited");
    }
  }

  for (ii=0; ii<nnodes; ii++)
    visited[node_get_old_idx[ii]]++;
  
  for (ii=0; ii<nnodes; ii++)
    if (visited[ii] != 2) {
      error("Bad visited list.");
      ASSERTFAIL("Bad place to be.\n");
    }

  for (ii=0; ii<nnodes; ii++) 
    visited[node_get_new_idx[ii]]++;
  
  for (ii=0; ii<nnodes; ii++)
    if (visited[ii] != 3) {
      error("Bat Visited list.");
      ASSERTFAIL("Bad place to be.\n");
    }

  // build up the nodes for the new mesh
  hvmesh->begin(nbi);
  while (nbi != nei) {
    Point p;
    hvmesh->get_center(p,(HexVolMesh::Node::index_type)node_get_old_idx[*nbi]);
    bwmesh->add_point(p);
    ++nbi;
  }

  // build up the elements for the new mesh
  HexVolMesh::Cell::iterator cbi, cei;
  hvmesh->begin(cbi);
  hvmesh->end(cei);
  while (cbi != cei) {
    HexVolMesh::Node::array_type onodes;
    hvmesh->get_nodes(onodes, *cbi);
    HexVolMesh::Node::array_type nnodes(onodes.size());
    for (unsigned int i = 0; i < onodes.size(); i++)
      nnodes[i] = node_get_new_idx[onodes[i]];
    bwmesh->add_elem(nnodes);
    ++cbi;
  }

  HexVolField<int> *bwfield = scinew HexVolField<int>(bwmesh, Field::NODE);
  hvmesh->begin(nbi);
  while (nbi != nei) {
    int val;
    hvfield->value(val, *nbi);
    bwfield->set_value(val,
		       (HexVolMesh::Node::index_type)node_get_new_idx[*nbi]);
    ++nbi;
  }

  bwmesh->synchronize(Mesh::NODE_NEIGHBORS_E);

  int new_max_half_bw=0;
  int new_bw;
  bwmesh->begin(nbi);
  bwmesh->end(nei);
  while(nbi != nei) {
    HexVolMesh::Node::array_type neighbors;
    bwmesh->get_neighbors(neighbors, *nbi);
    for (unsigned int i = 0; i < neighbors.size(); i++) {
      new_bw = abs(*nbi - (int)neighbors[i]);
      if (new_bw > new_max_half_bw) new_max_half_bw = new_bw;
    }    
    ++nbi;
  }

  msgStream_ << "Bandwidth after re-ordering: "<<new_max_half_bw*2+1<<"\n";

  *(PropertyManager *)bwfield = *(PropertyManager *)hvfield;
  bwfieldH = bwfield;
  ofp->send(bwfieldH);
}


} // End namespace CardioWave
