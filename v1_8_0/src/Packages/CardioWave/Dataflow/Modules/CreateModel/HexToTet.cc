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
 *  HexToTet.cc:  Clip out the portions of a HexVol with specified values
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


namespace CardioWave {

using namespace SCIRun;

class HexToTet : public Module {
private:
  unsigned int last_generation_;
  FieldHandle tvfieldH;

public:

  //! Constructor/Destructor
  HexToTet(GuiContext *context);
  virtual ~HexToTet();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexToTet)


HexToTet::HexToTet(GuiContext *context) : 
  Module("HexToTet", context, Filter, "CreateModel", "CardioWave"),
  last_generation_(0)
{
}


HexToTet::~HexToTet()
{
}



void
HexToTet::execute()
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

  FieldOPort *ofp = (FieldOPort *)get_oport("TetVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }

  // Cache generation.
  if (hvfield->generation == last_generation_)
  {
    ofp->send(tvfieldH);
    return;
  }
  last_generation_ = hvfield->generation;

  HexVolMeshHandle hvmesh = hvfield->get_typed_mesh();
  TetVolMeshHandle tvmesh = scinew TetVolMesh();

  // Copy points directly, assuming they will have the same order.
  HexVolMesh::Node::iterator nbi, nei;
  hvmesh->begin(nbi); hvmesh->end(nei);
  while (nbi != nei)
  {
    Point p;
    hvmesh->get_center(p, *nbi);
    tvmesh->add_point(p);
    ++nbi;
  }

  hvmesh->synchronize(Mesh::NODE_NEIGHBORS_E);

  vector<HexVolMesh::Elem::index_type> elemmap;

  HexVolMesh::Elem::size_type hesize; hvmesh->size(hesize);
  vector<bool> visited(hesize, false);

  HexVolMesh::Elem::iterator bi, ei;
  hvmesh->begin(bi); hvmesh->end(ei);

  const unsigned int surfsize = pow(hesize, 2.0 / 3.0);
  vector<HexVolMesh::Elem::index_type> buffers[2];
  buffers[0].reserve(surfsize);
  buffers[1].reserve(surfsize);
  bool flipflop = true;
  hvmesh->synchronize(Mesh::FACES_E);

  while (bi != ei)
  {
    if (!visited[(unsigned int)*bi])
    {
      buffers[flipflop].clear();
      buffers[flipflop].push_back(*bi);

      while (buffers[flipflop].size() > 0)
      {
	for (unsigned int i = 0; i < buffers[flipflop].size(); i++)
	{
	  if (visited[(unsigned int)buffers[flipflop][i]]) { continue; }
	  visited[(unsigned int)buffers[flipflop][i]] = true;

	  HexVolMesh::Node::array_type hvnodes;
	  hvmesh->get_nodes(hvnodes, buffers[flipflop][i]);

	  if (flipflop)
	  {
	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));
	  }
	  else
	  {
	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));
	  }

	  elemmap.push_back(buffers[flipflop][i]);

	  HexVolMesh::Cell::array_type neighbors;
	  hvmesh->get_neighbors(neighbors, buffers[flipflop][i]);

	  for (unsigned int i = 0; i < neighbors.size(); i++)
	  {
	    if (!visited[(unsigned int)neighbors[i]])
	    {
	      buffers[!flipflop].push_back(neighbors[i]);
	    }
	  }
	}
	buffers[flipflop].clear();
	flipflop = !flipflop;
      }
    }
    ++bi;
  }


  TetVolField<int> *tvfield = scinew TetVolField<int>(tvmesh, Field::CELL);
  *(PropertyManager *)tvfield = *(PropertyManager *)hvfield;

  int val;
  for (unsigned int i = 0; i < elemmap.size(); i++)
  {
    hvfield->value(val, elemmap[i]);
    tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+0));
    tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+1));
    tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+2));
    tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+3));
    tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+4));
  }

  // Forward the results.
  tvfieldH = tvfield;
  ofp->send(tvfieldH);
}


} // End namespace CardioWave
