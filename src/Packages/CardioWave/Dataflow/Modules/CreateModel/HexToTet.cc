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

using std::cerr;
using std::endl;

using namespace SCIRun;

class HexToTet : public Module {
  
public:

  //! Constructor/Destructor
  HexToTet(GuiContext *context);
  virtual ~HexToTet();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexToTet)


HexToTet::HexToTet(GuiContext *context) : 
  Module("HexToTet", context, Filter, "CreateModel", "CardioWave")
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
    postMessage("Unable to initialize "+name+"'s iport\n");
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

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    HexVolMesh::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    HexVolMesh::Node::index_type,
    equal_to<unsigned int> > hash_type;
#endif

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

  hash_type nodemap;
  vector<HexVolMesh::Elem::index_type> elemmap;

  HexVolMesh::Elem::iterator bi, ei;
  hvmesh->begin(bi); hvmesh->end(ei);
  while (bi != ei)
  {
    HexVolMesh::Node::array_type hvnodes;
    hvmesh->get_nodes(hvnodes, *bi);

    
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

    elemmap.push_back(*bi);

    ++bi;
  }
  tvmesh->flush_changes();


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
  FieldOPort *ofp = (FieldOPort *)get_oport("TetVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }
  ofp->send(tvfield);
}


} // End namespace CardioWave
