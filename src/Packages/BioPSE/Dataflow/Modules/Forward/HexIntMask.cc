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
 *  HexIntMask.cc:  Clip out the portions of a HexVol with specified values
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
#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <vector>
#include <algorithm>


namespace BioPSE {

using std::cerr;
using std::endl;

using namespace SCIRun;

class HexIntMask : public Module {
  
public:
  
  //! Constructor/Destructor
  HexIntMask(GuiContext *context);
  virtual ~HexIntMask();

  //! Public methods
  virtual void execute();
};


DECLARE_MAKER(HexIntMask)


HexIntMask::HexIntMask(GuiContext *context) : 
  Module("HexIntMask", context, Filter, "Forward", "BioPSE")
{
}


HexIntMask::~HexIntMask()
{
}


void
HexIntMask::execute()
{
  FieldIPort *ifieldport = (FieldIPort *)get_iport("Input HexVol");
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

  HexVolMeshHandle hvmesh = hvfield->get_typed_mesh();
  HexVolMeshHandle clipped = scinew HexVolMesh();

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

  hash_type nodemap;

  // Test values, to be taken from GUI later.
  vector<int> exclude(3);
  exclude[0] = 0;
  exclude[1] = 1;
  exclude[2] = 2;

  HexVolMesh::Elem::iterator bi, ei;
  hvmesh->begin(bi);
  hvmesh->end(ei);
  int counter = 0;
  while (bi != ei)
  {
    int val;
    hvfield->value(val, *bi);
    if (std::find(exclude.begin(), exclude.end(), val) == exclude.end())
    {
      HexVolMesh::Node::array_type onodes;
      hvmesh->get_nodes(onodes, *bi);
      HexVolMesh::Node::array_type nnodes(onodes.size());
      
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  Point np;
	  hvmesh->get_center(np, onodes[i]);
	  nodemap[(unsigned int)onodes[i]] = clipped->add_point(np);
	}
	nnodes[i] = nodemap[(unsigned int)onodes[i]];
      }

      clipped->add_elem(nnodes);
      counter++;
    }

    ++bi;
  }
  clipped->flush_changes();

  HexVolField<int> *ofield = 0;
  if (counter > 0)
  {
    ofield = scinew HexVolField<int>(clipped, Field::CELL);
    *(PropertyManager *)ofield = *(PropertyManager *)hvfield;
  }

  // Forward the results.
  FieldOPort *ofp = (FieldOPort *)get_oport("Masked HexVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }
  ofp->send(ofield);
}


} // End namespace BioPSE
