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


namespace CardioWave {

using std::cerr;
using std::endl;

using namespace SCIRun;

class HexIntMask : public Module {
  
public:
  GuiString gui_exclude_;
  unsigned int last_generation_;
  string last_gui_exclude_;
  
  //! Constructor/Destructor
  HexIntMask(GuiContext *context);
  virtual ~HexIntMask();

  //! Public methods
  virtual void execute();

  void parse_exclude_list(const string &guistr, vector<int> &exclude);
};


DECLARE_MAKER(HexIntMask)


HexIntMask::HexIntMask(GuiContext *context) : 
  Module("HexIntMask", context, Filter, "CreateModel", "CardioWave"),
  gui_exclude_(context->subVar("exclude")),
  last_generation_(0)
{
}


HexIntMask::~HexIntMask()
{
}


void
HexIntMask::parse_exclude_list(const string &guistr, vector<int> &exclude)
{
  // Test values, to be taken from GUI later.
  const string str = guistr + " ";
  bool skipping_p = true;
  int last = 0;
  for (unsigned int i = 0; i < str.size(); i++)
  {
    if (skipping_p)
    {
      if (str[i] >= '0' && str[i] <= '9')
      {
	last = i;
	skipping_p = false;
      }
    }
    if (!skipping_p)
    {
      if (str[i] < '0' || str[i] > '9')
      {
	const string val = str.substr(last, i-last);
	int v = atoi(val.c_str());
	exclude.push_back(v);
	skipping_p = true;
      }
    }
  }
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

  // Cache generation.
  if (hvfield->generation == last_generation_ &&
      gui_exclude_.get() == last_gui_exclude_)
  {
    return;
  }
  last_generation_ = hvfield->generation;
  last_gui_exclude_ = gui_exclude_.get();

  vector<int> exclude;
  parse_exclude_list(gui_exclude_.get(), exclude);

  for (unsigned int i = 0; i < exclude.size(); i++)
  {
    cout << "\nExcluding " << i << "  " << exclude[i] << "\n";
  }
  cout << "\n";


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

  vector<HexVolMesh::Elem::index_type> elemmap;

  HexVolMesh::Elem::iterator bi, ei;
  hvmesh->begin(bi);
  hvmesh->end(ei);
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
      elemmap.push_back(*bi);
    }

    ++bi;
  }
  clipped->flush_changes();

  HexVolField<int> *ofield = 0;
  if (elemmap.size() > 0)
  {
    ofield = scinew HexVolField<int>(clipped, Field::CELL);
    *(PropertyManager *)ofield = *(PropertyManager *)hvfield;

    int val;
    for (unsigned int i = 0; i < elemmap.size(); i++)
    {
      hvfield->value(val, elemmap[i]);
      ofield->set_value(val, (HexVolMesh::Elem::index_type)i);
    }
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


} // End namespace CardioWave
