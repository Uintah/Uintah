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

using namespace SCIRun;

class HexIntMask : public Module {
  
public:
  GuiString gui_exclude_;
  GuiInt gui_delete_nodes_;
  GuiInt gui_levels_;
  int last_generation_;
  string last_gui_exclude_;
  int last_gui_delete_nodes_;
  int last_gui_levels_;

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
  gui_delete_nodes_(context->subVar("delete-nodes")),
  gui_levels_(context->subVar("levels")),
  last_generation_(0),
  last_gui_delete_nodes_(-1),
  last_gui_levels_(-1)
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
      if ((str[i] >= '0' && str[i] <= '9') || str[i] == '-')
      {
	last = i;
	skipping_p = false;
      }
    }
    if (!skipping_p)
    {
      if ((str[i] < '0' || str[i] > '9') && str[i] != '-')
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
    error("Unable to initialize iport 'Input HexVol'.");
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
      gui_exclude_.get() == last_gui_exclude_ &&
      gui_delete_nodes_.get() == last_gui_delete_nodes_ &&
      gui_levels_.get() == last_gui_levels_) return;

  last_generation_ = hvfield->generation;
  last_gui_exclude_ = gui_exclude_.get();
  last_gui_delete_nodes_ = gui_delete_nodes_.get();
  last_gui_levels_ = gui_levels_.get();

  vector<int> exclude;
  parse_exclude_list(last_gui_exclude_, exclude);

#if 0
  for (unsigned int i = 0; i < exclude.size(); i++)
  {
    cout << "\nExcluding " << exclude[i] << "\n";
  }
  cout << "\n";
#endif

  HexVolMeshHandle hvmesh = hvfield->get_typed_mesh();
  HexVolMeshHandle clipped = scinew HexVolMesh();
  HexVolField<int> *ofield = 0;

  HexVolMesh::Cell::size_type ncells;
  hvmesh->size(ncells);
  HexVolMesh::Node::size_type nnodes;
  hvmesh->size(nnodes);
  HexVolMesh::Node::iterator bni, eni;
  HexVolMesh::Cell::iterator bci, eci;
  
  if (hvfield->data_at() == Field::CELL) {
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
    if (!last_gui_delete_nodes_) {
      hvmesh->begin(bni);
      hvmesh->end(eni);
      while (bni != eni) {
	Point p;
	hvmesh->get_center(p, *bni);
	nodemap[(unsigned int) *bni] = clipped->add_point(p);
	++bni;
      }
    }

    vector<bool> usednode(nnodes, false);
    vector<bool> usedcell(ncells, false);

    hvmesh->begin(bci);
    hvmesh->end(eci);
    while (bci != eci)
    {
      int val;
      hvfield->value(val, *bci);
      if (std::find(exclude.begin(), exclude.end(), val) == exclude.end())
      {
	usedcell[*bci]=true;
	HexVolMesh::Node::array_type onodes;
	hvmesh->get_nodes(onodes, *bci);
	unsigned int i;
	for (i = 0; i < onodes.size(); i++)
	  usednode[onodes[i]]=true;
      }
      ++bci;
    }

    vector<bool> usednodePlusHull(nnodes);
    vector<bool> usedcellPlusHull(ncells);

    for (int l=0; l<last_gui_levels_; l++) {
      unsigned int j;
      for (j=0; j<nnodes; j++) usednodePlusHull[j]=false;
      for (j=0; j<ncells; j++) usedcellPlusHull[j]=false;
      hvmesh->begin(bci);
      hvmesh->end(eci);
      while (bci != eci) {
	HexVolMesh::Node::array_type onodes;
	hvmesh->get_nodes(onodes, *bci);
	unsigned int i;
	for (i=0; i<onodes.size() && !usedcellPlusHull[*bci]; i++)
	  usedcellPlusHull[*bci] = usedcellPlusHull[*bci] || 
	    usednode[onodes[i]];
	if (usedcellPlusHull[*bci]) 
	  for (i=0; i<onodes.size(); i++)
	    usednodePlusHull[onodes[i]]=true;
	++bci;
      }
      usednode=usednodePlusHull;
      usedcell=usedcellPlusHull;
    }

    vector<HexVolMesh::Elem::index_type> elemmap;

    hvmesh->begin(bci);
    hvmesh->end(eci);
    while (bci != eci)
    {
      if (usedcell[*bci]) {
	HexVolMesh::Node::array_type onodes;
	hvmesh->get_nodes(onodes, *bci);
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
	elemmap.push_back(*bci);
      }
      ++bci;
    }
    
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
  } else {
    vector<HexVolMesh::Node::index_type> nodemap;
    vector<HexVolMesh::Node::index_type> invnodemap(nnodes);
    vector<bool> valid(nnodes, false);
    
    // find which nodes are valid
    hvmesh->begin(bni);
    hvmesh->end(eni);
    while (bni != eni) {
      int val;
      hvfield->value(val, *bni);
      if (std::find(exclude.begin(), exclude.end(), val) == exclude.end())
	valid[*bni] = true;
      ++bni;
    }
      

    // based on which nodes are valid, figure out which cells are used
    //   (a cell is only used if all of its nodes are valid), and
    //   based on which cells are used, figure out which nodes are
    //   actually used
    vector<bool> usednode(nnodes, false);
    vector<bool> usedcell(ncells, true);

    hvmesh->begin(bci);
    hvmesh->end(eci);
    while (bci != eci) {
      HexVolMesh::Node::array_type onodes;
      hvmesh->get_nodes(onodes, *bci);
      unsigned int i;
      for (i = 0; i < onodes.size() && usedcell[*bci]; i++)
	usedcell[*bci] = usedcell[*bci] && valid[onodes[i]];
      if (usedcell[*bci])
	for (i = 0; i < onodes.size(); i++)
	  usednode[onodes[i]]=true;
      ++bci;
    }

    vector<bool> usednodePlusHull(nnodes);
    vector<bool> usedcellPlusHull(ncells);

    for (int l=0; l<last_gui_levels_; l++) {
      unsigned int j;
      for (j=0; j<nnodes; j++) usednodePlusHull[j]=false;
      for (j=0; j<ncells; j++) usedcellPlusHull[j]=false;
      hvmesh->begin(bci);
      hvmesh->end(eci);
      while (bci != eci) {
	HexVolMesh::Node::array_type onodes;
	hvmesh->get_nodes(onodes, *bci);
	unsigned int i;
	for (i=0; i<onodes.size() && !usedcellPlusHull[*bci]; i++)
	  usedcellPlusHull[*bci] = usedcellPlusHull[*bci] || 
	    usednode[onodes[i]];
	if (usedcellPlusHull[*bci]) 
	  for (i=0; i<onodes.size(); i++)
	    usednodePlusHull[onodes[i]]=true;
	++bci;
      }
      usednode=usednodePlusHull;
      usedcell=usedcellPlusHull;
    }

    // now that we know which nodes will be used, add them into the mesh
    //   and build the mapping vectors to find which new nodes used to
    //   be which old nodes
    hvmesh->begin(bni);
    while (bni != eni) {
      if (usednode[*bni] || !last_gui_delete_nodes_) {
	Point np;
	hvmesh->get_center(np, *bni);
	clipped->add_point(np);
	invnodemap[*bni] = nodemap.size();
	nodemap.push_back(*bni);
      }
      ++bni;
    }

    // build the new cells with mapped node numbers
    hvmesh->begin(bci);
    while (bci != eci) {
      if (usedcell[*bci]) {
	HexVolMesh::Node::array_type onodes;
	hvmesh->get_nodes(onodes, *bci);
	HexVolMesh::Node::array_type nnodes(onodes.size());
	for (unsigned int i = 0; i < onodes.size(); i++)
	  nnodes[i] = invnodemap[onodes[i]];
	clipped->add_elem(nnodes);
      }
      ++bci;
    }

    
    // copy the field data - map it to the right nodes
    if (nodemap.size() > 0)
    {
      ofield = scinew HexVolField<int>(clipped, Field::NODE);
      *(PropertyManager *)ofield = *(PropertyManager *)hvfield;
      
      int val;
      for (unsigned int i = 0; i < nodemap.size(); i++)
      {
	hvfield->value(val, nodemap[i]);
	ofield->set_value(val, (HexVolMesh::Node::index_type)i);
      }
    }
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Masked HexVol");
  if (!ofp)
  {
    error("Unable to initialize " + name + "'s Output port.");
    return;
  }
  ofp->send(ofield);
}


} // End namespace CardioWave
