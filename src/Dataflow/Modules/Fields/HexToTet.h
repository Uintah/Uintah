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
 *  HexToTet.h:  Convert a Hex field into a Tet field using 1-5 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#if !defined(HexToTet_h)
#define HexToTet_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

class HexToTetAlgo : public DynamicAlgoBase
{
public:
  virtual bool execute(FieldHandle, FieldHandle&, std::ostream &) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *data_td);
};


template <class FSRC>
class HexToTetAlgoT : public HexToTetAlgo
{
public:
  //! virtual interface. 
  virtual bool execute(FieldHandle src, FieldHandle& dst, std::ostream &msg);
};


template <class FSRC>
bool
HexToTetAlgoT<FSRC>::execute(FieldHandle srcH, FieldHandle& dstH, 
			     std::ostream &msg)
{
  FSRC *hvfield = dynamic_cast<FSRC*>(srcH.get_rep());

  typename FSRC::mesh_type *hvmesh = hvfield->get_typed_mesh().get_rep();
  TetVolMeshHandle tvmesh = scinew TetVolMesh();

  // Copy points directly, assuming they will have the same order.
  typename FSRC::mesh_type::Node::iterator nbi, nei;
  hvmesh->begin(nbi); hvmesh->end(nei);
  while (nbi != nei)
  {
    Point p;
    hvmesh->get_center(p, *nbi);
    tvmesh->add_point(p);
    ++nbi;
  }

  hvmesh->synchronize(Mesh::NODE_NEIGHBORS_E);

  vector<typename FSRC::mesh_type::Elem::index_type> elemmap;

  typename FSRC::mesh_type::Node::size_type hnsize; hvmesh->size(hnsize);
  typename FSRC::mesh_type::Elem::size_type hesize; hvmesh->size(hesize);

  vector<bool> visited(hesize, false);

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  hvmesh->begin(bi); hvmesh->end(ei);

  const unsigned int surfsize = (unsigned int)pow(hesize, 2.0 / 3.0);
  vector<typename FSRC::mesh_type::Elem::index_type> buffers[2];
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

	  typename FSRC::mesh_type::Node::array_type hvnodes;
	  hvmesh->get_nodes(hvnodes, buffers[flipflop][i]);
	  ASSERT(hvnodes.size() == 8);
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
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));
	  }
	  else
	  {
	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[0]),
			    (TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[2]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[1]),
			    (TetVolMesh::Node::index_type)(hvnodes[5]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]));

	    tvmesh->add_tet((TetVolMesh::Node::index_type)(hvnodes[3]),
			    (TetVolMesh::Node::index_type)(hvnodes[4]),
			    (TetVolMesh::Node::index_type)(hvnodes[6]),
			    (TetVolMesh::Node::index_type)(hvnodes[7]));
	  }

	  elemmap.push_back(buffers[flipflop][i]);

	  typename FSRC::mesh_type::Cell::array_type neighbors;
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
  
  TetVolField<typename FSRC::value_type> *tvfield = 
    scinew TetVolField<typename FSRC::value_type>(tvmesh, hvfield->data_at());
  *(PropertyManager *)tvfield = *(PropertyManager *)hvfield;
  dstH = tvfield;

  typename FSRC::value_type val;

  if (hvfield->data_at() == Field::NODE) {
    for (int i = 0; i < hnsize; i++)
    {
      hvfield->value(val, (typename FSRC::mesh_type::Node::index_type)(i));
      tvfield->set_value(val, (TetVolMesh::Node::index_type)(i));
    }
  } else if (hvfield->data_at() == Field::CELL) {
    for (unsigned int i = 0; i < elemmap.size(); i++)
    {
      hvfield->value(val, elemmap[i]);
      tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+0));
      tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+1));
      tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+2));
      tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+3));
      tvfield->set_value(val, (TetVolMesh::Elem::index_type)(i*5+4));
    }
  } else {
    msg << "Warning: did not load data values, use DirectInterp" << endl;
  }
  return true;
}


class LatToTetAlgo : public DynamicAlgoBase
{
public:
  virtual bool execute(FieldHandle, FieldHandle&, std::ostream &) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *data_td);
};


template <class FSRC>
class LatToTetAlgoT : public LatToTetAlgo
{
public:
  //! virtual interface. 
  virtual bool execute(FieldHandle src, FieldHandle& dst, std::ostream &msg);
};


template <class FSRC>
bool
LatToTetAlgoT<FSRC>::execute(FieldHandle srcH, FieldHandle& dstH, 
			     std::ostream &msg)
{
  FSRC *hvfield = dynamic_cast<FSRC*>(srcH.get_rep());

  typename FSRC::mesh_type *hvmesh = hvfield->get_typed_mesh().get_rep();
  TetVolMeshHandle tvmesh = scinew TetVolMesh();

  // Copy points directly, assuming they will have the same order.
  typename FSRC::mesh_type::Node::iterator nbi, nei;
  hvmesh->begin(nbi); hvmesh->end(nei);
  while (nbi != nei)
  {
    Point p;
    hvmesh->get_center(p, *nbi);
    tvmesh->add_point(p);
    ++nbi;
  }

  hvmesh->synchronize(Mesh::NODE_NEIGHBORS_E);
  hvmesh->synchronize(Mesh::FACES_E);

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  hvmesh->begin(bi); hvmesh->end(ei);
  while (bi != ei)
  {
    typename FSRC::mesh_type::Node::array_type hvnodes(8);
    hvmesh->get_nodes(hvnodes, *bi);
    if (!(((*bi).i_ ^ (*bi).j_ ^ (*bi).k_)&1))
    {
      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[0]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[1]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[2]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[5]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[0]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[2]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[3]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[7]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[0]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[5]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[2]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[7]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[0]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[5]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[7]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[4]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[5]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[2]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[7]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[6]));
    }
    else
    {
      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[0]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[1]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[3]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[4]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[1]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[2]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[3]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[6]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[1]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[3]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[4]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[6]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[1]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[5]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[6]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[4]));

      tvmesh->add_tet((TetVolMesh::Node::index_type)((unsigned int)hvnodes[3]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[4]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[6]),
		      (TetVolMesh::Node::index_type)((unsigned int)hvnodes[7]));
    }
    ++bi;
  }
  
  TetVolField<typename FSRC::value_type> *tvfield = 
    scinew TetVolField<typename FSRC::value_type>(tvmesh, hvfield->data_at());
  *(PropertyManager *)tvfield = *(PropertyManager *)hvfield;
  dstH = tvfield;

  typename FSRC::value_type val;

  if (hvfield->data_at() == Field::NODE) {
    hvmesh->begin(nbi); hvmesh->end(nei);
    while (nbi != nei)
    {
      hvfield->value(val, *nbi);
      tvfield->set_value(val,
			 (TetVolMesh::Node::index_type)(unsigned int)(*nbi));
      ++nbi;
    }
  }
  else if (hvfield->data_at() == Field::CELL)
  {
    typename FSRC::mesh_type::Cell::iterator cbi, cei;    
    hvmesh->begin(cbi); hvmesh->end(cei);
    while (cbi != cei)
    {
      hvfield->value(val, *cbi);
      unsigned int i = (unsigned int)*cbi;
      tvfield->set_value(val, (TetVolMesh::Cell::index_type)(i*5+0));
      tvfield->set_value(val, (TetVolMesh::Cell::index_type)(i*5+1));
      tvfield->set_value(val, (TetVolMesh::Cell::index_type)(i*5+2));
      tvfield->set_value(val, (TetVolMesh::Cell::index_type)(i*5+3));
      tvfield->set_value(val, (TetVolMesh::Cell::index_type)(i*5+4));
      ++cbi;
    }
  } else {
    msg << "Warning: did not load data values, use DirectInterp" << endl;
  }
  return true;
}

}

#endif
