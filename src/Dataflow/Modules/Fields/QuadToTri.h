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
 *  QuadToTri.h:  Convert a Quad field into a Tri field using 1-2 split
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   December 2002
 *
 *  Copyright (C) 1994, 2001 SCI Group
 */

#if !defined(QuadToTri_h)
#define QuadToTri_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Datatypes/TriSurfField.h>
#include <Core/Util/Assert.h>

namespace SCIRun {

class QuadToTriAlgo : public DynamicAlgoBase
{
public:
  virtual bool execute(FieldHandle, FieldHandle&, ProgressReporter *) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *data_td);
};


template <class FSRC>
class QuadToTriAlgoT : public QuadToTriAlgo
{
public:
  //! virtual interface. 
  virtual bool execute(FieldHandle src, FieldHandle& dst, ProgressReporter *);
};


template <class FSRC>
bool
QuadToTriAlgoT<FSRC>::execute(FieldHandle srcH, FieldHandle& dstH,
			      ProgressReporter *mod)
{
  FSRC *qsfield = dynamic_cast<FSRC*>(srcH.get_rep());

  typename FSRC::mesh_type *qsmesh = qsfield->get_typed_mesh().get_rep();
  TriSurfMeshHandle tsmesh = scinew TriSurfMesh();

  // Copy points directly, assuming they will have the same order.
  typename FSRC::mesh_type::Node::iterator nbi, nei;
  qsmesh->begin(nbi); qsmesh->end(nei);
  while (nbi != nei)
  {
    Point p;
    qsmesh->get_center(p, *nbi);
    tsmesh->add_point(p);
    ++nbi;
  }

  qsmesh->synchronize(Mesh::NODE_NEIGHBORS_E);

  vector<typename FSRC::mesh_type::Elem::index_type> elemmap;

  typename FSRC::mesh_type::Node::size_type hnsize; 
  qsmesh->size(hnsize);
  typename FSRC::mesh_type::Elem::size_type hesize; 
  qsmesh->size(hesize);

  vector<bool> visited(hesize, false);

  typename FSRC::mesh_type::Elem::iterator bi, ei;
  qsmesh->begin(bi); qsmesh->end(ei);

  const unsigned int surfsize = (unsigned int)pow(hesize, 2.0 / 3.0);
  vector<typename FSRC::mesh_type::Elem::index_type> buffers[2];
  buffers[0].reserve(surfsize);
  buffers[1].reserve(surfsize);
  bool flipflop = true;
  qsmesh->synchronize(Mesh::EDGES_E);

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

	  typename FSRC::mesh_type::Node::array_type qsnodes;
	  qsmesh->get_nodes(qsnodes, buffers[flipflop][i]);
	  ASSERT(qsnodes.size() == 4);

	  if (flipflop)
	  {
	    tsmesh->add_triangle((TriSurfMesh::Node::index_type)(qsnodes[0]),
				 (TriSurfMesh::Node::index_type)(qsnodes[1]),
				 (TriSurfMesh::Node::index_type)(qsnodes[2]));

	    tsmesh->add_triangle((TriSurfMesh::Node::index_type)(qsnodes[0]),
				 (TriSurfMesh::Node::index_type)(qsnodes[2]),
				 (TriSurfMesh::Node::index_type)(qsnodes[3]));
	  }
	  else
	  {
	    tsmesh->add_triangle((TriSurfMesh::Node::index_type)(qsnodes[0]),
				 (TriSurfMesh::Node::index_type)(qsnodes[1]),
				 (TriSurfMesh::Node::index_type)(qsnodes[3]));

	    tsmesh->add_triangle((TriSurfMesh::Node::index_type)(qsnodes[1]),
				 (TriSurfMesh::Node::index_type)(qsnodes[2]),
				 (TriSurfMesh::Node::index_type)(qsnodes[3]));
	  }

	  elemmap.push_back(buffers[flipflop][i]);
	  qsmesh->synchronize(Mesh::EDGE_NEIGHBORS_E);
	  typename FSRC::mesh_type::Face::array_type neighbors;
	  qsmesh->get_neighbors(neighbors, buffers[flipflop][i]);

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
  
  TriSurfField<typename FSRC::value_type> *tvfield = 
    scinew TriSurfField<typename FSRC::value_type>(tsmesh, qsfield->data_at());
  *(PropertyManager *)tvfield = *(PropertyManager *)qsfield;
  dstH = tvfield;

  typename FSRC::value_type val;

  if (qsfield->data_at() == Field::NODE) {

    unsigned int i = 0;
    typename FSRC::fdata_type dat = qsfield->fdata();
    typename FSRC::fdata_type::iterator iter = dat.begin();
    while (iter != dat.end()) {
      val = *iter;
      tvfield->set_value(val, (TriSurfMesh::Node::index_type)(i));
      ++iter; ++i;
    }
  } else if (qsfield->data_at() == Field::FACE) {
    for (unsigned int i = 0; i < elemmap.size(); i++)
    {
      qsfield->value(val, elemmap[i]);
      tvfield->set_value(val, (TriSurfMesh::Elem::index_type)(i*2+0));
      tvfield->set_value(val, (TriSurfMesh::Elem::index_type)(i*2+1));
    }
  } else {
    mod->warning("Could not load data values, use DirectInterp if needed.");
  }
  return true;
}



class ImgToTriAlgo : public DynamicAlgoBase
{
public:
  virtual bool execute(FieldHandle, FieldHandle&, ProgressReporter *) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *data_td);
};


template <class FSRC>
class ImgToTriAlgoT : public ImgToTriAlgo
{
public:
  //! virtual interface. 
  virtual bool execute(FieldHandle src, FieldHandle& dst, ProgressReporter *m);
};


template <class FSRC>
bool
ImgToTriAlgoT<FSRC>::execute(FieldHandle srcH, FieldHandle& dstH, 
			     ProgressReporter *mod)
{
  FSRC *ifield = dynamic_cast<FSRC*>(srcH.get_rep());

  typename FSRC::mesh_type *imesh = ifield->get_typed_mesh().get_rep();
  TriSurfMeshHandle tmesh = scinew TriSurfMesh();

  // Copy points directly, assuming they will have the same order.
  typename FSRC::mesh_type::Node::iterator nbi, nei;
  imesh->begin(nbi); imesh->end(nei);
  while (nbi != nei)
  {
    Point p;
    imesh->get_center(p, *nbi);
    tmesh->add_point(p);
    ++nbi;
  }

  typedef TriSurfMesh::Node::index_type nindex_type;
  
  typename FSRC::mesh_type::Elem::iterator bi, ei;
  imesh->begin(bi); imesh->end(ei);
  while (bi != ei)
  {
    typename FSRC::mesh_type::Node::array_type inodes;
    imesh->get_nodes(inodes, *bi);
    if (((*bi).i_ ^ (*bi).j_)&1)
    {
      tmesh->add_triangle(nindex_type((unsigned int)inodes[0]),
			  nindex_type((unsigned int)inodes[1]),
			  nindex_type((unsigned int)inodes[2]));

      tmesh->add_triangle(nindex_type((unsigned int)inodes[0]),
			  nindex_type((unsigned int)inodes[2]),
			  nindex_type((unsigned int)inodes[3]));
    }
    else
    {
      tmesh->add_triangle(nindex_type((unsigned int)inodes[0]),
			  nindex_type((unsigned int)inodes[1]),
			  nindex_type((unsigned int)inodes[3]));

      tmesh->add_triangle(nindex_type((unsigned int)inodes[1]),
			  nindex_type((unsigned int)inodes[2]),
			  nindex_type((unsigned int)inodes[3]));
    }
    ++bi;
  }
  
  TriSurfField<typename FSRC::value_type> *tfield = 
    scinew TriSurfField<typename FSRC::value_type>(tmesh, ifield->data_at());
  *(PropertyManager *)tfield = *(PropertyManager *)ifield;
  dstH = tfield;

  typename FSRC::value_type val;

  if (ifield->data_at() == Field::NODE)
  {
    imesh->begin(nbi); imesh->end(nei);
    while (nbi != nei)
    {
      ifield->value(val, *nbi);
      tfield->set_value(val,
			(TriSurfMesh::Node::index_type)(unsigned int)(*nbi));
      ++nbi;
    }
  }
  else if (ifield->data_at() == Field::FACE)
  {
    typename FSRC::mesh_type::Cell::iterator cbi, cei;    
    imesh->begin(cbi); imesh->end(cei);
    while (cbi != cei)
    {
      ifield->value(val, *cbi);
      unsigned int i = (unsigned int)*cbi;
      tfield->set_value(val, (TriSurfMesh::Cell::index_type)(i*2+0));
      tfield->set_value(val, (TriSurfMesh::Cell::index_type)(i*2+1));
      ++cbi;
    }
  }
  else
  {
    mod->warning("Could not load data values, use DirectInterp if needed.");
  }
  return true;
}


}

#endif
