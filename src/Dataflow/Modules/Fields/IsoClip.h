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

//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/Clipper.h>
#include <sci_hash_map.h>
#include <algorithm>

namespace SCIRun {


class GuiInterface;

class IsoClipAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);

protected:
  static int tet_permute_table[15][4];
  static int tri_permute_table[7][3];
};


template <class FIELD>
class IsoClipAlgoTet : public IsoClipAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte);
private:

  struct upairhash
  {
    unsigned int operator()(const pair<unsigned int, unsigned int> &a) const
    {
      hash<unsigned int> h;
      return h(a.first ^ a.second);
    }
  };

  struct utripple
  {
    unsigned int first, second, third;
  };

  struct utripplehash
  {
    unsigned int operator()(const utripple &a) const
    {
      hash<unsigned int> h;
      return h(a.first ^ a.second ^ a.third);
    }
  };

  struct utrippleequal
  {
    unsigned int operator()(const utripple &a, const utripple &b) const
    {
      return a.first == b.first && a.second == b.second && a.third == b.third;
    }
  };
  

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type,
		   hash<unsigned int>,
		   equal_to<unsigned int> > node_hash_type;

  typedef hash_map<pair<unsigned int, unsigned int>,
		   typename FIELD::mesh_type::Node::index_type,
		   upairhash,
		   equal_to<pair<unsigned int, unsigned int> > > edge_hash_type;

  typedef hash_map<utripple,
		   typename FIELD::mesh_type::Node::index_type,
		   utripplehash,
		   utrippleequal> face_hash_type;
#else
  typedef map<unsigned int,
	      typename FIELD::mesh_type::Node::index_type,
	      equal_to<unsigned int> > node_hash_type;

  typedef map<pair<unsigned int, unsigned int>,
	      typename FIELD::mesh_type::Node::index_type,
	      equal_to<pair<unsigned int, unsigned int> > > edge_hash_type;

  typedef map<utripple,
	      typename FIELD::mesh_type::Node::index_type,
	      utrippleequal> face_hash_type;
#endif

  typename FIELD::mesh_type::Node::index_type
  edge_lookup(unsigned int u0, unsigned int u1,
	      const Point &p, edge_hash_type &edgemap,
	      typename FIELD::mesh_type *clipped) const;

  typename FIELD::mesh_type::Node::index_type
  face_lookup(unsigned int u0, unsigned int u1, unsigned int u2,
	      const Point &p, face_hash_type &facemap,
	      typename FIELD::mesh_type *clipped) const;
};


template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTet<FIELD>::edge_lookup(unsigned int u0, unsigned int u1,
				   const Point &p, edge_hash_type &edgemap,
				   typename FIELD::mesh_type *clipped) const
{
  pair<unsigned int, unsigned int> np;
  if (u0 < u1)  { np.first = u0; np.second = u1; }
  else { np.first = u1; np.second = u0; }
  if (edgemap.find(np) == edgemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    edgemap[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return edgemap[np];
  }
}


template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTet<FIELD>::face_lookup(unsigned int u0, unsigned int u1,
				   unsigned int u2, const Point &p,
				   face_hash_type &facemap,
				   typename FIELD::mesh_type *clipped) const
{
  utripple nt;
  if (u0 < u1)
  {
    if (u2 < u0)
    {
      nt.first = u2; nt.second = u0; nt.third = u1;
    }
    else if (u2 < u1)
    {
      nt.first = u0; nt.second = u2; nt.third = u1;
    }
    else
    {
      nt.first = u0; nt.second = u1; nt.third = u2;
    }
  }
  else
  {
    if (u2 > u0)
    {
      nt.first = u1; nt.second = u0; nt.third = u2;
    }
    else if (u2 > u1)
    {
      nt.first = u1; nt.second = u2; nt.third = u0;
    }
    else
    {
      nt.first = u2; nt.second = u1; nt.third = u0;
    }
  }
  if (facemap.find(nt) == facemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    facemap[nt] = nodeindex;
    return nodeindex;
  }
  else
  {
    return facemap[nt];
  }
}


template <class FIELD>
FieldHandle
IsoClipAlgoTet<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh,
			       double isoval, bool lte)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  *(PropertyManager *)clipped = *(PropertyManager *)mesh;

  node_hash_type nodemap;
  edge_hash_type edgemap;
  face_hash_type facemap;

  typename FIELD::mesh_type::Node::array_type onodes(4);
  typename FIELD::value_type v[4];
  Point p[4];

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);

    unsigned int inside = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      mesh->get_center(p[i], onodes[i]);
      field->value(v[i], onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
	inside |= 1;
      }
    }

    if (lte) { inside = ~inside & 0xf; }

    if (inside == 0)
    {
      // Discard outside elements.
    }
    else if (inside == 0xf)
    {
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[i]);
	  nodemap[(unsigned int)onodes[i]] = nodeindex;
	  nnodes[i] = nodeindex;
	}
	else
	{
	  nnodes[i] = nodemap[(unsigned int)onodes[i]];
	}
      }

      clipped->add_elem(nnodes);
    }
    else if (inside == 0x8 || inside == 0x4 || inside == 0x2 || inside == 0x1)
    {
      // Lop off 3 points and add resulting tet to the new mesh.
      const int *perm = tet_permute_table[inside];
      typename FIELD::mesh_type::Node::array_type nnodes(4);

      if (nodemap.find((unsigned int)onodes[perm[0]]) == nodemap.end())
      {
	const typename FIELD::mesh_type::Node::index_type nodeindex =
	  clipped->add_point(p[perm[0]]);
	nodemap[(unsigned int)onodes[perm[0]]] = nodeindex;
	nnodes[0] = nodeindex;
      }
      else
      {
	nnodes[0] = nodemap[(unsigned int)onodes[perm[0]]];
      }

      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]],
				   imv / (v[perm[3]] - v[perm[0]]));

      nnodes[1] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      l1, edgemap, clipped);

      nnodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      l2, edgemap, clipped);

      nnodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      l3, edgemap, clipped);

      clipped->add_elem(nnodes);
    }
    else if (inside == 0x7 || inside == 0xb || inside == 0xd || inside == 0xe)
    {
      // Lop off 1 point, break up the resulting quads and add the
      // resulting tets to the mesh.
      const int *perm = tet_permute_table[inside];
      typename FIELD::mesh_type::Node::array_type nnodes(4);

      typename FIELD::mesh_type::Node::index_type inodes[9];
      for (unsigned int i = 1; i < 4; i++)
      {
	if (nodemap.find((unsigned int)onodes[perm[i]]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[perm[i]]);
	  nodemap[(unsigned int)onodes[perm[i]]] = nodeindex;
	  inodes[i-1] = nodeindex;
	}
	else
	{
	  inodes[i-1] = nodemap[(unsigned int)onodes[perm[i]]];
	}
      }

      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]],
				   imv / (v[perm[3]] - v[perm[0]]));

      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      l1, edgemap, clipped);

      inodes[4] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      l2, edgemap, clipped);

      inodes[5] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      l3, edgemap, clipped);

      const Point c1 = Interpolate(l1, l2, 0.5);
      const Point c2 = Interpolate(l2, l3, 0.5);
      const Point c3 = Interpolate(l3, l1, 0.5);

      inodes[6] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      c1, facemap, clipped);
      inodes[7] = face_lookup((unsigned int)onodes[perm[2]],
			      (unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      c2, facemap, clipped);
      inodes[8] = face_lookup((unsigned int)onodes[perm[3]],
			      (unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      c3, facemap, clipped);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[3];
      nnodes[2] = inodes[8];
      nnodes[3] = inodes[6];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[1];
      nnodes[1] = inodes[4];
      nnodes[2] = inodes[6];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[2];
      nnodes[1] = inodes[5];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[8];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[6];
      nnodes[2] = inodes[8];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[8];
      nnodes[2] = inodes[2];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[6];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[1];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[1];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[2];
      clipped->add_elem(nnodes);
    }
    else// if (inside == 0x3 || inside == 0x5 || inside == 0x6 ||
    	//     inside == 0x9 || inside == 0xa || inside == 0xc)
    {
      // Lop off two points, break the resulting quads, then add the
      // new tets to the mesh.
      const int *perm = tet_permute_table[inside];
      typename FIELD::mesh_type::Node::array_type nnodes(4);

      typename FIELD::mesh_type::Node::index_type inodes[8];
      for (unsigned int i = 2; i < 4; i++)
      {
	if (nodemap.find((unsigned int)onodes[perm[i]]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[perm[i]]);
	  nodemap[(unsigned int)onodes[perm[i]]] = nodeindex;
	  inodes[i-2] = nodeindex;
	}
	else
	{
	  inodes[i-2] = nodemap[(unsigned int)onodes[perm[i]]];
	}
      }
      const double imv0 = isoval - v[perm[0]];
      const Point l02 = Interpolate(p[perm[0]], p[perm[2]],
				    imv0 / (v[perm[2]] - v[perm[0]]));
      const Point l03 = Interpolate(p[perm[0]], p[perm[3]],
				    imv0 / (v[perm[3]] - v[perm[0]]));

      const double imv1 = isoval - v[perm[1]];
      const Point l12 = Interpolate(p[perm[1]], p[perm[2]],
				    imv1 / (v[perm[2]] - v[perm[1]]));
      const Point l13 = Interpolate(p[perm[1]], p[perm[3]],
				    imv1 / (v[perm[3]] - v[perm[1]]));

      inodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      l02, edgemap, clipped);
      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      l03, edgemap, clipped);
      inodes[4] = edge_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      l12, edgemap, clipped);
      inodes[5] = edge_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[3]],
			      l13, edgemap, clipped);

      const Point c1 = Interpolate(l02, l03, 0.5);
      const Point c2 = Interpolate(l12, l13, 0.5);

      inodes[6] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
      			      (unsigned int)onodes[perm[3]],
      			      c1, facemap, clipped);
      inodes[7] = face_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      (unsigned int)onodes[perm[3]],
      			      c2, facemap, clipped);

      nnodes[0] = inodes[7];
      nnodes[1] = inodes[2];
      nnodes[2] = inodes[0];
      nnodes[3] = inodes[4];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[1];
      nnodes[1] = inodes[5];
      nnodes[2] = inodes[3];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[1];
      nnodes[1] = inodes[3];
      nnodes[2] = inodes[6];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[7];
      nnodes[2] = inodes[6];
      nnodes[3] = inodes[2];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[1];
      nnodes[2] = inodes[7];
      nnodes[3] = inodes[6];
      clipped->add_elem(nnodes);
    }

    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped, fieldh->data_at());
  *(PropertyManager *)ofield = *(PropertyManager *)(fieldh.get_rep());

  typename node_hash_type::iterator nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    typename FIELD::value_type val;
    field->value(val, (index_type)((*nmitr).first));
    ofield->set_value(val, (index_type)((*nmitr).second));

    ++nmitr;
  }

  typename edge_hash_type::iterator emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*emitr).second));

    ++emitr;
  }

  typename face_hash_type::iterator fmitr = facemap.begin();
  while (fmitr != facemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*fmitr).second));

    ++fmitr;
  }

  return ofield;
}



template <class FIELD>
class IsoClipAlgoTri : public IsoClipAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte);
private:

  struct upairhash
  {
    unsigned int operator()(const pair<unsigned int, unsigned int> &a) const
    {
      hash<unsigned int> h;
      return h(a.first ^ a.second);
    }
  };

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type,
		   hash<unsigned int>,
		   equal_to<unsigned int> > node_hash_type;

  typedef hash_map<pair<unsigned int, unsigned int>,
		   typename FIELD::mesh_type::Node::index_type,
		   upairhash,
		   equal_to<pair<unsigned int, unsigned int> > > edge_hash_type;
#else
  typedef map<unsigned int,
	      typename FIELD::mesh_type::Node::index_type,
	      equal_to<unsigned int> > node_hash_type;

  typedef map<pair<unsigned int, unsigned int>,
	      typename FIELD::mesh_type::Node::index_type,
	      equal_to<pair<unsigned int, unsigned int> > > edge_hash_type;

  typedef map<utripple,
	      typename FIELD::mesh_type::Node::index_type,
	      utrippleequal> face_hash_type;
#endif

  typename FIELD::mesh_type::Node::index_type
  edge_lookup(unsigned int u0, unsigned int u1,
	      const Point &p, edge_hash_type &edgemap,
	      typename FIELD::mesh_type *clipped) const;
};


template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTri<FIELD>::edge_lookup(unsigned int u0, unsigned int u1,
				   const Point &p, edge_hash_type &edgemap,
				   typename FIELD::mesh_type *clipped) const
{
  pair<unsigned int, unsigned int> np;
  if (u0 < u1)  { np.first = u0; np.second = u1; }
  else { np.first = u1; np.second = u0; }
  if (edgemap.find(np) == edgemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    edgemap[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return edgemap[np];
  }
}



template <class FIELD>
FieldHandle
IsoClipAlgoTri<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh,
			       double isoval, bool lte)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  *(PropertyManager *)clipped = *(PropertyManager *)mesh;

  node_hash_type nodemap;
  edge_hash_type edgemap;

  typename FIELD::mesh_type::Node::array_type onodes(3);
  typename FIELD::value_type v[3];
  Point p[3];

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);

    unsigned int inside = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      mesh->get_center(p[i], onodes[i]);
      field->value(v[i], onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
	inside |= 1;
      }
    }

    if (lte) { inside = ~inside & 0x7; }

    if (inside == 0)
    {
      // Discard outside elements.
    }
    else if (inside == 0x7)
    {
      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
	if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
	{
	  const typename FIELD::mesh_type::Node::index_type nodeindex =
	    clipped->add_point(p[i]);
	  nodemap[(unsigned int)onodes[i]] = nodeindex;
	  nnodes[i] = nodeindex;
	}
	else
	{
	  nnodes[i] = nodemap[(unsigned int)onodes[i]];
	}
      }

      clipped->add_elem(nnodes);
    }
    else if (inside == 0x1 || inside == 0x2 || inside == 0x4)
    {
      const int *perm = tri_permute_table[inside];
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());
      if (nodemap.find((unsigned int)onodes[perm[0]]) == nodemap.end())
      {
	const typename FIELD::mesh_type::Node::index_type nodeindex =
	  clipped->add_point(p[perm[0]]);
	nodemap[(unsigned int)onodes[perm[0]]] = nodeindex;
	nnodes[0] = nodeindex;
      }
      else
      {
	nnodes[0] = nodemap[(unsigned int)onodes[perm[0]]];
      }

      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));

      nnodes[1] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      l1, edgemap, clipped);

      nnodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      l2, edgemap, clipped);

      clipped->add_elem(nnodes);
    }
    else
    {
      const int *perm = tri_permute_table[inside];
      typename FIELD::mesh_type::Node::array_type inodes(4);
      if (nodemap.find((unsigned int)onodes[perm[1]]) == nodemap.end())
      {
	const typename FIELD::mesh_type::Node::index_type nodeindex =
	  clipped->add_point(p[perm[1]]);
	nodemap[(unsigned int)onodes[perm[1]]] = nodeindex;
	inodes[0] = nodeindex;
      }
      else
      {
	inodes[0] = nodemap[(unsigned int)onodes[perm[1]]];
      }

      if (nodemap.find((unsigned int)onodes[perm[2]]) == nodemap.end())
      {
	const typename FIELD::mesh_type::Node::index_type nodeindex =
	  clipped->add_point(p[perm[2]]);
	nodemap[(unsigned int)onodes[perm[2]]] = nodeindex;
	inodes[1] = nodeindex;
      }
      else
      {
	inodes[1] = nodemap[(unsigned int)onodes[perm[2]]];
      }

      const double imv = isoval - v[perm[0]];
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]],
				   imv / (v[perm[1]] - v[perm[0]]));
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]],
				   imv / (v[perm[2]] - v[perm[0]]));

      inodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      l1, edgemap, clipped);

      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      l2, edgemap, clipped);

      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[1];
      nnodes[2] = inodes[3];
      clipped->add_elem(nnodes);

      nnodes[0] = inodes[0];
      nnodes[1] = inodes[3];
      nnodes[2] = inodes[2];
      clipped->add_elem(nnodes);
    }
    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped, fieldh->data_at());
  *(PropertyManager *)ofield = *(PropertyManager *)(fieldh.get_rep());

  typename node_hash_type::iterator nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    typename FIELD::value_type val;
    field->value(val, (index_type)((*nmitr).first));
    ofield->set_value(val, (index_type)((*nmitr).second));

    ++nmitr;
  }

  typename edge_hash_type::iterator emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*emitr).second));

    ++emitr;
  }

  return ofield;
}



} // end namespace SCIRun

#endif // ClipField_h
