/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/


//    File   : ClipField.h
//    Author : Michael Callahan
//    Date   : August 2001

#if !defined(ClipField_h)
#define ClipField_h

#include <Dataflow/Network/Module.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Dataflow/Modules/Fields/HexToTet.h>
#include <Dataflow/Modules/Fields/FieldBoundary.h>
#include <sci_hash_map.h>
#include <algorithm>
#include <set>

#include <Dataflow/Modules/Fields/share.h>

namespace SCIRun {
typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;
class GuiInterface;

class SCISHARE IsoClipAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant) = 0;

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
			      double isoval, bool lte,
			      MatrixHandle &interpolant);
private:

  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
    double dfirst;
  };

  struct edgepairequal
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b) const
    {
      return a.first == b.first && a.second == b.second;
    }
  };

  struct edgepairless
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b)
    {
      return less(a, b);
    }
    static bool less(const edgepair_t &a, const edgepair_t &b)
    {
      return a.first < b.first || a.first == b.first && a.second < b.second;
    }
  };

#ifdef HAVE_HASH_MAP
  struct edgepairhash
  {
    unsigned int operator()(const edgepair_t &a) const
    {
#if defined(__ECC) || defined(_MSC_VER)
      hash_compare<unsigned int> h;
#else
      hash<unsigned int> h;
#endif
      return h(a.first ^ a.second);
    }
#if defined(__ECC) || defined(_MSC_VER)

      // These are particularly needed by ICC's hash stuff
      static const size_t bucket_size = 4;
      static const size_t min_buckets = 8;
      
      // This is a less than function.
      bool operator()(const edgepair_t & a, const edgepair_t & b) const {
        return edgepairless::less(a,b);
      }
#endif // endif ifdef __ICC
  };
#endif

  struct facetriple_t
  {
    unsigned int first, second, third;
    double dsecond, dthird;
  };

  struct facetripleequal
  {
    bool operator()(const facetriple_t &a, const facetriple_t &b) const
    {
      return a.first == b.first && a.second == b.second && a.third == b.third;
    }
  };

  struct facetripleless
  {
    bool operator()(const facetriple_t &a, const facetriple_t &b) const
    {
      return a.first < b.first || a.first == b.first && ( a.second < b.second || a.second == b.second && a.third < b.third);
    }
  };

#ifdef HAVE_HASH_MAP
  struct facetriplehash
  {
    unsigned int operator()(const facetriple_t &a) const
    {
#  if defined(__ECC) || defined(_MSC_VER)
      hash_compare<unsigned int> h;
#  else
      hash<unsigned int> h;
#  endif
      return h(a.first ^ a.second ^ a.third);
    }
#  if defined(__ECC) || defined(_MSC_VER)

    // These are particularly needed by ICC's hash stuff
    static const size_t bucket_size = 4;
    static const size_t min_buckets = 8;
    bool operator()(const facetriple_t &a, const facetriple_t &b) const
    {
      return a.first < b.first || a.first == b.first && ( a.second < b.second || a.second == b.second && a.third < b.third);
    }
#endif
  };
#endif

  

#ifdef HAVE_HASH_MAP
#  if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type> node_hash_type;

  typedef hash_map<edgepair_t,
		   typename FIELD::mesh_type::Node::index_type,
		   edgepairhash> edge_hash_type;

  typedef hash_map<facetriple_t,
		   typename FIELD::mesh_type::Node::index_type,
		   facetriplehash> face_hash_type;
#  else
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type,
		   hash<unsigned int>,
		   equal_to<unsigned int> > node_hash_type;

  typedef hash_map<edgepair_t,
		   typename FIELD::mesh_type::Node::index_type,
		   edgepairhash,
		   edgepairequal> edge_hash_type;

  typedef hash_map<facetriple_t,
		   typename FIELD::mesh_type::Node::index_type,
		   facetriplehash,
		   facetripleequal> face_hash_type;
#  endif
#else
  typedef map<unsigned int,
	      typename FIELD::mesh_type::Node::index_type,
	      less<unsigned int> > node_hash_type;

  typedef map<edgepair_t,
	      typename FIELD::mesh_type::Node::index_type,
	      edgepairless> edge_hash_type;

  typedef map<facetriple_t,
	      typename FIELD::mesh_type::Node::index_type,
	      facetripleless> face_hash_type;
#endif

  typename FIELD::mesh_type::Node::index_type
  edge_lookup(unsigned int u0, unsigned int u1, double d0,
	      const Point &p, edge_hash_type &edgemap,
	      typename FIELD::mesh_type *clipped) const;

  typename FIELD::mesh_type::Node::index_type
  face_lookup(unsigned int u0, unsigned int u1, unsigned int u2,
	      double d1, double d2,
	      const Point &p, face_hash_type &facemap,
	      typename FIELD::mesh_type *clipped) const;
};



template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTet<FIELD>::edge_lookup(unsigned int u0, unsigned int u1,
				   double d0,
				   const Point &p, edge_hash_type &edgemap,
				   typename FIELD::mesh_type *clipped) const
{
  edgepair_t np;
  if (u0 < u1)  { np.first = u0; np.second = u1; np.dfirst = d0; }
  else { np.first = u1; np.second = u0; np.dfirst = 1.0 - d0; }
  const typename edge_hash_type::iterator loc = edgemap.find(np);
  if (loc == edgemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    edgemap[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return (*loc).second;
  }
}



template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTet<FIELD>::face_lookup(unsigned int u0, unsigned int u1,
				   unsigned int u2, double d1,
				   double d2, const Point &p,
				   face_hash_type &facemap,
				   typename FIELD::mesh_type *clipped) const
{
  facetriple_t nt;
  if (u0 < u1)
  {
    if (u2 < u0)
    {
      nt.first = u2; nt.second = u0; nt.third = u1;
      nt.dsecond = 1.0 - d1 - d2; nt.dthird = d1;
    }
    else if (u2 < u1)
    {
      nt.first = u0; nt.second = u2; nt.third = u1;
      nt.dsecond = d2; nt.dthird = d1;
    }
    else
    {
      nt.first = u0; nt.second = u1; nt.third = u2;
      nt.dsecond = d1; nt.dthird = d2;
    }
  }
  else
  {
    if (u2 > u0)
    {
      nt.first = u1; nt.second = u0; nt.third = u2;
      nt.dsecond = 1.0 - d1 - d2; nt.dthird = d2;
    }
    else if (u2 > u1)
    {
      nt.first = u1; nt.second = u2; nt.third = u0;
      nt.dsecond = d2; nt.dthird = 1.0 - d1 - d2;
    }
    else
    {
      nt.first = u2; nt.second = u1; nt.third = u0;
      nt.dsecond = d1; nt.dthird = 1.0 - d1 - d2;
    }
  }
  const typename face_hash_type::iterator loc = facemap.find(nt);
  if (loc == facemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    facemap[nt] = nodeindex;
    return nodeindex;
  }
  else
  {
    return (*loc).second;
  }
}



template <class FIELD>
FieldHandle
IsoClipAlgoTet<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh,
			       double isoval, bool lte, MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  clipped->copy_properties(mesh);

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

    // Get the values and compute an inside/outside mask.
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

    // Invert the mask if we are doing less than.
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
      const double dl1 = imv / (v[perm[1]] - v[perm[0]]);
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]], dl1);
      const double dl2 = imv / (v[perm[2]] - v[perm[0]]);
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]], dl2);
      const double dl3 = imv / (v[perm[3]] - v[perm[0]]);
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]], dl3);


      nnodes[1] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      dl1, l1, edgemap, clipped);

      nnodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      dl2, l2, edgemap, clipped);

      nnodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      dl3, l3, edgemap, clipped);

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
      const double dl1 = imv / (v[perm[1]] - v[perm[0]]);
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]], dl1);
      const double dl2 = imv / (v[perm[2]] - v[perm[0]]);
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]], dl2);
      const double dl3 = imv / (v[perm[3]] - v[perm[0]]);
      const Point l3 = Interpolate(p[perm[0]], p[perm[3]], dl3);
				   

      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      dl1, l1, edgemap, clipped);

      inodes[4] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      dl2, l2, edgemap, clipped);

      inodes[5] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      dl3, l3, edgemap, clipped);

      const Point c1 = Interpolate(l1, l2, 0.5);
      const Point c2 = Interpolate(l2, l3, 0.5);
      const Point c3 = Interpolate(l3, l1, 0.5);

      inodes[6] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      dl1*0.5, dl2*0.5,
			      c1, facemap, clipped);
      inodes[7] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      (unsigned int)onodes[perm[3]],
			      dl2*0.5, dl3*0.5,
			      c2, facemap, clipped);
      inodes[8] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      (unsigned int)onodes[perm[1]],
			      dl3*0.5, dl1*0.5,
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
      const double dl02 = imv0 / (v[perm[2]] - v[perm[0]]);
      const Point l02 = Interpolate(p[perm[0]], p[perm[2]], dl02);
      const double dl03 = imv0 / (v[perm[3]] - v[perm[0]]);
      const Point l03 = Interpolate(p[perm[0]], p[perm[3]], dl03);
				    

      const double imv1 = isoval - v[perm[1]];
      const double dl12 = imv1 / (v[perm[2]] - v[perm[1]]);
      const Point l12 = Interpolate(p[perm[1]], p[perm[2]], dl12);
      const double dl13 = imv1 / (v[perm[3]] - v[perm[1]]);
      const Point l13 = Interpolate(p[perm[1]], p[perm[3]], dl13);
				    

      inodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      dl02, l02, edgemap, clipped);
      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[3]],
			      dl03, l03, edgemap, clipped);
      inodes[4] = edge_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      dl12, l12, edgemap, clipped);
      inodes[5] = edge_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[3]],
			      dl13, l13, edgemap, clipped);

      const Point c1 = Interpolate(l02, l03, 0.5);
      const Point c2 = Interpolate(l12, l13, 0.5);

      inodes[6] = face_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
      			      (unsigned int)onodes[perm[3]],
			      dl02*0.5,
			      dl03*0.5,
      			      c1, facemap, clipped);
      inodes[7] = face_lookup((unsigned int)onodes[perm[1]],
			      (unsigned int)onodes[perm[2]],
			      (unsigned int)onodes[perm[3]],
			      dl12*0.5,
			      dl13*0.5,
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
      nnodes[2] = inodes[6];
      nnodes[3] = inodes[7];
      clipped->add_elem(nnodes);
    }

    ++bi;
  }

  FIELD *ofield = scinew FIELD(clipped);
  ofield->copy_properties(fieldh.get_rep());

  // Add the data values from the old field to the new field.
  typename node_hash_type::iterator nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    typename FIELD::value_type val;
    field->value(val, (index_type)((*nmitr).first));
    ofield->set_value(val, (index_type)((*nmitr).second));

    ++nmitr;
  }

  // Put the isovalue at the edge break points.
  typename edge_hash_type::iterator emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*emitr).second));

    ++emitr;
  }

  // Put the isovalue at the face break points.  Assumes linear
  // interpolation across the faces (which seems safe, this is what we
  // used to cut with.)
  typename face_hash_type::iterator fmitr = facemap.begin();
  while (fmitr != facemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*fmitr).second));

    ++fmitr;
  }

  // Create the interpolant matrix.
  typename FIELD::mesh_type::Node::size_type nodesize;
  clipped->size(nodesize);
  const int nrows = (int)nodesize;
  mesh->size(nodesize);
  const int ncols = (int)nodesize;
  int *rr = scinew int[nrows+1];
  int *cctmp = scinew int[nrows*3];
  double *dtmp = scinew double[nrows*3];

  for (int i = 0; i < nrows * 3; i++)
  {
    cctmp[i] = -1;
  }

  int nnz = 0;

  // Add the data values from the old field to the new field.
  nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    cctmp[(*nmitr).second * 3] = (*nmitr).first;
    dtmp[(*nmitr).second * 3 + 0] = 1.0;
    nnz++;
    ++nmitr;
  }

  // Insert the double hits into cc.
  // Put the isovalue at the edge break points.
  emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    cctmp[(*emitr).second * 3 + 0] = (*emitr).first.first;
    cctmp[(*emitr).second * 3 + 1] = (*emitr).first.second;
    dtmp[(*emitr).second * 3 + 0] = 1.0 - (*emitr).first.dfirst;
    dtmp[(*emitr).second * 3 + 1] = (*emitr).first.dfirst;
    nnz+=2;
    ++emitr;
  }

  // Insert the double hits into cc.
  // Put the isovalue at the edge break points.
  fmitr = facemap.begin();
  while (fmitr != facemap.end())
  {
    cctmp[(*fmitr).second * 3 + 0] = (*fmitr).first.first;
    cctmp[(*fmitr).second * 3 + 1] = (*fmitr).first.second;
    cctmp[(*fmitr).second * 3 + 2] = (*fmitr).first.third;
    dtmp[(*fmitr).second * 3 + 0] =
      1.0 - (*fmitr).first.dsecond - (*fmitr).first.dthird;;
    dtmp[(*fmitr).second * 3 + 1] = (*fmitr).first.dsecond;
    dtmp[(*fmitr).second * 3 + 2] = (*fmitr).first.dthird;
    nnz+=3;

    ++fmitr;
  }

  int *cc = scinew int[nnz];
  double *d = scinew double[nnz];
  
  int j;
  int counter = 0;
  rr[0] = 0;
  for (j = 0; j < nrows*3; j++)
  {
    if (j%3 == 0) { rr[j/3 + 1] = rr[j/3]; }
    if (cctmp[j] != -1)
    {
      cc[counter] = cctmp[j];
      d[counter] = dtmp[j];
      rr[j/3 + 1]++;
      counter++;
    }
  }
  delete [] cctmp;
  delete [] dtmp;
  interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);

  return ofield;
}



template <class FIELD>
class IsoClipAlgoTri : public IsoClipAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte, MatrixHandle &interp);
private:

  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
    double dfirst;
  };

  struct edgepairequal
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b) const
    {
      return a.first == b.first && a.second == b.second;
    }
  };

  struct edgepairless
  {
    bool operator()(const edgepair_t &a, const edgepair_t &b)
    {
      return less(a, b);
    }
    static bool less(const edgepair_t &a, const edgepair_t &b)
    {
      return a.first < b.first || a.first == b.first && a.second < b.second;
    }
  };

#ifdef HAVE_HASH_MAP
  struct edgepairhash
  {
    unsigned int operator()(const edgepair_t &a) const
    {
#if defined(__ECC) || defined(_MSC_VER)
      hash_compare<unsigned int> h;
#else
      hash<unsigned int> h;
#endif
      return h(a.first ^ a.second);
    }
#if defined(__ECC) || defined(_MSC_VER)

      // These are particularly needed by ICC's hash stuff
      static const size_t bucket_size = 4;
      static const size_t min_buckets = 8;
      
      // This is a less than function.
      bool operator()(const edgepair_t & a, const edgepair_t & b) const {
        return edgepairless::less(a,b);
      }
#endif // endif ifdef __ICC
  };


#  if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type> node_hash_type;

  typedef hash_map<edgepair_t,
		   typename FIELD::mesh_type::Node::index_type,
		   edgepairhash> edge_hash_type;
#  else
  typedef hash_map<unsigned int,
		   typename FIELD::mesh_type::Node::index_type,
		   hash<unsigned int>,
		   equal_to<unsigned int> > node_hash_type;

  typedef hash_map<edgepair_t,
		   typename FIELD::mesh_type::Node::index_type,
		   edgepairhash, edgepairequal> edge_hash_type;
#  endif
#else
  typedef map<unsigned int,
	      typename FIELD::mesh_type::Node::index_type,
	      less<unsigned int> > node_hash_type;

  typedef map<edgepair_t,
	      typename FIELD::mesh_type::Node::index_type,
	      edgepairless> edge_hash_type;
#endif

  typename FIELD::mesh_type::Node::index_type
  edge_lookup(unsigned int u0, unsigned int u1, double d0,
	      const Point &p, edge_hash_type &edgemap,
	      typename FIELD::mesh_type *clipped) const;
};



template <class FIELD>
typename FIELD::mesh_type::Node::index_type
IsoClipAlgoTri<FIELD>::edge_lookup(unsigned int u0, unsigned int u1,
				   double d0, const Point &p,
				   edge_hash_type &edgemap,
				   typename FIELD::mesh_type *clipped) const
{
  edgepair_t np;
  if (u0 < u1)  { np.first = u0; np.second = u1; np.dfirst = d0; }
  else { np.first = u1; np.second = u0; np.dfirst = 1.0 - d0; }
  const typename edge_hash_type::iterator loc = edgemap.find(np);
  if (loc == edgemap.end())
  {
    const typename FIELD::mesh_type::Node::index_type nodeindex =
      clipped->add_point(p);
    edgemap[np] = nodeindex;
    return nodeindex;
  }
  else
  {
    return (*loc).second;
  }
}



template <class FIELD>
FieldHandle
IsoClipAlgoTri<FIELD>::execute(ProgressReporter *mod, FieldHandle fieldh,
			       double isoval, bool lte, MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
    dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  clipped->copy_properties(mesh);

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

    // Get the values and compute an inside/outside mask.
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

    // Invert the mask if we are doing less than.
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
      // Add the corner containing the inside point to the mesh.
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
      
      const double dl1 = imv / (v[perm[1]] - v[perm[0]]);
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]], dl1);
      const double dl2 = imv / (v[perm[2]] - v[perm[0]]);
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]], dl2);


      nnodes[1] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      dl1, l1, edgemap, clipped);

      nnodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      dl2, l2, edgemap, clipped);

      clipped->add_elem(nnodes);
    }
    else
    {
      // Lop off the one point that is outside of the mesh, then add
      // the remaining quad to the mesh by dicing it into two
      // triangles.
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
      const double dl1 = imv / (v[perm[1]] - v[perm[0]]);
      const Point l1 = Interpolate(p[perm[0]], p[perm[1]], dl1);
      const double dl2 = imv / (v[perm[2]] - v[perm[0]]);
      const Point l2 = Interpolate(p[perm[0]], p[perm[2]], dl2);

      inodes[2] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[1]],
			      dl1, l1, edgemap, clipped);

      inodes[3] = edge_lookup((unsigned int)onodes[perm[0]],
			      (unsigned int)onodes[perm[2]],
			      dl2, l2, edgemap, clipped);

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

  FIELD *ofield = scinew FIELD(clipped);
  ofield->copy_properties(fieldh.get_rep());

  // Add the data values from the old field to the new field.
  typename node_hash_type::iterator nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    typename FIELD::value_type val;
    field->value(val, (index_type)((*nmitr).first));
    ofield->set_value(val, (index_type)((*nmitr).second));

    ++nmitr;
  }

  // Put the isovalue at the edge break points.
  typename edge_hash_type::iterator emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    typedef typename FIELD::mesh_type::Node::index_type index_type;
    ofield->set_value(isoval, (index_type)((*emitr).second));

    ++emitr;
  }

  // Create the interpolant matrix.
  typename FIELD::mesh_type::Node::size_type nodesize;
  clipped->size(nodesize);
  const int nrows = (int)nodesize;
  mesh->size(nodesize);
  const int ncols = (int)nodesize;
  int *rr = scinew int[nrows+1];
  int *cctmp = scinew int[nrows*2];
  double *dtmp = scinew double[nrows*2];

  for (int i = 0; i < nrows * 2; i++)
  {
    cctmp[i] = -1;
  }

  int nnz = 0;

  // Add the data values from the old field to the new field.
  nmitr = nodemap.begin();
  while (nmitr != nodemap.end())
  {
    cctmp[(*nmitr).second * 2] = (*nmitr).first;
    dtmp[(*nmitr).second * 2 + 0] = 1.0;
    nnz++;
    ++nmitr;
  }

  // Insert the double hits into cc.
  // Put the isovalue at the edge break points.
  emitr = edgemap.begin();
  while (emitr != edgemap.end())
  {
    cctmp[(*emitr).second * 2 + 0] = (*emitr).first.first;
    cctmp[(*emitr).second * 2 + 1] = (*emitr).first.second;
    dtmp[(*emitr).second * 2 + 0] = 1.0 - (*emitr).first.dfirst;
    dtmp[(*emitr).second * 2 + 1] = (*emitr).first.dfirst;
    nnz+=2;
    ++emitr;
  }

  int *cc = scinew int[nnz];
  double *d = scinew double[nnz];
  
  int j;
  int counter = 0;
  rr[0] = 0;
  for (j = 0; j < nrows*2; j++)
  {
    if (j%2 == 0) { rr[j/2 + 1] = rr[j/2]; }
    if (cctmp[j] != -1)
    {
      cc[counter] = cctmp[j];
      d[counter] = dtmp[j];
      rr[j/2 + 1]++;
      counter++;
    }
  }
  delete [] cctmp;
  delete [] dtmp;
  interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nnz, d);

  return ofield;
}


template <class FIELD>
class IsoClipAlgoHex : public IsoClipAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute( ProgressReporter *reporter, FieldHandle fieldh,
                               double isoval, bool lte, MatrixHandle &interpolant );
private:

};

template <class FIELD>
FieldHandle
IsoClipAlgoHex<FIELD>::execute( ProgressReporter *mod, FieldHandle fieldh,
                                double isoval, bool lte, MatrixHandle &interp )
{
  mod->warning( "The IsoClip module for hexes is still under development..." );
  
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
      dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  typename FIELD::mesh_type *clipped = scinew typename FIELD::mesh_type();
  clipped->copy_properties(mesh);
//  typename FIELD::mesh_type *final = scinew typename FIELD::mesh_type();
//  final->copy_properties(mesh);

    //Get the original boundary elements (code from FieldBoundary)
  vector<typename FIELD::mesh_type::Face::index_type> original_face_list;
  
  mesh->synchronize(Mesh::NODE_NEIGHBORS_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E);

    // Walk all the cells in the mesh.
  typename FIELD::mesh_type::Cell::iterator o_citer; 
  mesh->begin(o_citer);
  typename FIELD::mesh_type::Cell::iterator o_citere; 
  mesh->end(o_citere);

  while( o_citer != o_citere )
  {
    typename FIELD::mesh_type::Cell::index_type o_ci = *o_citer;
    ++o_citer;
    
      // Get all the faces in the cell.
    typename FIELD::mesh_type::Face::array_type o_faces;
    mesh->get_faces(o_faces, o_ci);
    
      // Check each face for neighbors.
    typename FIELD::mesh_type::Face::array_type::iterator o_fiter = o_faces.begin();
    
    while (o_fiter != o_faces.end())
    {
      typename FIELD::mesh_type::Cell::index_type o_nci;
      typename FIELD::mesh_type::Face::index_type o_fi = *o_fiter;
      ++o_fiter;
      
      if( !mesh->get_neighbor( o_nci, o_ci, o_fi ) )
      {
          // Faces with no neighbors are on the boundary...
        original_face_list.push_back( o_fi );
      }
    }
  }

     //get a copy of the mesh in tets... 
  const TypeDescription *src_td = fieldh->get_type_description();
  CompileInfoHandle ci = HexToTetAlgo::get_compile_info( src_td );
  Handle<HexToTetAlgo> algo;
  if (!DynamicCompilation::compile( ci, algo, mod)) return fieldh;
  FieldHandle tet_field_h;
  if( !algo.get_rep() || !algo->execute( fieldh, tet_field_h, mod ) )
  {
    mod->warning("HexToTet conversion failed to copy data.");
    return fieldh;
  }

  const TypeDescription *ftd = tet_field_h->get_type_description();
  ci = IsoClipAlgo::get_compile_info( ftd, "Tet" );
  Handle<IsoClipAlgo> iso_algo;
  if( !DynamicCompilation::compile( ci, iso_algo, false, mod ) )
  {
    mod->error("Unable to compile IsoClip algorithm.");
    return fieldh;
  }
  if( !algo.get_rep() || !algo->execute( fieldh, tet_field_h, mod ) )
  {
    mod->warning("HexToTet conversion failed to copy data.");
    return fieldh;
  }
  MatrixHandle tet_interp(0);
  FieldHandle clipped_tet_field = iso_algo->execute( mod, tet_field_h, isoval, lte, tet_interp );
  TetVolMesh<TetLinearLgn<Point> > *tet_mesh = dynamic_cast<TetVolMesh<TetLinearLgn<Point> >*>(clipped_tet_field->mesh().get_rep());

  const TypeDescription *mtd = tet_mesh->get_type_description();
  CompileInfoHandle ci_boundary = FieldBoundaryAlgo::get_compile_info( mtd );
  Handle<FieldBoundaryAlgo> boundary_algo;
  FieldHandle tri_field_h;
  if( !DynamicCompilation::compile( ci_boundary, boundary_algo, false, mod ) ) return fieldh;

  MatrixHandle tet_interp1(0);  
  boundary_algo->execute( mod, tet_mesh, tri_field_h, tet_interp1, 0 );
  TriSurfMesh<TriLinearLgn<Point> > *tri_mesh = dynamic_cast<TriSurfMesh<TriLinearLgn<Point> >*>(tri_field_h->mesh().get_rep());
//   typename FIELD::mesh_type::Elem::size_type hex_mesh_size;
//   typename FIELD::mesh_type::Elem::size_type tet_mesh_size;
//   typename FIELD::mesh_type::Face::size_type tri_mesh_size;
//   mesh->size( hex_mesh_size );
//   tet_mesh->size( tet_mesh_size ); 
//   tri_mesh->size( tri_mesh_size );
//   cout << "Mesh sizes = " << hex_mesh_size << " " << tet_mesh_size << " " << tri_mesh_size << endl;

     //create a map to help differentiate between new nodes created for 
     //  the pillow, and the nodes on the stair stepped boundary...
  map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type> clipped_to_original_nodemap;

#ifdef HAVE_HASH_MAP
  typedef hash_map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    hash<unsigned int>,
    equal_to<unsigned int> > hash_type;
#else
  typedef map<unsigned int,
    typename FIELD::mesh_type::Node::index_type,
    less<unsigned int> > hash_type;
#endif

  hash_type nodemap;

  vector<typename FIELD::mesh_type::Elem::index_type> elemmap;
  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);

  while (bi != ei)
  {
    bool inside = false;

    typename FIELD::mesh_type::Node::array_type onodes;
    mesh->get_nodes(onodes, *bi);
    inside = true;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
//       Point p;
//       mesh->get_center(p, onodes[i]);
      typename FIELD::value_type v(0);
      if (field->basis_order() == 1) { field->value(v, onodes[i]); }

      if( lte )
      {
        if( v > isoval )
        {
          inside = false;
          break;
        }
      }
      else
      {
        if( v < isoval )
        {
          inside = false;
          break;
        }
      }
    }

    if (inside)
    {
      typename FIELD::mesh_type::Node::array_type onodes;
      mesh->get_nodes(onodes, *bi);

      // Add this element to the new mesh.
      typename FIELD::mesh_type::Node::array_type nnodes(onodes.size());

      for (unsigned int i = 0; i<onodes.size(); i++)
      {
        if (nodemap.find((unsigned int)onodes[i]) == nodemap.end())
        {
          Point np;
          mesh->get_center(np, onodes[i]);
          const typename FIELD::mesh_type::Node::index_type nodeindex =
              clipped->add_point(np);
          clipped_to_original_nodemap[nodeindex] = onodes[i];
          nodemap[(unsigned int)onodes[i]] = nodeindex;
          nnodes[i] = nodeindex;
        }
        else
        {
          nnodes[i] = nodemap[(unsigned int)onodes[i]];
        }
      }

      clipped->add_elem(nnodes);
      elemmap.push_back(*bi); // Assumes elements always added to end.
    }

    ++bi;
  }

//Get the boundary elements (code from FieldBoundary)
  map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type> vertex_map;
  typename map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type>::iterator node_iter;

  vector<typename FIELD::mesh_type::Node::index_type> node_list;
  vector<typename FIELD::mesh_type::Face::index_type> face_list;
  
  clipped->synchronize( Mesh::NODE_NEIGHBORS_E | Mesh::FACE_NEIGHBORS_E | Mesh::FACES_E );

  // Walk all the cells in the mesh.
  typename FIELD::mesh_type::Cell::iterator citer; clipped->begin(citer);
  typename FIELD::mesh_type::Cell::iterator citere; clipped->end(citere);
//  int counter = 0;
  while( citer != citere )
  {
    typename FIELD::mesh_type::Cell::index_type ci = *citer;
    ++citer;
    
      // Get all the faces in the cell.
    typename FIELD::mesh_type::Face::array_type faces;
    clipped->get_faces( faces, ci );
    
      // Check each face for neighbors.
    typename FIELD::mesh_type::Face::array_type::iterator fiter = faces.begin();
    
    while( fiter != faces.end() )
    {
      typename FIELD::mesh_type::Cell::index_type nci;
      typename FIELD::mesh_type::Face::index_type fi = *fiter;
      ++fiter;
      
      if( !clipped->get_neighbor( nci , ci, fi ) )
      {
          // Faces with no neighbors are on the boundary...
          //  make sure that this face isn't on the original boundary
        Point p;
        clipped->get_center( p, fi );
        typename FIELD::mesh_type::Face::index_type old_face;
        mesh->locate( old_face, p );
        unsigned int i;
        bool is_old_boundary = false;
        for( i = 0; i < original_face_list.size(); i++ )
        {
          if( original_face_list[i] == old_face )
          {
//            cout << "Found an old boundary face..." << ++counter << endl;
            is_old_boundary = true;
            break;
          }
        }

        if( !is_old_boundary )
        {
          face_list.push_back( fi );
          
          typename FIELD::mesh_type::Node::array_type nodes;
          clipped->get_nodes(nodes, fi);
          
          typename FIELD::mesh_type::Node::array_type::iterator niter = nodes.begin();
          
          for( i = 0; i < nodes.size(); i++ )
          {
            node_iter = vertex_map.find(*niter);
            if (node_iter == vertex_map.end())
            {
              node_list.push_back(*niter);
              vertex_map[*niter] = *niter;
            }
            ++niter;
          }
        }
      }
    }
  }

    //project a new node to the isoval for each node on the clipped boundary
  map<typename FIELD::mesh_type::Node::index_type, typename FIELD::mesh_type::Node::index_type> new_map;

  unsigned int i;  
  for( i = 0; i < node_list.size(); i++ )
  {
    typename FIELD::mesh_type::Node::index_type this_node = node_list[i];
    Point n_p;
    clipped->get_center( n_p, this_node );

    Point new_result;
    typename FIELD::mesh_type::Face::index_type face_id;
    tri_mesh->find_closest_face( new_result, face_id, n_p );

    Point new_point( new_result );

      //add the new node to the clipped mesh
    typename FIELD::mesh_type::Node::index_type this_index = clipped->add_point( new_point );
      
      //create a map for the new node to a node on the boundary of the clipped mesh...
    new_map[this_node] = this_index;
  }

    //for each quad on the clipped boundary we have a map to the new projected nodes
    // so, create a new hex for each quad on the boundary
  for( i = 0; i < face_list.size(); i++ )
  { 
    typename FIELD::mesh_type::Node::array_type nodes;
    clipped->get_nodes( nodes, face_list[i] );

//     Point p0, p1, p3, p4;
//     mesh->get_center( p0, nodes[0] );
//     mesh->get_center( p1, nodes[1] );
//     mesh->get_center( p3, nodes[2] );
//     mesh->get_center( p4, new_map[nodes[0]] );
//     Vector p0v( p0 );
//     Vector p1v( p1 );
//     Vector p3v( p3 );
//     Vector p4v( p4 );
//     Vector normal = Cross( (p1v-p0v), (p3v-p0v) );
//     Vector out = (p4v-p0v);
//     double angle = Dot( normal, out  );

//     if( angle > 1.5707963 )
//     {
//       cout << "Different order..." << endl;
//       clipped->add_hex( nodes[0], nodes[1], nodes[2], nodes[3], 
//                         new_map[nodes[0]], new_map[nodes[1]],
//                         new_map[nodes[2]], new_map[nodes[3]] ); 
//     }
//     else
//     {
      clipped->add_hex( nodes[3], nodes[2], nodes[1], nodes[0],
                        new_map[nodes[3]], new_map[nodes[2]],
                        new_map[nodes[1]], new_map[nodes[0]] );
//     }
  }
  
  // force all the synch data to be rebuilt on next synch call.
  clipped->unsynchronize();
  FIELD *ofield = scinew FIELD(clipped);

//   typename FIELD::mesh_type::Elem::iterator fbi, fei;
//   clipped->begin( fbi ); clipped->end( fei );

//   while( fbi != fei )
//   {
//     typename FIELD::mesh_type::Node::array_type fonodes;
//     clipped->get_nodes( fonodes, *fbi );
//     final->add_elem( fonodes );
//     ++fbi;
//   }
  
//  FIELD *ofield = scinew FIELD(final);
  ofield->copy_properties(fieldh.get_rep());
  
//NOTE TO JS: We'll worry about the interpolation matrix when we've finished the other part of the coding...
//   if (fieldh->basis_order() == 1)
//   {
//     FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());
//     typename hash_type::iterator hitr = nodemap.begin();

//     const int nrows = nodemap.size();;
//     const int ncols = field->fdata().size();
//     int *rr = scinew int[nrows+1];
//     int *cc = scinew int[nrows];
//     double *d = scinew double[nrows];

//     while (hitr != nodemap.end())
//     {
//       typename FIELD::value_type val;
//       field->value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).first));
//       ofield->set_value(val, (typename FIELD::mesh_type::Node::index_type)((*hitr).second));

//       cc[(*hitr).second] = (*hitr).first;

//       ++hitr;
//     }

//     int i;
//     for (i = 0; i < nrows; i++)
//     {
//       rr[i] = i;
//       d[i] = 1.0;
//     }
//     rr[i] = i; // An extra entry goes on the end of rr.

//     interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
//   }
//   else if (fieldh->order_type_description()->get_name() ==
// 	   get_type_description((typename FIELD::mesh_type::Elem *)0)->get_name())
//   {
//     FIELD *field = dynamic_cast<FIELD *>(fieldh.get_rep());

//     const int nrows = elemmap.size();
//     const int ncols = field->fdata().size();
//     int *rr = scinew int[nrows+1];
//     int *cc = scinew int[nrows];
//     double *d = scinew double[nrows];

//     for (unsigned int i=0; i < elemmap.size(); i++)
//     {
//       typename FIELD::value_type val;
//       field->value(val,
// 		   (typename FIELD::mesh_type::Elem::index_type)elemmap[i]);
//       ofield->set_value(val, (typename FIELD::mesh_type::Elem::index_type)i);

//       cc[i] = elemmap[i];
//     }

//     int j;
//     for (j = 0; j < nrows; j++)
//     {
//       rr[j] = j;
//       d[j] = 1.0;
//     }
//     rr[j] = j; // An extra entry goes on the end of rr.

//     interp = scinew SparseRowMatrix(nrows, ncols, rr, cc, nrows, d);
//   }
//   else
//   {
//     mod->warning("Unable to copy data at this field data location.");
//     mod->warning("No interpolant computed for field data location.");
//     interp = 0;
//   }

  return ofield;
}


} // end namespace SCIRun

#endif // ClipField_h
