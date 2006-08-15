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
#include <Core/Basis/TriLinearLgn.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/HexVolMesh.h>
#include <Core/Datatypes/GenericField.h>

#include <sci_hash_map.h>
#include <algorithm>
#include <set>

#include <Dataflow/Modules/Fields/share.h>

namespace SCIRun {
typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
typedef HexVolMesh<HexTrilinearLgn<Point> > HVMesh;
class GuiInterface;

class SCISHARE IsoRefineAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsrc,
					    string ext);

  static int hex_reorder_table[14][8];
  static double hcoords_double[8][3];
};


template <class FIELD>
class IsoRefineAlgoQuad : public IsoRefineAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant);

  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
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

#ifdef HAVE_HASH_MAP
#  if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<edgepair_t,
		   QSMesh::Node::index_type,
		   edgepairhash> edge_hash_type;
#  else
  typedef hash_map<edgepair_t,
		   QSMesh::Node::index_type,
		   edgepairhash,
		   edgepairequal> edge_hash_type;
#  endif
#else
  typedef map<edgepair_t,
	      QSMesh::Node::index_type,
	      edgepairless> edge_hash_type;
#endif

  typename QSMesh::Node::index_type
  lookup(typename FIELD::mesh_type *mesh,
         QSMesh *refined,
         edge_hash_type &edgemap,
         typename FIELD::mesh_type::Node::index_type a,
         typename FIELD::mesh_type::Node::index_type b)
  {
    edgepair_t ep;
    ep.first = a; ep.second = b;
    const typename edge_hash_type::iterator loc = edgemap.find(ep);
    if (loc == edgemap.end())
    {
      Point pa, pb;
      mesh->get_point(pa, a);
      mesh->get_point(pb, b);
      const Point inbetween = Interpolate(pa, pb, 1.0/3.0);
      const QSMesh::Node::index_type newnode = refined->add_point(inbetween);
      edgemap[ep] = newnode;
      return newnode;
    }
    else
    {
      return (*loc).second;
    }
  }
};


template <class FIELD>
FieldHandle
IsoRefineAlgoQuad<FIELD>::execute(ProgressReporter *reporter,
                                  FieldHandle fieldh,
                                  double isoval, bool lte,
                                  MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
      dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  QSMesh *refined = scinew QSMesh();
  refined->copy_properties(mesh);

  edge_hash_type emap;

  typename FIELD::mesh_type::Node::array_type onodes(4);
  QSMesh::Node::array_type oqnodes(4);
  QSMesh::Node::array_type nnodes(4);
  typename FIELD::value_type v[4];
  
  // Copy all of the nodes from mesh to refined.  They won't change,
  // we only add nodes.
  typename FIELD::mesh_type::Node::iterator bni, eni;
  mesh->begin(bni); mesh->end(eni);
  while (bni != eni)
  {
    Point p;
    mesh->get_point(p, *bni);
    refined->add_point(p);
    ++bni;
  }

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);
    
    // Get the values and compute an inside/outside mask.
    unsigned int inside = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      field->value(v[i], onodes[i]);
      oqnodes[i] = QSMesh::Node::index_type((unsigned int)onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
        inside |= 1;
      }
    }

    bool refine_elem = false;
    
    // Invert the mask if we are doing less than.
    if (lte) { inside = ~inside & 0xf; }
    
    if (!refine_elem && inside == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(oqnodes);
    }
    else if (!refine_elem &&
             (inside == 1 || inside == 2 || inside == 4 || inside == 8))
    {
      int index;
      if (inside == 1) index = 3;
      else if (inside == 2) index = 2;
      else if (inside == 4) index = 1;
      else index = 0;

      const int i0 = index;
      const int i1 = (index+1)%4;
      const int i2 = (index+2)%4;
      const int i3 = (index+3)%4;

      const int tab[4][2] = {{0,0}, {1, 0}, {1, 1}, {0, 1}};
      vector<double> coords(2);
      coords[0] = tab[index][0] * 1.0/3.0 + 1.0/3.0;
      coords[1] = tab[index][1] * 1.0/3.0 + 1.0/3.0;
      Point interior;
      mesh->interpolate(interior, coords, *bi);
      const QSMesh::Node::index_type interior_node =
        refined->add_point(interior);

      nnodes[0] = oqnodes[i0];
      nnodes[1] = lookup(mesh, refined, emap, onodes[i0], onodes[i1]);
      nnodes[2] = interior_node;
      nnodes[3] = lookup(mesh, refined, emap, onodes[i0], onodes[i3]);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, onodes[i0], onodes[i1]);
      nnodes[1] = oqnodes[i1];
      nnodes[2] = oqnodes[i2];
      nnodes[3] = interior_node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, onodes[i0], onodes[i3]);
      nnodes[1] = interior_node;
      nnodes[2] = oqnodes[i2];
      nnodes[3] = oqnodes[i3];
      refined->add_elem(nnodes);
    }
    else if (!refine_elem && (inside == 5 || inside == 10))
    {
      int index = 0;
      if (inside == 5) index = 1;

      const int i0 = index;
      const int i1 = (index+1)%4;
      const int i2 = (index+2)%4;
      const int i3 = (index+3)%4;

      vector<double> coords(2);
      coords[0] = 0.5;
      coords[1] = 0.5;
      Point center;
      mesh->interpolate(center, coords, *bi);
      const QSMesh::Node::index_type center_node =
        refined->add_point(center);

      nnodes[0] = oqnodes[i0];
      nnodes[1] = lookup(mesh, refined, emap, onodes[i0], onodes[i1]);
      nnodes[2] = center_node;
      nnodes[3] = lookup(mesh, refined, emap, onodes[i0], onodes[i3]);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, onodes[i0], onodes[i1]);
      nnodes[1] = oqnodes[i1];
      nnodes[2] = lookup(mesh, refined, emap, onodes[i2], onodes[i1]);
      nnodes[3] = center_node;
      refined->add_elem(nnodes);

      nnodes[0] = center_node;
      nnodes[1] = lookup(mesh, refined, emap, onodes[i2], onodes[i1]);
      nnodes[2] = oqnodes[i2];
      nnodes[3] = lookup(mesh, refined, emap, onodes[i2], onodes[i3]);
      refined->add_elem(nnodes);
      
      nnodes[0] = lookup(mesh, refined, emap, onodes[i0], onodes[i3]);
      nnodes[1] = center_node;
      nnodes[2] = lookup(mesh, refined, emap, onodes[i2], onodes[i3]);
      nnodes[3] = oqnodes[i3];
      refined->add_elem(nnodes);
    }
    else
    {
      Point interiorp[4];
      QSMesh::Node::array_type inodes(4);
      for (unsigned int i = 0; i < 4; i++)
      {
        const int tab[4][2] =
          {{0,0}, {1, 0}, {1, 1}, {0, 1}};
        vector<double> coords(2);
        coords[0] = tab[i][0] * 1.0/3.0 + 1.0/3.0;
        coords[1] = tab[i][1] * 1.0/3.0 + 1.0/3.0;
        mesh->interpolate(interiorp[i], coords, *bi);
        inodes[i] = refined->add_point(interiorp[i]);
      }
      refined->add_elem(inodes);
      
      for (unsigned int i = 0; i < 4; i++)
      {
        if (inside & (1 << (3-i)))
        {
          nnodes[0] = oqnodes[i];
          nnodes[1] = lookup(mesh, refined, emap, onodes[i], onodes[(i+1)%4]);
          nnodes[2] = inodes[i];
          nnodes[3] = lookup(mesh, refined, emap, onodes[i], onodes[(i+3)%4]);
          refined->add_elem(nnodes);
        }

        if (inside & (1 << (3-i)))
        {
          nnodes[0] = lookup(mesh, refined, emap, onodes[i], onodes[(i+1)%4]);
        }
        else
        {
          nnodes[0] = oqnodes[i];
        }
        if (inside & (1 << (3 - (i+1)%4)))
        {
          nnodes[1] = lookup(mesh, refined, emap, onodes[(i+1)%4], onodes[i]);
        }
        else
        {
          nnodes[1] = oqnodes[(i+1)%4];
        }
        nnodes[2] = inodes[(i+1)%4];
        nnodes[3] = inodes[i];
        refined->add_elem(nnodes);
      }
    }
    ++bi;
  }

  GenericField<QSMesh, QuadBilinearLgn<double>, vector<double> > *ofield =
    scinew GenericField<QSMesh, QuadBilinearLgn<double>, vector<double> >(refined);
  ofield->copy_properties(fieldh.get_rep());
  return ofield;
}



template <class FIELD>
class IsoRefineAlgoHex : public IsoRefineAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh,
			      double isoval, bool lte,
			      MatrixHandle &interpolant);


  struct edgepair_t
  {
    unsigned int first;
    unsigned int second;
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

#ifdef HAVE_HASH_MAP
#  if defined(__ECC) || defined(_MSC_VER)
  typedef hash_map<edgepair_t,
		   HVMesh::Node::index_type,
		   edgepairhash> edge_hash_type;
#  else
  typedef hash_map<edgepair_t,
		   HVMesh::Node::index_type,
		   edgepairhash,
		   edgepairequal> edge_hash_type;
#  endif
#else
  typedef map<edgepair_t,
	      HVMesh::Node::index_type,
	      edgepairless> edge_hash_type;
#endif

  HVMesh::Node::index_type
  add_point(typename FIELD::mesh_type *mesh,
            HVMesh *refined,
            const typename FIELD::mesh_type::Elem::index_type &elem,
            const Point &coordsp)
  {
    vector<double> coords(3);
    coords[0] = coordsp.x();
    coords[1] = coordsp.y();
    coords[2] = coordsp.z();
    Point inbetween;
    mesh->interpolate(inbetween, coords, elem);
    return refined->add_point(inbetween);
  }


  HVMesh::Node::index_type
  add_point(typename FIELD::mesh_type *mesh,
            HVMesh *refined,
            const typename FIELD::mesh_type::Elem::index_type &elem,
            const int *reorder, int a, int b)
  {
    const Point coordsp =
      Interpolate(hcoords[reorder[a]], hcoords[reorder[b]], 1.0/3.0);
    vector<double> coords(3);
    coords[0] = coordsp.x();
    coords[1] = coordsp.y();
    coords[2] = coordsp.z();
    Point inbetween;
    mesh->interpolate(inbetween, coords, elem);
    return refined->add_point(inbetween);
  }

  HVMesh::Node::index_type
  lookup(typename FIELD::mesh_type *mesh,
         HVMesh *refined,
         edge_hash_type &edgemap,
         const typename FIELD::mesh_type::Elem::index_type &elem,
         const typename FIELD::mesh_type::Node::array_type &onodes,
         const int *reorder, int a, int b)
  {
    edgepair_t ep;
    ep.first = onodes[reorder[a]]; ep.second = onodes[reorder[b]];
    const typename edge_hash_type::iterator loc = edgemap.find(ep);
    if (loc == edgemap.end())
    {
      const HVMesh::Node::index_type newnode =
        add_point(mesh, refined, elem, reorder, a, b);
      edgemap[ep] = newnode;
      return newnode;
    }
    else
    {
      return (*loc).second;
    }
  }


  Point hcoords[8];
  int pattern_table[256][2];

  inline unsigned int iedge(unsigned int a, unsigned int b)
  {
    return (1<<(7-a)) | (1<<(7-b));
  }

  inline unsigned int iface(unsigned int a, unsigned int b,
                     unsigned int c, unsigned int d)
  {
    return iedge(a, b) | iedge(c, d);
  }

  inline void set_table(int i, int pattern, int reorder)
  {
    pattern_table[i][0] = pattern;
    pattern_table[i][1] = reorder;
  }

  void init_pattern_table()
  {
    for (int i = 0; i < 256; i++)
    {
      set_table(i, -1, 0);
    }

    set_table(0, 0, 0);

    // Add corners
    set_table(1, 1, 7);
    set_table(2, 1, 6);
    set_table(4, 1, 5);
    set_table(8, 1, 4);
    set_table(16, 1, 3);
    set_table(32, 1, 2);
    set_table(64, 1, 1);
    set_table(128, 1, 0);

    // Add edges
    set_table(iedge(0, 1), 2, 0);
    set_table(iedge(1, 2), 2, 1);
    set_table(iedge(2, 3), 2, 2);
    set_table(iedge(3, 0), 2, 3);
    set_table(iedge(4, 5), 2, 5);
    set_table(iedge(5, 6), 2, 6);
    set_table(iedge(6, 7), 2, 7);
    set_table(iedge(7, 4), 2, 4);
    set_table(iedge(0, 4), 2, 8);
    set_table(iedge(1, 5), 2, 9);
    set_table(iedge(2, 6), 2, 10);
    set_table(iedge(3, 7), 2, 11);

    set_table(iface(0, 1, 2, 3), 4, 0);
    set_table(iface(0, 1, 5, 4), 4, 12);
    set_table(iface(1, 2, 6, 5), 4, 9);
    set_table(iface(2, 3, 7, 6), 4, 13);
    set_table(iface(3, 0, 4, 7), 4, 8);
    set_table(iface(4, 5, 6, 7), 4, 7);

    set_table(255, 8, 0);

    for (int i = 0; i < 8; i++)
    {
      hcoords[i] = Point(hcoords_double[i][0],
                         hcoords_double[i][1],
                         hcoords_double[i][2]);
    }
  }
};


template <class FIELD>
FieldHandle
IsoRefineAlgoHex<FIELD>::execute(ProgressReporter *reporter,
                                 FieldHandle fieldh,
                                 double isoval, bool lte,
                                 MatrixHandle &interp)
{
  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh =
      dynamic_cast<typename FIELD::mesh_type *>(fieldh->mesh().get_rep());
  HVMesh *refined = scinew HVMesh();
  refined->copy_properties(mesh);

  init_pattern_table();
  edge_hash_type emap;
  
  typename FIELD::mesh_type::Node::array_type onodes(8);
  HVMesh::Node::array_type ohnodes(8);
  HVMesh::Node::array_type nnodes(8);
  typename FIELD::value_type v[8];
  
  // Copy all of the nodes from mesh to refined.  They won't change,
  // we only add nodes.
  typename FIELD::mesh_type::Node::iterator bni, eni;
  mesh->begin(bni); mesh->end(eni);
  while (bni != eni)
  {
    Point p;
    mesh->get_point(p, *bni);
    refined->add_point(p);
    ++bni;
  }

  typename FIELD::mesh_type::Elem::iterator bi, ei;
  mesh->begin(bi); mesh->end(ei);
  while (bi != ei)
  {
    mesh->get_nodes(onodes, *bi);
    
    // Get the values and compute an inside/outside mask.
    unsigned int inside = 0;
    unsigned int inside_count = 0;
    for (unsigned int i = 0; i < onodes.size(); i++)
    {
      field->value(v[i], onodes[i]);
      ohnodes[i] = HVMesh::Node::index_type((unsigned int)onodes[i]);
      inside = inside << 1;
      if (v[i] > isoval)
      {
        inside |= 1;
        inside_count++;
      }
    }

    // Invert the mask if we are doing less than.
    if (lte) { inside = ~inside & 0xff; inside_count = 8 - inside_count; }

    const int pattern = pattern_table[inside][0];
    const int which = pattern_table[inside][1];

    if (pattern == 0)
    {
      // Nodes are the same order, so just add the element.
      refined->add_elem(ohnodes);
    }
    else if (pattern == 1)
    {
      const int *ro = hex_reorder_table[which];
      
      const HVMesh::Node::index_type i06node =
        add_point(mesh, refined, *bi, ro, 0, 6);

      // Add this corner.
      nnodes[0] = ohnodes[ro[0]];
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[6] = i06node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      refined->add_elem(nnodes);

      // Add the other three pieces.
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[1] = ohnodes[ro[1]];
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = i06node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = ohnodes[ro[3]];
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[5] = i06node;
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);
      
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[2] = i06node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else if (pattern == 2)
    {
      const int *ro = hex_reorder_table[which];

      const HVMesh::Node::index_type i06node =
        add_point(mesh, refined, *bi, ro, 0, 6);
      const HVMesh::Node::index_type i17node =
        add_point(mesh, refined, *bi, ro, 1, 7);
      const HVMesh::Node::index_type i60node =
        add_point(mesh, refined, *bi, ro, 6, 0);
      const HVMesh::Node::index_type i71node =
        add_point(mesh, refined, *bi, ro, 7, 1);

      // Leading edge.
      nnodes[0] = ohnodes[ro[0]];
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[6] = i06node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[6] = i17node;
      nnodes[7] = i06node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[1] = ohnodes[ro[1]];
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[7] = i17node;
      refined->add_elem(nnodes);

      // Top center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[3] = ohnodes[ro[3]];
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[5] = i06node;
      nnodes[6] = i71node;
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[4] = i06node;
      nnodes[5] = i17node;
      nnodes[6] = i60node;
      nnodes[7] = i71node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[4] = i17node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = i60node;
      refined->add_elem(nnodes);

      // Front Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[2] = i06node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[6] = i71node;
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[2] = i17node;
      nnodes[3] = i06node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[6] = i60node;
      nnodes[7] = i71node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[3] = i17node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = i60node;
      refined->add_elem(nnodes);

      // Outside wedges
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = ohnodes[ro[3]];
      nnodes[4] = i71node;
      nnodes[5] = i60node;
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[2] = i60node;
      nnodes[3] = i71node;
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else if (pattern == 4)
    {
      const int *ro = hex_reorder_table[which];

      // Interior
      const HVMesh::Node::index_type i06node =
        add_point(mesh, refined, *bi, ro, 0, 6);
      const HVMesh::Node::index_type i17node =
        add_point(mesh, refined, *bi, ro, 1, 7);
      const HVMesh::Node::index_type i24node =
        add_point(mesh, refined, *bi, ro, 2, 4);
      const HVMesh::Node::index_type i35node =
        add_point(mesh, refined, *bi, ro, 3, 5);

      
      const Point i06 = Interpolate(hcoords[ro[0]], hcoords[ro[6]], 1.0/3.0);
      const Point i17 = Interpolate(hcoords[ro[1]], hcoords[ro[7]], 1.0/3.0);
      const Point i24 = Interpolate(hcoords[ro[2]], hcoords[ro[4]], 1.0/3.0);
      const Point i35 = Interpolate(hcoords[ro[3]], hcoords[ro[5]], 1.0/3.0);
      const Point i42a = Interpolate(hcoords[ro[4]], hcoords[ro[2]], 1.0/3.0);
      const Point i53a = Interpolate(hcoords[ro[5]], hcoords[ro[3]], 1.0/3.0);
      const Point i60a = Interpolate(hcoords[ro[6]], hcoords[ro[0]], 1.0/3.0);
      const Point i71a = Interpolate(hcoords[ro[7]], hcoords[ro[1]], 1.0/3.0);
      const Point i42 = Interpolate(i06, i42a, 0.5);
      const Point i53 = Interpolate(i17, i53a, 0.5);
      const Point i60 = Interpolate(i24, i60a, 0.5);
      const Point i71 = Interpolate(i35, i71a, 0.5);
      const HVMesh::Node::index_type i42node =
        add_point(mesh, refined, *bi, i42);
      const HVMesh::Node::index_type i53node =
        add_point(mesh, refined, *bi, i53);
      const HVMesh::Node::index_type i60node =
        add_point(mesh, refined, *bi, i60);
      const HVMesh::Node::index_type i71node =
        add_point(mesh, refined, *bi, i71);

      // Top Front
      nnodes[0] = ohnodes[ro[0]];
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[6] = i06node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[6] = i17node;
      nnodes[7] = i06node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[1] = ohnodes[ro[1]];
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[7] = i17node;
      refined->add_elem(nnodes);

      // Top Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 0);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[5] = i06node;
      nnodes[6] = i35node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[4] = i06node;
      nnodes[5] = i17node;
      nnodes[6] = i24node;
      nnodes[7] = i35node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 1);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[4] = i17node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[7] = i24node;
      refined->add_elem(nnodes);

      // Top Back
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 2);
      nnodes[3] = ohnodes[ro[3]];
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[5] = i35node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 2);
      nnodes[4] = i35node;
      nnodes[5] = i24node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 1);
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 3);
      nnodes[4] = i24node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      refined->add_elem(nnodes);

      // Front
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[2] = i06node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[6] = i42node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[2] = i17node;
      nnodes[3] = i06node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[6] = i53node;
      nnodes[7] = i42node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[3] = i17node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[7] = i53node;
      refined->add_elem(nnodes);

      // Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[1] = i06node;
      nnodes[2] = i35node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      nnodes[5] = i42node;
      nnodes[6] = i71node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      refined->add_elem(nnodes);

      nnodes[0] = i06node;
      nnodes[1] = i17node;
      nnodes[2] = i24node;
      nnodes[3] = i35node;
      nnodes[4] = i42node;
      nnodes[5] = i53node;
      nnodes[6] = i60node;
      nnodes[7] = i71node;
      refined->add_elem(nnodes);

      nnodes[0] = i17node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[3] = i24node;
      nnodes[4] = i53node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[7] = i60node;
      refined->add_elem(nnodes);

      // Back
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[1] = i35node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 7);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      nnodes[5] = i71node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = i35node;
      nnodes[1] = i24node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[4] = i71node;
      nnodes[5] = i60node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      refined->add_elem(nnodes);

      nnodes[0] = i24node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 6);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[4] = i60node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      refined->add_elem(nnodes);

      // Bottom Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      nnodes[1] = i42node;
      nnodes[2] = i71node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = i42node;
      nnodes[1] = i53node;
      nnodes[2] = i60node;
      nnodes[3] = i71node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      refined->add_elem(nnodes);

      nnodes[0] = i53node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[3] = i60node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      refined->add_elem(nnodes);

      // Bottom
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);
    }
    else if (pattern == 8)
    {
      const int *ro = hex_reorder_table[which];

      // Interior
      const HVMesh::Node::index_type i06node =
        add_point(mesh, refined, *bi, ro, 0, 6);
      const HVMesh::Node::index_type i17node =
        add_point(mesh, refined, *bi, ro, 1, 7);
      const HVMesh::Node::index_type i24node =
        add_point(mesh, refined, *bi, ro, 2, 4);
      const HVMesh::Node::index_type i35node =
        add_point(mesh, refined, *bi, ro, 3, 5);
      const HVMesh::Node::index_type i42node =
        add_point(mesh, refined, *bi, ro, 4, 2);
      const HVMesh::Node::index_type i53node =
        add_point(mesh, refined, *bi, ro, 5, 3);
      const HVMesh::Node::index_type i60node =
        add_point(mesh, refined, *bi, ro, 6, 0);
      const HVMesh::Node::index_type i71node =
        add_point(mesh, refined, *bi, ro, 7, 1);

      // Top Front
      nnodes[0] = ohnodes[ro[0]];
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[6] = i06node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[6] = i17node;
      nnodes[7] = i06node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 0);
      nnodes[1] = ohnodes[ro[1]];
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[7] = i17node;
      refined->add_elem(nnodes);

      // Top Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 0);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[5] = i06node;
      nnodes[6] = i35node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 2);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[4] = i06node;
      nnodes[5] = i17node;
      nnodes[6] = i24node;
      nnodes[7] = i35node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 3);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 1);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[4] = i17node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[7] = i24node;
      refined->add_elem(nnodes);

      // Top Back
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 2);
      nnodes[3] = ohnodes[ro[3]];
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[5] = i35node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 2);
      nnodes[4] = i35node;
      nnodes[5] = i24node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 1);
      nnodes[2] = ohnodes[ro[2]];
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 3);
      nnodes[4] = i24node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      refined->add_elem(nnodes);

      // Front
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[2] = i06node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 0);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[6] = i42node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 5);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[2] = i17node;
      nnodes[3] = i06node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[6] = i53node;
      nnodes[7] = i42node;
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 4);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 5);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[3] = i17node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 1);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[7] = i53node;
      refined->add_elem(nnodes);

      // Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 0, 7);
      nnodes[1] = i06node;
      nnodes[2] = i35node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      nnodes[5] = i42node;
      nnodes[6] = i71node;
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      refined->add_elem(nnodes);

      nnodes[0] = i06node;
      nnodes[1] = i17node;
      nnodes[2] = i24node;
      nnodes[3] = i35node;
      nnodes[4] = i42node;
      nnodes[5] = i53node;
      nnodes[6] = i60node;
      nnodes[7] = i71node;
      refined->add_elem(nnodes);

      nnodes[0] = i17node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 1, 6);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[3] = i24node;
      nnodes[4] = i53node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[7] = i60node;
      refined->add_elem(nnodes);

      // Back
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 4);
      nnodes[1] = i35node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 7);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      nnodes[5] = i71node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 3);
      refined->add_elem(nnodes);

      nnodes[0] = i35node;
      nnodes[1] = i24node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 3, 6);
      nnodes[4] = i71node;
      nnodes[5] = i60node;
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      refined->add_elem(nnodes);

      nnodes[0] = i24node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 5);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 6);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 2, 7);
      nnodes[4] = i60node;
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 2);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      refined->add_elem(nnodes);

      // Bottom Front
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[2] = i42node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      nnodes[4] = ohnodes[ro[4]];
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 7);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 1);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[2] = i53node;
      nnodes[3] = i42node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 5);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 4);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 7);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 6);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 0);
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[3] = i53node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 4);
      nnodes[5] = ohnodes[ro[5]];
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 6);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 7);
      refined->add_elem(nnodes);

      // Bottom Center
      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 3);
      nnodes[1] = i42node;
      nnodes[2] = i71node;
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 7);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 6);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 5);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 4);
      refined->add_elem(nnodes);

      nnodes[0] = i42node;
      nnodes[1] = i53node;
      nnodes[2] = i60node;
      nnodes[3] = i71node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 4, 6);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 7);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 4);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 5);
      refined->add_elem(nnodes);

      nnodes[0] = i53node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 2);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[3] = i60node;
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 7);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 5, 6);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 5);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 4);
      refined->add_elem(nnodes);

      nnodes[0] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 0);
      nnodes[1] = i71node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 5);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 6);
      nnodes[7] = ohnodes[ro[7]];
      refined->add_elem(nnodes);

      nnodes[0] = i71node;
      nnodes[1] = i60node;
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 2);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 5);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 4);
      nnodes[6] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 7);
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 7, 6);
      refined->add_elem(nnodes);

      nnodes[0] = i60node;
      nnodes[1] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 1);
      nnodes[2] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 2);
      nnodes[3] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 3);
      nnodes[4] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 4);
      nnodes[5] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 5);
      nnodes[6] = ohnodes[ro[6]];
      nnodes[7] = lookup(mesh, refined, emap, *bi, onodes, ro, 6, 7);
      refined->add_elem(nnodes);
    }
    else
    {
      // non convex, emit error.
      cout << "Element not convex, cannot replace.\n";
    }
    ++bi;
  }

  GenericField<HVMesh, HexTrilinearLgn<double>, vector<double> > *ofield =
    scinew GenericField<HVMesh, HexTrilinearLgn<double>, vector<double> >(refined);
  ofield->copy_properties(fieldh.get_rep());
  return ofield;
}



class SCISHARE IRMakeLinearAlgo : public DynamicAlgoBase
{
public:

  virtual FieldHandle execute(ProgressReporter *reporter,
                              FieldHandle fieldh) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsr);
};


template <class IFIELD, class OFIELD>
class IRMakeLinearAlgoT : public IRMakeLinearAlgo
{
public:
  //! virtual interface. 
  virtual FieldHandle execute(ProgressReporter *reporter, FieldHandle fieldh);
};


template <class IFIELD, class OFIELD>
FieldHandle
IRMakeLinearAlgoT<IFIELD, OFIELD>::execute(ProgressReporter *reporter,
                                           FieldHandle fieldh)
{
  IFIELD *ifield = dynamic_cast<IFIELD*>(fieldh.get_rep());
  typename IFIELD::mesh_type *imesh = ifield->get_typed_mesh().get_rep();
  OFIELD *ofield = scinew OFIELD(imesh);

  typename IFIELD::mesh_type::Node::array_type nodes;
  typename IFIELD::value_type val;

  typename IFIELD::mesh_type::Elem::iterator itr, eitr;
  imesh->begin(itr);
  imesh->end(eitr);

  while (itr != eitr)
  {
    ifield->value(val, *itr);
    if (val > 0.5)
    {
      imesh->get_nodes(nodes, *itr);
      for (unsigned int i = 0; i < nodes.size(); i++)
      {
        ofield->set_value(val, nodes[i]);
      }
    }
    ++itr;
  }
  
  return ofield;
}


class SCISHARE IRMakeConvexAlgo : public DynamicAlgoBase
{
public:

  virtual void execute(ProgressReporter *reporter, FieldHandle fieldh, double isoval, bool lte) = 0;

  //! support the dynamically compiled algorithm concept
  static CompileInfoHandle get_compile_info(const TypeDescription *fsr);

  int pattern_table[256][2];

  inline unsigned int iedge(unsigned int a, unsigned int b)
  {
    return (1<<(7-a)) | (1<<(7-b));
  }

  inline unsigned int iface(unsigned int a, unsigned int b,
                            unsigned int c, unsigned int d)
  {
    return iedge(a, b) | iedge(c, d);
  }

  inline void set_table(int i, int pattern, int reorder)
  {
    pattern_table[i][0] = pattern;
    pattern_table[i][1] = reorder;
  }

  inline void set_table_once(int i, int pattern, int reorder)
  {
    if (pattern_table[i][0] < 0)
    {
      pattern_table[i][0] = pattern;
      pattern_table[i][1] = reorder;
    }
  }

  inline void set_iface_partials(unsigned int a, unsigned int b,
                                 unsigned int c, unsigned int d,
                                 int pattern, int reorder)
  {
//    set_table_once(iedge(a, b), pattern, reorder);
    set_table_once(iedge(a, c), pattern, reorder);
//    set_table_once(iedge(a, d), pattern, reorder);
//    set_table_once(iedge(b, c), pattern, reorder);
    set_table_once(iedge(b, d), pattern, reorder);
//    set_table_once(iedge(c, d), pattern, reorder);
    set_table(iface(b, b, c, d), pattern, reorder);
    set_table(iface(a, c, c, d), pattern, reorder);
    set_table(iface(a, b, d, d), pattern, reorder);
    set_table(iface(a, b, c, a), pattern, reorder);
  }

  void init_pattern_table()
  {
    for (int i = 0; i < 256; i++)
    {
      set_table(i, -1, 0);
    }

    set_table(0, 0, 0);

    // Add corners
    set_table(1, 1, 7);
    set_table(2, 1, 6);
    set_table(4, 1, 5);
    set_table(8, 1, 4);
    set_table(16, 1, 3);
    set_table(32, 1, 2);
    set_table(64, 1, 1);
    set_table(128, 1, 0);

    // Add edges
    set_table(iedge(0, 1), 2, 0);
    set_table(iedge(1, 2), 2, 1);
    set_table(iedge(2, 3), 2, 2);
    set_table(iedge(3, 0), 2, 3);
    set_table(iedge(4, 5), 2, 5);
    set_table(iedge(5, 6), 2, 6);
    set_table(iedge(6, 7), 2, 7);
    set_table(iedge(7, 4), 2, 4);
    set_table(iedge(0, 4), 2, 8);
    set_table(iedge(1, 5), 2, 9);
    set_table(iedge(2, 6), 2, 10);
    set_table(iedge(3, 7), 2, 11);

    set_table(iface(0, 1, 2, 3), 4, 0);
    set_table(iface(0, 1, 5, 4), 4, 12);
    set_table(iface(1, 2, 6, 5), 4, 9);
    set_table(iface(2, 3, 7, 6), 4, 13);
    set_table(iface(3, 0, 4, 7), 4, 8);
    set_table(iface(4, 5, 6, 7), 4, 7);

    set_iface_partials(0, 1, 2, 3, -4, 0);
    set_iface_partials(0, 1, 5, 4, -4, 12);
    set_iface_partials(1, 2, 6, 5, -4, 9);
    set_iface_partials(2, 3, 7, 6, -4, 13);
    set_iface_partials(3, 0, 4, 7, -4, 8);
    set_iface_partials(4, 5, 6, 7, -4, 7);

    set_table(255, 8, 0);
  }
};


template <class FIELD>
class IRMakeConvexAlgoT : public IRMakeConvexAlgo
{
public:
  //! virtual interface. 
  virtual void execute(ProgressReporter *reporter, FieldHandle fieldh, double isoval, bool lte);
};


template <class FIELD>
void
IRMakeConvexAlgoT<FIELD>::execute(ProgressReporter *reporter,
                                  FieldHandle fieldh, double isoval, bool lte)
{
  double newval = isoval+1.0;
  if (lte) newval = isoval-1.0;

  FIELD *field = dynamic_cast<FIELD*>(fieldh.get_rep());
  typename FIELD::mesh_type *mesh = field->get_typed_mesh().get_rep();

  init_pattern_table();
  
  typename FIELD::mesh_type::Node::array_type onodes(8);
  typename FIELD::value_type v[8];
  
  bool changed;
  do {
//    newval -= 1.0;
    changed = false;
    typename FIELD::mesh_type::Elem::iterator bi, ei;
    mesh->begin(bi); mesh->end(ei);
    while (bi != ei)
    {
      mesh->get_nodes(onodes, *bi);
    
      // Get the values and compute an inside/outside mask.
      unsigned int inside = 0;
      unsigned int inside_count = 0;
      for (unsigned int i = 0; i < onodes.size(); i++)
      {
        field->value(v[i], onodes[i]);
        inside = inside << 1;
        if (v[i] > isoval)
        {
          inside |= 1;
          inside_count++;
        }
      }

      // Invert the mask if we are doing less than.
      if (lte) { inside = ~inside & 0xff; inside_count = 8 - inside_count; }

      const int pattern = pattern_table[inside][0];
      const int which = pattern_table[inside][1];

      if (pattern == -1)
      {
        changed = true;
        for (unsigned int i = 0; i < onodes.size(); i++)
        {
          field->set_value(newval, onodes[i]);
        }
      }
      else if (pattern == -4)
      {
        changed = true;
        const int *ro = IsoRefineAlgo::hex_reorder_table[which];

        for (unsigned int i = 0; i < 4; i++)
        {
          field->set_value(newval, onodes[ro[i]]);
        }
      }

      ++bi;
    }
  } while (changed);
}


} // namespace SCIRun

#endif // ClipField_h
