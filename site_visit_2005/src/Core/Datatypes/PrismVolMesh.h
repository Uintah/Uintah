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


/*
 *  PrismVolMesh.h: Templated Meshs defined on a 3D Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   July 2003
 *
 *  Copyright (C) 2003 SCI Group
 *
 */

#ifndef SCI_project_PrismVolMesh_h
#define SCI_project_PrismVolMesh_h 1

#include <Core/Thread/Mutex.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Datatypes/SearchGrid.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>
#include <Core/Persistent/PersistentSTL.h>
#include <Core/Math/MinMax.h>
#include <sci_hash_set.h>
#include <sci_hash_map.h>
#include <sgi_stl_warnings_off.h>
#include <set>
#include <sgi_stl_warnings_on.h>


namespace SCIRun {

#define TRI_NNODES 3
#define QUAD_NNODES 4

#define PRISM_NNODES 6
#define PRISM_NEDGES 9
#define PRISM_NFACES 5
#define PRISM_NTRIS  2
#define PRISM_NQUADS 3

#define isTRI(i)  ((i) < PRISM_NTRIS)
#define isQUAD(i) ( PRISM_NTRIS <= (i) && (i) < PRISM_NTRIS + PRISM_NQUADS)

static const unsigned int
PrismFaceTable[PRISM_NFACES][4] = { { 0, 1, 2, 6 },
				    { 5, 4, 3, 6 },
				    { 4, 5, 2, 1 },
				    { 3, 4, 1, 0 },
				    { 0, 2, 5, 3 } };

static const unsigned int
PrismNodeNeighborTable[PRISM_NNODES][3] = { { 1,2,3 },
					    { 0,2,4 },
					    { 0,1,5 },
					    { 0,4,5 },
					    { 1,3,5 },
					    { 2,3,4 } };

typedef unsigned int under_type;

//! Index and Iterator types required for Mesh Concept.
struct PrismVolMeshNode {
  typedef NodeIndex<under_type>       index_type;
  typedef NodeIterator<under_type>    iterator;
  typedef NodeIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};					

struct PrismVolMeshCell {
  typedef CellIndex<under_type>       index_type;
  typedef CellIterator<under_type>    iterator;
  typedef CellIndex<under_type>       size_type;
  typedef vector<index_type>          array_type;
};


class PrismVolMesh : public Mesh
{
public:

  typedef PrismVolMeshNode Node;
  typedef PrismVolMeshCell Cell;

  // Used for hashing operations below
  static const int sizeof_uint = sizeof(unsigned int) * 8; // in bits

  //! An edge is indexed via the cells structure.
  //! There are 9 unique edges in each cell of 6 nodes.
  //! Therefore, the edge index / 9 == cell index
  //! And, the edge index % 9 == which edge in that cell
  //! Edges indices are stored in a hash_set and a hash_multiset.
  //! The hash_set stores shared edges only once.
  //! The hash_multiset stores all shared edges together.   
  struct Edge {				
    typedef EdgeIndex<under_type>       index_type;

    //! edgei return the two nodes make the edge
    static pair<Node::index_type, Node::index_type> edgei(index_type idx)
    { 
      const int base = (idx / PRISM_NEDGES) * PRISM_NNODES;
      switch (idx % PRISM_NEDGES)
      {
      default:
      case 0: return pair<Node::index_type,Node::index_type>(base+0,base+1);
      case 1: return pair<Node::index_type,Node::index_type>(base+0,base+2);
      case 2: return pair<Node::index_type,Node::index_type>(base+0,base+3);
      case 3: return pair<Node::index_type,Node::index_type>(base+1,base+2);
      case 4: return pair<Node::index_type,Node::index_type>(base+1,base+4);
      case 5: return pair<Node::index_type,Node::index_type>(base+2,base+5);
      case 6: return pair<Node::index_type,Node::index_type>(base+3,base+4);
      case 7: return pair<Node::index_type,Node::index_type>(base+3,base+5);
      case 8: return pair<Node::index_type,Node::index_type>(base+4,base+5);
      }
    }

    //! A fucntor that returns a boolean indicating weather two
    //! edges indices share the same nodes, and thus the same edge in space
    //! Used as a template parameter to STL containers typedef'd below
    struct eqEdge : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      eqEdge(const vector<under_type> &cells) : 
	cells_(cells) {};

      // Since the indicies of the nodes can be in any order, we need
      // to order them before we do comparisons.  This can be done
      // simply for two items using Min and Max.  This works, because
      // index_type is a single integral value.
      bool operator()(index_type ei1, index_type ei2) const
      {
	const pair<index_type, index_type> e1 = edgei(ei1), e2 = edgei(ei2);
	return (Max(cells_[e1.first], cells_[e1.second]) == 
		Max(cells_[e2.first], cells_[e2.second]) &&
		Min(cells_[e1.first], cells_[e1.second]) == 
		Min(cells_[e2.first], cells_[e2.second]));
      };
    };
      
    //! A fucntor that returns a boolean indicating weather two
    //! edges indices are less than each other.
    //! Used as a template parameter to STL containers typedef'd below
    struct lessEdge : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      lessEdge(const vector<under_type> &cells) : 
	cells_(cells) {};

      static bool lessthen(const vector<under_type> &cells, index_type ei1, index_type ei2) {
	const pair<index_type, index_type> e1 = edgei(ei1), e2 = edgei(ei2);
        index_type e1min = Min(cells[e1.first], cells[e1.second]);
        index_type e2min = Min(cells[e2.first], cells[e2.second]);
        if (e1min == e2min) {
          index_type e1max = Max(cells[e1.first], cells[e1.second]);
          index_type e2max = Max(cells[e2.first], cells[e2.second]);
          return  e1max < e2max;
        } else {
          return e1min < e2min;
        }
      }

      bool operator()(index_type ei1, index_type ei2) const
      {
        return lessthen(cells_, ei1, ei2);
      };
    };
      
#ifdef HAVE_HASH_SET
    //! A functor that hashes an edge index according to the node
    //! indices of that edge
    //! Used as a template parameter to STL hash_[set,map] containers
    struct CellEdgeHasher : public unary_function<size_t, index_type>
    {
    private:
      const vector<under_type> &cells_;
    public:
      CellEdgeHasher(const vector<under_type> &cells) : 
	cells_(cells) {};
      static const int size = sizeof_uint / 2; // in bits
      static const int mask = (~(unsigned int)0) >> (sizeof_uint - size);
      size_t operator()(index_type cell) const 
      { 
	pair<index_type,index_type> e = edgei(cell);
	const int n0 = cells_[e.first] & mask;
	const int n1 = cells_[e.second] & mask;
	return Min(n0, n1) << size | Max(n0, n1);
      }
#ifdef __ECC

      // These are particularly needed by ICC's hash stuff
      static const size_t bucket_size = 4;
      static const size_t min_buckets = 8;
      
      // This is a less than function.
      bool operator()(index_type ei1, index_type ei2) const {
        return lessEdge::lessthen(cells_, ei1, ei2);
      }
#endif // endif ifdef __ICC
    };   

#ifdef __ECC
    // The comparator function needs to be a member of CellEdgeHasher
    typedef hash_multiset<index_type, CellEdgeHasher> HalfEdgeSet;
    typedef hash_set<index_type, CellEdgeHasher> EdgeSet;
#else
    typedef eqEdge EdgeComparitor;
    typedef hash_multiset<index_type, CellEdgeHasher, EdgeComparitor> HalfEdgeSet;
    typedef hash_set<index_type, CellEdgeHasher, EdgeComparitor> EdgeSet;
#endif // end ifdef __ECC
#else // ifdef HAVE_HASH_SET
    typedef lessEdge EdgeComparitor;
    typedef multiset<index_type, EdgeComparitor> HalfEdgeSet;
    typedef set<index_type, EdgeComparitor> EdgeSet;
#endif
    //! This iterator will traverse each shared edge once in no 
    //! particular order.
    typedef EdgeSet::iterator		iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
  


  struct Face 
  {				
    typedef FaceIndex<under_type> index_type;
    
    struct eqFace : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      eqFace(const vector<under_type> &cells) : 
	cells_(cells) {};
      bool operator()(index_type fi1, index_type fi2) const
      {
	const int f1_offset = fi1 % PRISM_NFACES;
	const int f1_base = fi1 / PRISM_NFACES * PRISM_NNODES;

	const int f2_offset = fi2 % PRISM_NFACES;
	const int f2_base = fi2 / PRISM_NFACES * PRISM_NNODES;

	const under_type f1_n0 =
	  cells_[f1_base + PrismFaceTable[f1_offset][0] ];
	const under_type f1_n1 =
	  cells_[f1_base + PrismFaceTable[f1_offset][1] ];
	const under_type f1_n2 =
	  cells_[f1_base + PrismFaceTable[f1_offset][2] ];
	
	const under_type f2_n0 =
	  cells_[f2_base + PrismFaceTable[f2_offset][0] ];
	const under_type f2_n1 =
	  cells_[f2_base + PrismFaceTable[f2_offset][1] ];
	const under_type f2_n2 =
	  cells_[f2_base + PrismFaceTable[f2_offset][2] ];
	
	if( isTRI(f1_offset) && isTRI(f2_offset) ) {
	  return (Max(f1_n0, f1_n1, f1_n2) == Max(f2_n0, f2_n1, f2_n2) &&
		  Mid(f1_n0, f1_n1, f1_n2) == Mid(f2_n0, f2_n1, f2_n2) &&
		  Min(f1_n0, f1_n1, f1_n2) == Min(f2_n0, f2_n1, f2_n2));
	} else if( isQUAD(f1_offset) && isQUAD(f2_offset) ) {
	  const under_type f1_n3 =
	    cells_[f1_base + PrismFaceTable[f1_offset][3] ];
	  const under_type f2_n3 =
	    cells_[f2_base + PrismFaceTable[f2_offset][3] ];

	  return
	    (Max( Max(f1_n0, f1_n1), Max(f1_n2, f1_n3)) ==
	     Max( Max(f2_n0, f2_n1), Max(f2_n2, f1_n3)) &&

	     Max( Mid(f1_n0, f1_n1, f1_n2), Mid(f1_n1, f1_n2, f1_n3)) ==
	     Max( Mid(f2_n0, f2_n1, f2_n2), Mid(f2_n1, f2_n2, f2_n3)) &&

	     Min( Mid(f1_n0, f1_n1, f1_n2), Mid(f1_n1, f1_n2, f1_n3)) ==
	     Min( Mid(f2_n0, f2_n1, f2_n2), Mid(f2_n1, f2_n2, f2_n3)) &&

	     Min( Min(f1_n0, f1_n1), Min(f1_n2, f1_n3)) ==
	     Min( Min(f2_n0, f2_n1), Min(f2_n2, f1_n3)) );
	} else 
	  return false;
      }
    };
    
    struct lessFace : public binary_function<index_type, index_type, bool>
    {
    private:
      const vector<under_type> &cells_;
    public:
      lessFace(const vector<under_type> &cells) : 
	cells_(cells) {};
      static bool lessthen(const vector<under_type> &cells, index_type fi1, index_type fi2)
      {
	const int f1_offset = fi1 % PRISM_NFACES;
	const int f1_base = fi1 / PRISM_NFACES * PRISM_NNODES;

	const int f2_offset = fi2 % PRISM_NFACES;
	const int f2_base = fi2 / PRISM_NFACES * PRISM_NNODES;

	const under_type f1_n0 =
	  cells[f1_base + PrismFaceTable[f1_offset][0] ];
	const under_type f1_n1 =
	  cells[f1_base + PrismFaceTable[f1_offset][1] ];
	const under_type f1_n2 =
	  cells[f1_base + PrismFaceTable[f1_offset][2] ];
	
	const under_type f2_n0 =
	  cells[f2_base + PrismFaceTable[f2_offset][0] ];
	const under_type f2_n1 =
	  cells[f2_base + PrismFaceTable[f2_offset][1] ];
	const under_type f2_n2 =
	  cells[f2_base + PrismFaceTable[f2_offset][2] ];
	
	if( isTRI(f1_offset) && isTRI(f2_offset) ) {
          index_type f1max = Max(f1_n0, f1_n1, f1_n2);
          index_type f2max = Max(f2_n0, f2_n1, f2_n2);
	  if (f1max == f2max) {
            index_type f1mid = Mid(f1_n0, f1_n1, f1_n2);
            index_type f2mid = Mid(f2_n0, f2_n1, f2_n2);
            if (f1mid == f2mid)
              return Min(f1_n0, f1_n1, f1_n2) <
                     Min(f2_n0, f2_n1, f2_n2);
            else
              return f1mid < f2mid;
          } else
            return f1max < f2max;
	} else if( isQUAD(f1_offset) && isQUAD(f2_offset) ) {
	  const under_type f1_n3 =
	    cells[f1_base + PrismFaceTable[f1_offset][3] ];
	  const under_type f2_n3 =
	    cells[f2_base + PrismFaceTable[f2_offset][3] ];

          index_type f1max1 = Max( Max(f1_n0, f1_n1),
                                   Max(f1_n2, f1_n3));
          index_type f2max1 = Max( Max(f2_n0, f2_n1),
                                   Max(f2_n2, f1_n3));
          if (f1max1 == f2max1) {
            index_type f1max2 = Max( Mid(f1_n0, f1_n1, f1_n2),
                                     Mid(f1_n1, f1_n2, f1_n3));
            index_type f2max2 = Max( Mid(f2_n0, f2_n1, f2_n2),
                                     Mid(f2_n1, f2_n2, f2_n3));
            if (f1max2 == f2max2) {
              index_type f1min1 = Min( Mid(f1_n0, f1_n1, f1_n2),
                                       Mid(f1_n1, f1_n2, f1_n3));
              index_type f2min1 = Min( Mid(f2_n0, f2_n1, f2_n2),
                                       Mid(f2_n1, f2_n2, f2_n3));
              if (f1min1 == f2min1)
                return Min( Min(f1_n0, f1_n1), Min(f1_n2, f1_n3)) <
                       Min( Min(f2_n0, f2_n1), Min(f2_n2, f1_n3));
              else
                return f1min1 < f2min1;
            } else {
              return f1max2 < f2max2;
            }
          } else {
            return f1max1 < f2max1;
          }
	} else if( isQUAD(f1_offset) && isTRI(f2_offset) ) {
          // We need to have a way of differentiating when f1 and f2
          // are not both quads or tris.
	  return true;
        } else
          return false;
      }
      bool operator()(index_type fi1, index_type fi2) const
      {
        return lessthen(cells_, fi1, fi2);
      }
    };
    
#ifdef HAVE_HASH_SET
    struct CellFaceHasher: public unary_function<size_t, index_type>
    {
    private:
      const vector<under_type> &cells_;
    public:
      CellFaceHasher(const vector<under_type> &cells) : 
	cells_(cells) {};
      static const int size = sizeof_uint / 4; // in bits
      static const int mask = (~(unsigned int)0) >> (sizeof_uint - size);
      size_t operator()(index_type idx) const 
      {
	const unsigned int offset = idx%PRISM_NFACES;
	const unsigned int base   = idx / PRISM_NFACES * PRISM_NNODES; // base cell index

	const under_type n0 = cells_[base + PrismFaceTable[offset][0]] & mask;
	const under_type n1 = cells_[base + PrismFaceTable[offset][1]] & mask;
	const under_type n2 = cells_[base + PrismFaceTable[offset][2]] & mask;

	if( isTRI(offset) )
	  return Min(n0,n1,n2)<<size*2 | Mid(n0,n1,n2)<<size | Max(n0,n1,n2);
	else if( isQUAD(offset) ) {
	  const under_type n3 = cells_[base + PrismFaceTable[offset][3]] & mask;

	  return
	    Min(Min(n0,n1),Min(n2,n3))<<size*3 |
		     
	    Min( Mid(n0,n1,n2),Mid(n1,n2,n3))<<size*2 |

	    Max( Mid(n0,n1,n2),Mid(n1,n2,n3))<<size |

	    Max(Max(n0,n1),Max(n2,n3));
	} else
	  return 0;
      }
      
#ifdef __ECC

      // These are particularly needed by ICC's hash stuff
      static const size_t bucket_size = 4;
      static const size_t min_buckets = 8;
      
      // This is a less than function.
      bool operator()(index_type fi1, index_type fi2) const {
        return lessFace::lessthen(cells_, fi1, fi2);
      }
#endif // endif ifdef __ICC

    };
#ifdef __ECC
    // The comparator function needs to be a member of CellFaceHasher
    typedef hash_multiset<index_type, CellFaceHasher> HalfFaceSet;
    typedef hash_set<index_type, CellFaceHasher> FaceSet;
#else
    typedef eqFace FaceComparitor;
    typedef hash_multiset<index_type, CellFaceHasher,FaceComparitor> HalfFaceSet;
    typedef hash_set<index_type, CellFaceHasher, FaceComparitor> FaceSet;
#endif // end ifdef __ECC
#else // ifdef HAVE_HASH_SET
    typedef lessFace FaceComparitor;
    typedef multiset<index_type, FaceComparitor> HalfFaceSet;
    typedef set<index_type, FaceComparitor> FaceSet;
#endif
    typedef FaceSet::iterator		iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };					
  

  typedef Cell Elem;
  enum { ELEMENTS_E = CELLS_E };

  PrismVolMesh();
  PrismVolMesh(const PrismVolMesh &copy);
  virtual PrismVolMesh *clone() { return new PrismVolMesh(*this); }
  virtual ~PrismVolMesh();

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  bool get_dim(vector<unsigned int>&) const { return false;  }

  void begin(Node::iterator &) const;
  void begin(Edge::iterator &) const;
  void begin(Face::iterator &) const;
  void begin(Cell::iterator &) const;

  void end(Node::iterator &) const;
  void end(Edge::iterator &) const;
  void end(Face::iterator &) const;
  void end(Cell::iterator &) const;

  void size(Node::size_type &) const;
  void size(Edge::size_type &) const;
  void size(Face::size_type &) const;
  void size(Cell::size_type &) const;

  void to_index(Node::index_type &index, unsigned int i) const { index = i; }
  void to_index(Edge::index_type &index, unsigned int i) const { index = i; }
  void to_index(Face::index_type &index, unsigned int i) const { index = i; }
  void to_index(Cell::index_type &index, unsigned int i) const { index = i; }

  void get_nodes(Node::array_type &array, Edge::index_type idx) const;
  void get_nodes(Node::array_type &array, Face::index_type idx) const;
  void get_nodes(Node::array_type &array, Cell::index_type idx) const;

  void get_edges(Edge::array_type &array, Node::index_type idx) const;
  void get_edges(Edge::array_type &array, Face::index_type idx) const;
  void get_edges(Edge::array_type &array, Cell::index_type idx) const;

  void get_faces(Face::array_type &array, Node::index_type idx) const;
  void get_faces(Face::array_type &array, Edge::index_type idx) const;
  void get_faces(Face::array_type &array, Cell::index_type idx) const;

  void get_cells(Cell::array_type &array, Node::index_type idx) const;
  void get_cells(Cell::array_type &array, Edge::index_type idx) const;
  void get_cells(Cell::array_type &array, Face::index_type idx) const;
  
  // This function is redundant, the next one can be used with less parameters 
  bool get_neighbor(Cell::index_type &neighbor, Cell::index_type from,
		   Face::index_type idx) const;
  // Use this one instead
  bool get_neighbor(Face::index_type &neighbor, Face::index_type idx) const;
  void get_neighbors(Cell::array_type &array, Cell::index_type idx) const;
  void get_neighbors(vector<Node::index_type> &array,
                     Node::index_type idx) const;

  void get_center(Point &result, Node::index_type idx) const;
  void get_center(Point &result, Edge::index_type idx) const;
  void get_center(Point &result, Face::index_type idx) const;
  void get_center(Point &result, Cell::index_type idx) const;
  void get_center(Point &result, Node::array_type& arr) const;


  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type ra;
    get_nodes(ra, idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);

    if( isTRI( idx % PRISM_NFACES ) )
      return (Cross(p0-p1,p2-p0)).length()*0.5;
    else if( isQUAD( idx % PRISM_NFACES ) ){
      const Point &p3 = point(ra[3]);
      return ((Cross(p0-p1,p2-p0)).length()+
	      (Cross(p0-p3,p2-p0)).length())*0.5;

    }
    ASSERTFAIL("Index not TRI or QUAD.");
  }
  double get_size(Cell::index_type idx) const
  {
    Node::array_type ra(PRISM_NNODES);
    get_nodes(ra,idx);
    const Point &p0 = point(ra[0]);
    const Point &p1 = point(ra[1]);
    const Point &p2 = point(ra[2]);
    const Point &p3 = point(ra[3]);
    const Point &p4 = point(ra[4]);
    const Point &p5 = point(ra[5]);

    return ((Dot(Cross(p1-p0,p2-p0),p4-p0)) +
	    (Dot(Cross(p2-p0,p3-p0),p4-p0)) +
	    (Dot(Cross(p3-p2,p4-p2),p5-p2)) ) * 0.1666666666666666;
  } 
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area  (Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };



  int get_valence(Node::index_type idx) const
  {
    Node::array_type arr;
    get_neighbors(arr, idx);
    return arr.size();
  }
  int get_valence(Edge::index_type /*idx*/) const { return 0; }
  int get_valence(Face::index_type idx) const
  {
    Face::index_type tmp;
    return (get_neighbor(tmp, idx) ? 1 : 0);
  }
  int get_valence(Cell::index_type idx) const 
  {
    Cell::array_type arr;
    get_neighbors(arr, idx);
    return arr.size();
  }


  //! return false if point is out of range.
  bool locate(Node::index_type &loc, const Point &p);
  bool locate(Edge::index_type &loc, const Point &p);
  bool locate(Face::index_type &loc, const Point &p);
  bool locate(Cell::index_type &loc, const Point &p);

  double polygon_area(const Node::array_type &ni, const Vector N) const;
  void orient(Cell::index_type idx);
  bool inside(Cell::index_type idx, const Point &p);

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  { ASSERTFAIL("PrismVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double * )
  { ASSERTFAIL("PrismVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, Cell::array_type &l, double *w);

  void get_point(Point &result, Node::index_type index) const
  { result = points_[index]; }

  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  void set_point(const Point &point, Node::index_type index)
  { points_[index] = point; }

  void get_random_point(Point &p, const Cell::index_type &ei, 
			int seed=0) const;

  //! the double return val is the volume of the prism.
  double get_gradient_basis(Cell::index_type /*ci*/,
			    Vector& /*g0*/, Vector& /*g1*/,
			    Vector& /*g2*/, Vector& /*g3*/,
			    Vector& /*g4*/, Vector& /*g5*/)
  { ASSERTFAIL("not implemented") }

  //! function to test if at least one of cell's nodes are in supplied range
  inline bool test_nodes_range(Cell::index_type ci,
			       unsigned int sn,
			       unsigned int en)
  {
    for ( int i=0; i<PRISM_NNODES; i++ )
    {
      if (sn<=cells_[ci*PRISM_NNODES+i] && cells_[ci*PRISM_NNODES+i]<en )
      {
	return true;
      }
    }
    return false;
  }

  template <class Iter, class Functor>
  void fill_points(Iter begin, Iter end, Functor fill_ftor);
  template <class Iter, class Functor>
  void fill_cells(Iter begin, Iter end, Functor fill_ftor);


  void flip(Cell::index_type, bool recalculate = false);
  void rewind_mesh();


  virtual bool		synchronize(unsigned int);

  //! Persistent IO
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;


  // Extra functionality needed by this specific geometry.
  void			set_nodes(Node::array_type &, Cell::index_type);

  Node::index_type	add_point(const Point &p);
  Node::index_type	add_find_point(const Point &p, double err = 1.0e-3);

  Elem::index_type	add_prism(Node::index_type a, 
				  Node::index_type b,
				  Node::index_type c,
				  Node::index_type d,
				  Node::index_type e,
				  Node::index_type f);
  Elem::index_type	add_prism(const Point &p0,
				  const Point &p1,
				  const Point &p2,
				  const Point &p3,
				  const Point &p4,
				  const Point &p5);
  Elem::index_type	add_elem(Node::array_type a);

  void node_reserve(size_t s) { points_.reserve(s); }
  void elem_reserve(size_t s) { cells_.reserve(s*6); }


  //! Subdivision methods
  void			delete_cells(set<int> &to_delete);
  
  bool			is_edge(Node::index_type n0,
				Node::index_type n1,
				Edge::array_type *edges = 0);

  bool			is_face(Node::index_type n0,
				Node::index_type n1,
				Node::index_type n2,
				Face::array_type *faces = 0);

  bool			is_face(Node::index_type n0,
				Node::index_type n1,
				Node::index_type n2,
				Node::index_type n3,
				Face::array_type *faces = 0);


  virtual bool		is_editable() const { return true; }
  virtual int           dimensionality() const { return 3; }

protected:
  const Point &point(Node::index_type idx) const { return points_[idx]; }

  void			compute_node_neighbors();
  void			compute_edges();
  void			compute_faces();
  void			compute_grid();

  //! Used to recompute data for individual cells
  void			create_cell_edges(Cell::index_type);
  void			delete_cell_edges(Cell::index_type);
  void			create_cell_faces(Cell::index_type);
  void			delete_cell_faces(Cell::index_type);
  void			create_cell_node_neighbors(Cell::index_type);
  void			delete_cell_node_neighbors(Cell::index_type);

  Elem::index_type	mod_prism(Cell::index_type cell, 
				  Node::index_type a,
				  Node::index_type b,
				  Node::index_type c,
				  Node::index_type d,
				  Node::index_type e,
				  Node::index_type f);
  


  //! all the vertices
  vector<Point>		points_;
  Mutex			points_lock_;

  //! each 6 indecies make up a prism
  vector<under_type>	cells_;
  Mutex			cells_lock_;

  typedef LockingHandle<Edge::HalfEdgeSet> HalfEdgeSetHandle;
  typedef LockingHandle<Edge::EdgeSet> EdgeSetHandle;
#ifdef HAVE_HASH_SET
  Edge::CellEdgeHasher	edge_hasher_;
#ifndef __ECC
  Edge::EdgeComparitor	edge_comp_;
#endif
#else // ifdef HAVE_HASH_SET
  Edge::EdgeComparitor  edge_comp_;
#endif
  
  Edge::HalfEdgeSet	all_edges_;

#if defined(__digital__) || defined(_AIX) || defined(__ECC)
  mutable
#endif
  Edge::EdgeSet		edges_;
  Mutex			edge_lock_;

  typedef LockingHandle<Face::HalfFaceSet> HalfFaceSetHandle;
  typedef LockingHandle<Face::FaceSet> FaceSetHandle;
#ifdef HAVE_HASH_SET
  Face::CellFaceHasher	face_hasher_;
#ifndef __ECC
  Face::FaceComparitor  face_comp_;
#endif
#else // ifdef HAVE_HASH_SET
  Face::FaceComparitor	face_comp_;
#endif

  Face::HalfFaceSet	all_faces_;

#if defined(__digital__) || defined(_AIX) || defined(__ECC)
  mutable
#endif
  Face::FaceSet		faces_;
  Mutex			face_lock_;

  typedef vector<vector<Cell::index_type> > NodeNeighborMap;
  //  typedef LockingHandle<NodeMap> NodeMapHandle;
  NodeNeighborMap	node_neighbors_;
  Mutex			node_neighbor_lock_;

  //! This grid is used as an acceleration structure to expedite calls
  //!  to locate.  For each cell in the grid, we store a list of which
  //!  prisms overlap that grid cell -- to find the prism which contains a
  //!  point, we simply find which grid cell contains that point, and
  //!  then search just those prisms that overlap that grid cell.
  //!  The grid is only built if synchronize(Mesh::LOCATE_E) is called.
  LockingHandle<SearchGrid>  grid_;
  Mutex                      grid_lock_; // Bad traffic!
  Cell::index_type           locate_cache_;

  unsigned int		synchronized_;
};


// Handle type for PrismVolMesh mesh.
typedef LockingHandle<PrismVolMesh> PrismVolMeshHandle;



template <class Iter, class Functor>
void
PrismVolMesh::fill_points(Iter begin, Iter end, Functor fill_ftor) {
  points_lock_.lock();
  Iter iter = begin;
  points_.resize(end - begin); // resize to the new size
  vector<Point>::iterator piter = points_.begin();
  while (iter != end) {
    *piter = fill_ftor(*iter);
    ++piter; ++iter;
  }
  points_lock_.unlock();
  //dirty_ = true; // FIXME
}

template <class Iter, class Functor>
void
PrismVolMesh::fill_cells(Iter begin, Iter end, Functor fill_ftor) {
  cells_lock_.lock();
  Iter iter = begin;
  cells_.resize((end - begin) * PRISM_NNODES); // resize to the new size
  vector<under_type>::iterator citer = cells_.begin();
  while (iter != end) {
    int *nodes = fill_ftor(*iter); // returns an array of length NNODES

    for( int i=0; i<PRISM_NNODES; i++ ) {
      *citer = nodes[i];
      ++citer;
    }
    ++iter;
  }
  cells_lock_.unlock();
  //dirty_ = true; // FIXME
}


const TypeDescription* get_type_description(PrismVolMesh *);
const TypeDescription* get_type_description(PrismVolMesh::Node *);
const TypeDescription* get_type_description(PrismVolMesh::Edge *);
const TypeDescription* get_type_description(PrismVolMesh::Face *);
const TypeDescription* get_type_description(PrismVolMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_PrismVolMesh_h
