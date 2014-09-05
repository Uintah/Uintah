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
 *  LatVolMesh.h: Templated Mesh defined on a 3D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 *
 */

#ifndef SCI_project_LatVolMesh_h
#define SCI_project_LatVolMesh_h 1

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Containers/StackVector.h>

namespace SCIRun {

using std::string;

class LatVolMesh;

struct LatVolMeshLatIndex
{
public:
  LatVolMeshLatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}

  LatVolMeshLatIndex(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : i_(i), j_(j), k_(k), mesh_(m) {}
    
  operator unsigned() const;
    
  // Make sure mesh_ is valid before calling these convience accessors
  unsigned ni() const;
  unsigned nj() const;
  unsigned nk() const;

  unsigned i_, j_, k_;

  // Needs to be here so we can compute a sensible index.
  const LatVolMesh *mesh_;
};


struct LatVolMeshCellIndex : public LatVolMeshLatIndex
{
  LatVolMeshCellIndex() : LatVolMeshLatIndex() {}
  LatVolMeshCellIndex(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : LatVolMeshLatIndex(m,i,j,k) {}

  operator unsigned() const;

  friend void Pio(Piostream&, LatVolMeshCellIndex&);
  friend const TypeDescription* get_type_description(LatVolMeshCellIndex *);
  friend const string find_type_name(LatVolMeshCellIndex *);
};


struct LatVolMeshNodeIndex : public LatVolMeshLatIndex
{
  LatVolMeshNodeIndex() : LatVolMeshLatIndex() {}
  LatVolMeshNodeIndex(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : LatVolMeshLatIndex(m, i,j,k) {}
  static string type_name(int i=-1) { ASSERT(i<1); return "LatVolMesh::NodeIndex"; }
  friend void Pio(Piostream&, LatVolMeshNodeIndex&);
  friend const TypeDescription* get_type_description(LatVolMeshNodeIndex *);
  friend const string find_type_name(LatVolMeshNodeIndex *);
};


struct LatVolMeshLatSize
{ 
public:
  LatVolMeshLatSize() : i_(0), j_(0), k_(0) {}
  LatVolMeshLatSize(unsigned i, unsigned j, unsigned k) : i_(i), j_(j), k_(k) {}
  operator unsigned() const { return i_*j_*k_; }

  unsigned i_, j_, k_;
};


struct LatVolMeshCellSize : public LatVolMeshLatSize
{
  LatVolMeshCellSize() : LatVolMeshLatSize() {}
  LatVolMeshCellSize(unsigned i, unsigned j, unsigned k) : LatVolMeshLatSize(i,j,k) {}
};


struct LatVolMeshNodeSize : public LatVolMeshLatSize
{
  LatVolMeshNodeSize() : LatVolMeshLatSize() {}
  LatVolMeshNodeSize(unsigned i, unsigned j, unsigned k) : LatVolMeshLatSize(i,j,k) {}
};


struct LatVolMeshLatIter : public LatVolMeshLatIndex
{
  LatVolMeshLatIter() : LatVolMeshLatIndex() {}
  LatVolMeshLatIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : LatVolMeshLatIndex(m, i, j, k) {}
    
  const LatVolMeshLatIndex &operator *() { return *this; }

  bool operator ==(const LatVolMeshLatIter &a) const
  {
    return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
  }

  bool operator !=(const LatVolMeshLatIter &a) const
  {
    return !(*this == a);
  }
};


struct LatVolMeshNodeIter : public LatVolMeshLatIter
{
  LatVolMeshNodeIter() : LatVolMeshLatIter() {}
  LatVolMeshNodeIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : LatVolMeshLatIter(m, i, j, k) {}

  const LatVolMeshNodeIndex &operator *() const { return (const LatVolMeshNodeIndex&)(*this); }

  LatVolMeshNodeIter &operator++();

private:

  LatVolMeshNodeIter operator++(int)
  {
    LatVolMeshNodeIter result(*this);
    operator++();
    return result;
  }
};


struct LatVolMeshCellIter : public LatVolMeshLatIter
{
  LatVolMeshCellIter() : LatVolMeshLatIter() {}
  LatVolMeshCellIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
    : LatVolMeshLatIter(m, i, j, k) {}

  const LatVolMeshCellIndex &operator *() const { return (const LatVolMeshCellIndex&)(*this); }

  operator unsigned() const;

  LatVolMeshCellIter &operator++();

private:

  LatVolMeshCellIter operator++(int)
  {
    LatVolMeshCellIter result(*this);
    operator++();
    return result;
  }
};

//////////////////////////////////////////////////////////////////
// Range Iterators
// 
// These iterators are designed to loop over a sub-set of the mesh
//

struct LatVolMeshRangeNodeIter : public LatVolMeshNodeIter
{
  LatVolMeshRangeNodeIter() : LatVolMeshNodeIter() {}
  // Pre: min, and max are both valid iterators over this mesh
  //      min.A <= max.A where A is (i_, j_, k_)
  LatVolMeshRangeNodeIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k,
                unsigned max_i, unsigned max_j, unsigned max_k)
    : LatVolMeshNodeIter(m, i, j, k), min_i_(i), min_j_(j), min_k_(k),
      max_i_(max_i), max_j_(max_j), max_k_(max_k)
  {}

  const LatVolMeshNodeIndex &operator *() const { return (const LatVolMeshNodeIndex&)(*this); }

  LatVolMeshRangeNodeIter &operator++();

  void end(LatVolMeshNodeIter &end_iter) {
    // This tests is designed for a slice in the xy plane.  If z (or k)
    // is equal then you have this condition.  When this happens you
    // need to increment k so that you will iterate over the xy values.
    if (min_k_ != max_k_)
      end_iter = LatVolMeshNodeIter(mesh_, min_i_, min_j_, max_k_);
    else {
      // We need to check to see if the min and max extents are the same.
      // If they are then set the end iterator such that it will be equal
      // to the beginning.  When they are the same anj for() loop using
      // these iterators [for(;iter != end_iter; iter++)] will never enter.
      if (min_i_ != max_i_ || min_j_ != max_j_)
        end_iter = LatVolMeshNodeIter(mesh_, min_i_, min_j_, max_k_ + 1);
      else
        end_iter = LatVolMeshNodeIter(mesh_, min_i_, min_j_, max_k_);
    }
  }
    
private:
  // The minimum extents
  unsigned min_i_, min_j_, min_k_;
  // The maximum extents
  unsigned max_i_, max_j_, max_k_;

  LatVolMeshRangeNodeIter operator++(int)
  {
    LatVolMeshRangeNodeIter result(*this);
    operator++();
    return result;
  }
};

struct LatVolMeshRangeCellIter : public LatVolMeshCellIter
{
  LatVolMeshRangeCellIter() : LatVolMeshCellIter() {}
  // Pre: min, and max are both valid iterators over this mesh
  //      min.A <= max.A where A is (i_, j_, k_)
  LatVolMeshRangeCellIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k,
                unsigned max_i, unsigned max_j, unsigned max_k)
    : LatVolMeshCellIter(m, i, j, k), min_i_(i), min_j_(j), min_k_(k),
      max_i_(max_i), max_j_(max_j), max_k_(max_k)
  {}

  const LatVolMeshCellIndex &operator *() const { return (const LatVolMeshCellIndex&)(*this); }

  LatVolMeshRangeCellIter &operator++();
  void end(LatVolMeshCellIter &end_iter) {
    // This tests is designed for a slice in the xy plane.  If z (or k)
    // is equal then you have this condition.  When this happens you
    // need to increment k so that you will iterate over the xy values.
    if (min_k_ != max_k_)
      end_iter = LatVolMeshCellIter(mesh_, min_i_, min_j_, max_k_);
    else {
      // We need to check to see if the min and max extents are the same.
      // If they are then set the end iterator such that it will be equal
      // to the beginning.  When they are the same anj for() loop using
      // these iterators [for(;iter != end_iter; iter++)] will never enter.
      if (min_i_ != max_i_ || min_j_ != max_j_)
        end_iter = LatVolMeshCellIter(mesh_, min_i_, min_j_, max_k_ + 1);
      else
        end_iter = LatVolMeshCellIter(mesh_, min_i_, min_j_, max_k_);
    }
  }

private:
  // The minimum extents
  unsigned int min_i_, min_j_, min_k_;
  // The maximum extents
  unsigned int max_i_, max_j_, max_k_;

  LatVolMeshRangeCellIter operator++(int)
  {
    LatVolMeshRangeCellIter result(*this);
    operator++();
    return result;
  }
};

//! Index and Iterator types required for Mesh Concept.
struct LatVolMeshNode {
  typedef LatVolMeshNodeIndex                  index_type;
  typedef LatVolMeshNodeIter                   iterator;
  typedef LatVolMeshNodeSize                   size_type;
  typedef StackVector<index_type, 8>           array_type;
  typedef LatVolMeshRangeNodeIter              range_iter;
};			
  			
struct LatVolMeshEdge {		
  typedef EdgeIndex<unsigned int>          index_type;
  typedef EdgeIterator<unsigned int>       iterator;
  typedef EdgeIndex<unsigned int>          size_type;
  typedef vector<index_type>               array_type;
};			
			
struct LatVolMeshFace {		
  typedef FaceIndex<unsigned int>          index_type;
  typedef FaceIterator<unsigned int>       iterator;
  typedef FaceIndex<unsigned int>          size_type;
  typedef vector<index_type>               array_type;
};			
			
struct LatVolMeshCell {		
  typedef LatVolMeshCellIndex          index_type;
  typedef LatVolMeshCellIter           iterator;
  typedef LatVolMeshCellSize           size_type;
  typedef vector<index_type>           array_type;
  typedef LatVolMeshRangeCellIter          range_iter;
};


class LatVolMesh : public Mesh
{
public:

  friend struct LatVolMeshLatIndex;
  friend struct LatVolMeshNodeIter;
  friend struct LatVolMeshCellIter;
  friend struct LatVolMeshRangeNodeIter;
  friend struct LatVolMeshRangeCellIter;

  // Backwards compatability with interp fields
  typedef LatVolMeshNodeIndex NodeIndex;
  typedef LatVolMeshCellIndex CellIndex;

  typedef LatVolMeshNode Node;
  typedef LatVolMeshEdge Edge;
  typedef LatVolMeshFace Face;
  typedef LatVolMeshCell Cell;
  typedef Cell Elem;
  
  LatVolMesh()
    : min_i_(0), min_j_(0), min_k_(0),
      ni_(1), nj_(1), nk_(1) {}
  LatVolMesh(unsigned x, unsigned y, unsigned z,
	     const Point &min, const Point &max);
  LatVolMesh(LatVolMesh* /* mh */,  // FIXME: Is this constructor broken?
	     unsigned mx, unsigned my, unsigned mz,
	     unsigned x, unsigned y, unsigned z)
    : min_i_(mx), min_j_(my), min_k_(mz),
      ni_(x), nj_(y), nk_(z) {}
  LatVolMesh(const LatVolMesh &copy)
    : min_i_(copy.min_i_), min_j_(copy.min_j_), min_k_(copy.min_k_),
      ni_(copy.get_ni()), nj_(copy.get_nj()), nk_(copy.get_nk()),
      transform_(copy.transform_) {}
  virtual LatVolMesh *clone() { return new LatVolMesh(*this); }
  virtual ~LatVolMesh() {}

  //! get the mesh statistics
  unsigned get_min_i() const { return min_i_; }
  unsigned get_min_j() const { return min_j_; }
  unsigned get_min_k() const { return min_k_; }
  bool get_min(vector<unsigned int>&) const;
  unsigned get_ni() const { return ni_; }
  unsigned get_nj() const { return nj_; }
  unsigned get_nk() const { return nk_; }
  bool get_dim(vector<unsigned int>&) const;
  Vector diagonal() const;

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);
  virtual void get_canonical_transform(Transform &t);

  //! set the mesh statistics
  void set_min_i(unsigned i) {min_i_ = i; }
  void set_min_j(unsigned j) {min_j_ = j; }
  void set_min_k(unsigned k) {min_k_ = k; }
  void set_min(vector<unsigned int> mins);
  void set_ni(unsigned i) { ni_ = i; }
  void set_nj(unsigned j) { nj_ = j; }
  void set_nk(unsigned k) { nk_ = k; }
  void set_dim(vector<unsigned int> dims);

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

  void to_index(Node::index_type &index, unsigned int i);
  void to_index(Edge::index_type &index, unsigned int i) { index = i; }
  void to_index(Face::index_type &index, unsigned int i) { index = i; }
  void to_index(Cell::index_type &index, unsigned int i);

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, const Cell::index_type &) const;
  void get_edges(Edge::array_type &, Face::index_type) const; 
  void get_edges(Edge::array_type &, const Cell::index_type &) const;
  void get_faces(Face::array_type &, const Cell::index_type &) const;

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const
  { ASSERTFAIL("LatVolMesh::get_edges not implemented."); }
  unsigned get_faces(Face::array_type &, Node::index_type) const
  { ASSERTFAIL("LatVolMesh::get_faces not implemented."); }
  unsigned get_faces(Face::array_type &, Edge::index_type) const
  { ASSERTFAIL("LatVolMesh::get_faces not implemented."); }
  unsigned get_cells(Cell::array_type &, Node::index_type) const;
  unsigned get_cells(Cell::array_type &, Edge::index_type)
  { ASSERTFAIL("LatVolMesh::get_cells not implemented."); }
  unsigned get_cells(Cell::array_type &, Face::index_type)
  { ASSERTFAIL("LatVolMesh::get_cells not implemented."); }

  //! return all cell_indecies that overlap the BBox in arr.
  void get_cells(Cell::array_type &arr, const BBox &box);
  //! returns the min and max indices that fall within or on the BBox
  void get_cells(Cell::index_type &begin, Cell::index_type &end,
		 const BBox &bbox);
  void get_nodes(Node::index_type &begin, Node::index_type &end,
		 const BBox &bbox);
  
  bool get_neighbor(Cell::index_type &neighbor,
		    const Cell::index_type &from,
		    Face::index_type face) const;

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, Face::index_type) const;
  void get_center(Point &, const Cell::index_type &) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const Node::index_type &idx) const;
  double get_size(Edge::index_type idx) const;
  double get_size(Face::index_type idx) const;
  double get_size(const Cell::index_type &idx) const;
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(const Cell::index_type &i) const { return get_size(i); };

  int get_valence(const Node::index_type &idx) const;
  int get_valence(Edge::index_type idx) const;
  int get_valence(Face::index_type idx) const;
  int get_valence(const Cell::index_type &idx) const;
  
  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , Face::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, Cell::array_type &l, double *w);

  void get_point(Point &p, const Node::index_type &i) const
  { get_center(p, i); }

  void get_normal(Vector &/*normal*/, const Node::index_type &/*index*/) const
  { ASSERTFAIL("LatVolMesh::get_normal not implemented") }

  void get_random_point(Point &, const Elem::index_type &, int seed=0) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 3; }

protected:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned min_i_, min_j_, min_k_;
  //! the Node::index_type space extents of a LatVolMesh
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned ni_, nj_, nk_;

  Transform transform_;

  // returns a LatVolMesh
  static Persistent *maker() { return new LatVolMesh(); }
};

typedef LockingHandle<LatVolMesh> LatVolMeshHandle;

const TypeDescription* get_type_description(LatVolMesh *);
const TypeDescription* get_type_description(LatVolMesh::Node *);
const TypeDescription* get_type_description(LatVolMesh::Edge *);
const TypeDescription* get_type_description(LatVolMesh::Face *);
const TypeDescription* get_type_description(LatVolMesh::Cell *);

const TypeDescription* get_type_description(LatVolMeshCellIndex *);


std::ostream& operator<<(std::ostream& os, const LatVolMeshLatIndex& n);
std::ostream& operator<<(std::ostream& os, const LatVolMeshLatSize& s);

} // namespace SCIRun

#endif // SCI_project_LatVolMesh_h
