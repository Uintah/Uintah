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
 *  MaskedLatVolMesh.h:
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   Feb 2003
 *
 *  Copyright (C) 2002 SCI Group
 *
 */



#ifndef SCI_project_MaskedLatVolMesh_h
#define SCI_project_MaskedLatVolMesh_h 1

#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

class MaskedLatVolMesh;

struct MLVMIndex
{
public:
  MLVMIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
  MLVMIndex(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, 
                 unsigned int k) : i_(i), j_(j), k_(k), mesh_(m) {}
        
  unsigned int i_, j_, k_;

  // Needs to be here so we can compute a sensible index.
  const MaskedLatVolMesh *mesh_;
};


struct MLVMCellIndex : public MLVMIndex
{
  MLVMCellIndex() : MLVMIndex() {}
  MLVMCellIndex(const MaskedLatVolMesh *m, 
                unsigned int i, unsigned int j, unsigned int k)
    : MLVMIndex(m,i,j,k) {}

  // The 'operator unsigned()' cast is used to convert a MLVMCellIndex
  // into a single scalar, in this case an 'index' value that is used
  // to index into a field.
  operator unsigned() const;

  bool operator ==(const MLVMCellIndex &a) const
  {
    return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
  }

  bool operator !=(const MLVMCellIndex &a) const
  {
    return !(*this == a);
  }

  static string type_name(int i=-1) 
  { ASSERT(i<1); return "MaskedLatVolMesh::CellIndex"; }
  friend void Pio(Piostream&, MLVMCellIndex&);
  friend const TypeDescription* get_type_description(MLVMCellIndex *);
  friend const string find_type_name(MLVMCellIndex *);
};


struct MLVMNodeIndex : public MLVMIndex
{
  MLVMNodeIndex() : MLVMIndex() {}
  MLVMNodeIndex(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
    : MLVMIndex(m,i,j,k) {}

  // The 'operator unsigned()' cast is used to convert a MLVMNodeIndex
  // into a single scalar, in this case an 'index' value that is used
  // to index into a field.
  operator unsigned() const;

  bool operator ==(const MLVMIndex &a) const
  {
    return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
  }

  bool operator !=(const MLVMIndex &a) const
  {
    return !(*this == a);
  }

  static string type_name(int i=-1) 
  { ASSERT(i<1); return "MaskedLatVolMesh::NodeIndex"; }
  friend void Pio(Piostream&, MLVMNodeIndex&);
  friend const TypeDescription* get_type_description(MLVMNodeIndex *);
  friend const string find_type_name(MLVMNodeIndex *);
};


struct MLVMEdgeIndex : public MLVMIndex
{
  MLVMEdgeIndex() : MLVMIndex(), dir_(0) {}
  MLVMEdgeIndex(const MaskedLatVolMesh *m, 
                unsigned int i, unsigned int j, unsigned int k, unsigned int dir)
    : MLVMIndex(m, i,j,k) , dir_(dir){}

  // The 'operator unsigned()' cast is used to convert a MLVMEdgeIndex
  // into a single scalar, in this case an 'index' value that is used
  // to index into a field.
  operator unsigned() const;

  bool operator ==(const MLVMEdgeIndex &a) const
  {
    return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
            mesh_ == a.mesh_ && dir_ == a.dir_);
  }

  bool operator !=(const MLVMEdgeIndex &a) const
  {
    return !(*this == a);
  }

  static string type_name(int i=-1) { 
    ASSERT(i<1); return "MaskedLatVolMesh::EdgeIndex";
  }

  unsigned int dir_;

  friend void Pio(Piostream&, MLVMEdgeIndex&);
  friend const TypeDescription* get_type_description(MLVMEdgeIndex *);
  friend const string find_type_name(MLVMEdgeIndex *);

};


struct MLVMFaceIndex : public MLVMIndex
{
  MLVMFaceIndex() : MLVMIndex() {}
  MLVMFaceIndex(const MaskedLatVolMesh *m,
                unsigned int i, unsigned int j, unsigned int k, unsigned int dir)
    : MLVMIndex(m, i,j,k) , dir_(dir){}

  // The 'operator unsigned()' cast is used to convert a MLVMFaceIndex
  // into a single scalar, in this case an 'index' value that is used
  // to index into a field.
  operator unsigned() const;

  bool operator ==(const MLVMFaceIndex &a) const
  {
    return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
            mesh_ == a.mesh_ && dir_ == a.dir_);
  }

  bool operator !=(const MLVMFaceIndex &a) const
  {
    return !(*this == a);
  }

  static string type_name(int i=-1) { 
    ASSERT(i<1); return "MaskedLatVolMesh::FaceIndex";
  }

  unsigned int dir_;

  friend void Pio(Piostream&, MLVMFaceIndex&);
  friend const TypeDescription* get_type_description(MLVMFaceIndex *);
  friend const string find_type_name(MLVMFaceIndex *);
};


struct MLVMCellSize : public MLVMIndex
{
public:
  MLVMCellSize() : MLVMIndex() {}
  MLVMCellSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k) 
    : MLVMIndex(m, i, j, k) {}
  operator unsigned() const 
  { 
    return i_*j_*k_;
  } 
};

struct MLVMNodeSize : public MLVMIndex
{
public:
  MLVMNodeSize() : MLVMIndex() {}
  MLVMNodeSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k) 
    : MLVMIndex(m, i, j, k) {}
  operator unsigned() const 
  { 
    return i_*j_*k_;
  } 
};


struct MLVMEdgeSize : public MLVMIndex
{
public:
  MLVMEdgeSize() : MLVMIndex() {}
  MLVMEdgeSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k) 
    : MLVMIndex(m, i, j, k) {}
  operator unsigned() const 
  { 
    return (i_-1)*j_*k_ + i_*(j_-1)*k_ + i_*j_*(k_-1);
  } 
};


struct MLVMFaceSize : public MLVMIndex
{
public:
  MLVMFaceSize() : MLVMIndex() {}
  MLVMFaceSize(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k) 
    : MLVMIndex(m, i, j, k) {}
  operator unsigned() const 
  { 
    return i_*(j_-1)*(k_-1) + (i_-1)*j_*(k_-1) + (i_-1)*(j_-1)*k_; 
  } 
};


struct MLVMNodeIter : public MLVMIndex
{
public:
  MLVMNodeIter() : MLVMIndex() {}
  MLVMNodeIter(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
    : MLVMIndex(m, i, j, k) {}

  const MLVMNodeIndex &operator *() const
  { 
    return (const MLVMNodeIndex&)(*this); 
  }

  operator unsigned() const;

  MLVMNodeIter &operator++();
  MLVMNodeIter &operator--();

  void next();
  void prev();

private:
  MLVMNodeIter operator++(int)
  {
    MLVMNodeIter result(*this);
    operator++();
    return result;
  }
  MLVMNodeIter operator--(int)
  {
    MLVMNodeIter result(*this);
    operator--();
    return result;
  }
};


struct MLVMEdgeIter : public MLVMEdgeIndex
{
public:
  MLVMEdgeIter() : MLVMEdgeIndex() {}
  MLVMEdgeIter(const MaskedLatVolMesh *m, 
           unsigned int i, unsigned int j, 
           unsigned int k, unsigned int dir)
    : MLVMEdgeIndex(m, i, j, k,dir) {}

  const MLVMEdgeIndex &operator *() const 
  { 
    return (const MLVMEdgeIndex&)(*this); 
  }

  bool operator ==(const MLVMEdgeIter &a) const
  {
    return (i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && 
            mesh_ == a.mesh_ && dir_ == a.dir_);
  }

  bool operator !=(const MLVMEdgeIter &a) const
  {
    return !(*this == a);
  }

  MLVMEdgeIter &operator++();
  MLVMEdgeIter &operator--();

  void next();
  void prev();

private:    
  MLVMEdgeIter operator++(int)
  {
    MLVMEdgeIter result(*this);
    operator++();
    return result;
  }
  MLVMEdgeIter operator--(int)
  {
    MLVMEdgeIter result(*this);
    operator--();
    return result;
  }
};


struct MLVMFaceIter : public MLVMFaceIndex
{
public:
  MLVMFaceIter() : MLVMFaceIndex() {}
  MLVMFaceIter(const MaskedLatVolMesh *m, 
           unsigned int i, unsigned int j, unsigned int k, unsigned int dir)
    : MLVMFaceIndex(m, i, j, k, dir){}

  const MLVMFaceIndex &operator *() const 
  { 
    return (const MLVMFaceIndex&)(*this); 
  }

  MLVMFaceIter &operator++();
  MLVMFaceIter &operator--();

  void next();
  void prev();

private:
  MLVMFaceIter operator++(int)
  {
    MLVMFaceIter result(*this);
    operator++();
    return result;
  }
  MLVMFaceIter operator--(int)
  {
    MLVMFaceIter result(*this);
    operator--();
    return result;
  }
};




struct MLVMCellIter : public MLVMIndex
{
public:
  MLVMCellIter() : MLVMIndex() {}
  MLVMCellIter(const MaskedLatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
    : MLVMIndex(m, i, j, k) {}

  const MLVMCellIndex &operator *() const 
  { 
    return (const MLVMCellIndex&)(*this); 
  }

  operator unsigned() const;

  MLVMCellIter &operator++();
  MLVMCellIter &operator--();

  void next();
  void prev();

private:      
  MLVMCellIter operator++(int)
  {
    MLVMCellIter result(*this);
    operator++();
    return result;
  }
  MLVMCellIter operator--(int)
  {
    MLVMCellIter result(*this);
    operator++();
    return result;
  }

};



struct MaskedLatVolMeshNode {
  typedef MLVMNodeIndex		index_type;
  typedef MLVMNodeIter            iterator;
  typedef MLVMNodeSize            size_type;
  typedef vector<index_type>  array_type;
};			
  			
struct MaskedLatVolMeshEdge {		
  typedef MLVMEdgeIndex		index_type;
  typedef MLVMEdgeIter		iterator;
  typedef MLVMEdgeSize		size_type;
  typedef vector<index_type>  array_type;
};			
			
struct MaskedLatVolMeshFace {		
  typedef MLVMFaceIndex           index_type;
  typedef MLVMFaceIter            iterator;
  typedef MLVMFaceSize            size_type;
  typedef vector<index_type>  array_type;
};			
			
struct MaskedLatVolMeshCell {		
  typedef MLVMCellIndex           index_type;
  typedef MLVMCellIter            iterator;
  typedef MLVMCellSize            size_type;
  typedef vector<index_type>  array_type;
};



class MaskedLatVolMesh : public LatVolMesh
{
private:

public:

  struct MLVMIndex;
  friend struct MLVMIndex;

  // Backwards compatability with interp fields
  typedef MLVMNodeIndex NodeIndex;
  typedef MLVMCellIndex CellIndex;

  typedef MaskedLatVolMeshNode Node;
  typedef MaskedLatVolMeshEdge Edge;
  typedef MaskedLatVolMeshFace Face;
  typedef MaskedLatVolMeshCell Cell;
  typedef Cell Elem;

  friend class MLVMNodeIter;
  friend class MLVMCellIter;
  friend class MLVMEdgeIter;
  friend class MLVMFaceIter;

  friend class MLVMNodeIndex;
  friend class MLVMCellIndex;
  friend class MLVMEdgeIndex;
  friend class MLVMFaceIndex;

  MaskedLatVolMesh();
  MaskedLatVolMesh(unsigned int x, unsigned int y, unsigned int z,
		   const Point &min, const Point &max);
  MaskedLatVolMesh(const MaskedLatVolMesh &copy);
  virtual MaskedLatVolMesh *clone() { return new MaskedLatVolMesh(*this); }
  virtual ~MaskedLatVolMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;

  //! Methods specific to MaskedLatVolMesh
  void mask_node(Node::index_type);
  void mask_edge(Edge::index_type);
  void mask_face(Face::index_type);
  void mask_cell(Cell::index_type);
  void unmask_node(Node::index_type);
  void unmask_edge(Edge::index_type);
  void unmask_face(Face::index_type);
  void unmask_cell(Cell::index_type);
  //! Special Method to Reset Mesh
  void unmask_everything();  
  
  unsigned int num_masked_nodes() const;
  unsigned int num_masked_edges() const;
  unsigned int num_masked_faces() const;
  unsigned int num_masked_cells() const;

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
  void to_index(Edge::index_type &index, unsigned int i);
  void to_index(Face::index_type &index, unsigned int i);
  void to_index(Cell::index_type &index, unsigned int i);

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const;
  void get_edges(Edge::array_type &, Face::index_type) const; 
  void get_edges(Edge::array_type &, Cell::index_type) const;
  void get_faces(Face::array_type &, Cell::index_type) const;

  // returns 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
							  Cell::index_type idx) const;
  // return 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Node::index_type> > &nbrs, 
							  Node::index_type idx) const;
  
  // return 8 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
							  Node::index_type idx) const;


  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, const Edge::index_type &) const;
  void get_center(Point &, const Face::index_type &) const;
  void get_center(Point &, const Cell::index_type &) const;

  double get_size(Node::index_type idx) const;
  double get_size(Edge::index_type idx) const;
  double get_size(Face::index_type idx) const;
  double get_size(Cell::index_type idx) const;
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point &p, Cell::array_type &l, double *w);

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }

  void get_random_point(Point &/*p*/, const Elem::index_type &/*ei*/, int /*seed=0*/) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type /*ci*/, Vector& /*g0*/, Vector& /*g1*/,
			    Vector& /*g2*/, Vector& /*g3*/, Vector& /*g4*/,
			    Vector& /*g5*/, Vector& /*g6*/, Vector& /*g7*/)
  { ASSERTFAIL("not implemented") }

  void get_normal(Vector &/*normal*/, Node::index_type /*index*/) const
  { ASSERTFAIL("not implemented") }


  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

  unsigned int	get_sequential_node_index(const Node::index_type idx);

private:
  unsigned int	synchronized_;
  map<Node::index_type, unsigned int>	nodes_;
  Mutex					node_lock_;
  set<unsigned int> masked_cells_;
  unsigned	    masked_nodes_count_;
  unsigned	    masked_edges_count_;
  unsigned	    masked_faces_count_;

  bool		update_count(Cell::index_type, bool masking);
  unsigned int	num_missing_faces(Cell::index_type);
  bool		check_valid(Node::index_type idx) const;
  bool		check_valid(Edge::index_type idx) const;
  bool		check_valid(Face::index_type idx) const;
  bool		check_valid(Cell::index_type idx) const;
  bool		check_valid(Node::iterator idx) const;
  bool		check_valid(Edge::iterator idx) const;
  bool		check_valid(Face::iterator idx) const;    
  bool		check_valid(Cell::iterator idx) const; 
#if 0
  inline bool	check_valid(unsigned int i, unsigned int j, unsigned int k) const
  { 
    return (masked_cells_.find(unsigned int(Cell::index_type(this,i,j,k))) == masked_cells_.end()); 
  }
#endif
  inline bool	check_valid(int i, int j, int k) const
  { 
    if ((i >= int(min_i_)) && (i < (int(min_i_ + ni_) - 1)) &&
	(j >= int(min_j_)) && (j < (int(min_j_ + nj_) - 1)) &&
	(k >= int(min_k_)) && (k < (int(min_k_ + nk_) - 1)) &&
	(masked_cells_.find(unsigned(Cell::index_type(this,i,j,k))) == masked_cells_.end()))
      {
	return true;
      }
    return false;
  }
  
  
  // returns a MaskedLatVolMesh
  static Persistent *maker() { return new MaskedLatVolMesh(); }
};

typedef LockingHandle<MaskedLatVolMesh> MaskedLatVolMeshHandle;

const TypeDescription* get_type_description(MaskedLatVolMesh *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Node *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Edge *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Face *);
const TypeDescription* get_type_description(MaskedLatVolMesh::Cell *);

const TypeDescription* get_type_description(MLVMCellIndex *);

std::ostream& operator<<(std::ostream& os, const MLVMNodeIndex& n);
std::ostream& operator<<(std::ostream& os, const MLVMCellIndex& n);
std::ostream& operator<<(std::ostream& os, const MLVMNodeSize& n);
std::ostream& operator<<(std::ostream& os, const MLVMCellSize& n);

} // namespace SCIRun

#endif // SCI_project_MaskedLatVolMesh_h
