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

#include <Core/Datatypes/LatVolField.h>
#include <Core/Containers/Array3.h>
#include <Core/Geometry/Point.h>
#include <vector>
#include <set>


namespace SCIRun {

using std::string;

class SCICORESHARE MaskedLatVolMesh : public LatVolMesh
{
private:

public:

  struct MaskedLatIndex;
  friend struct MaskedLatIndex;

  struct MaskedLatIndex
  {
  public:
    MaskedLatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
    MaskedLatIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, 
		   unsigned k) : i_(i), j_(j), k_(k), mesh_(m) {}
    
    operator unsigned() const { 
      if (mesh_ == 0) 
	return i_*j_*k_; 
      else 
	return i_ + ni()*j_ + ni()*nj()*k_;;
    }
    
    // Make sure mesh_ is valid before calling these convience accessors
    unsigned ni() const { ASSERT(mesh_); return mesh_->get_ni(); }
    unsigned nj() const { ASSERT(mesh_); return mesh_->get_nj(); }
    unsigned nk() const { ASSERT(mesh_); return mesh_->get_nk(); }

    unsigned i_, j_, k_;

    // Needs to be here so we can compute a sensible index.
    const MaskedLatVolMesh *mesh_;
  };

  struct CellIndex : public MaskedLatIndex
  {
    CellIndex() : MaskedLatIndex() {}
    CellIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i,j,k) {}

    operator unsigned() const { 
      if (mesh_ == 0) 
	return (i_-1)*(j_-1)*(k_-1); 
      else 
	return i_ + (ni()-1)*j_ + (ni()-1)*(nj()-1)*k_;;
    }

    friend void Pio(Piostream&, CellIndex&);
    friend const TypeDescription* get_type_description(CellIndex *);
    friend const string find_type_name(CellIndex *);
  };

  struct NodeIndex : public MaskedLatIndex
  {
    NodeIndex() : MaskedLatIndex() {}
    NodeIndex(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i,j,k) {}
    static string type_name(int i=-1) 
    { ASSERT(i<1); return "MaskedLatVolMesh::NodeIndex"; }
    friend void Pio(Piostream&, NodeIndex&);
    friend const TypeDescription* get_type_description(NodeIndex *);
    friend const string find_type_name(NodeIndex *);
  };



  struct CellSize : public MaskedLatIndex
  {
  public:
    CellSize() : MaskedLatIndex() {}
    CellSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      //      const MaskedLatVolMesh *const m = 
      //	dynamic_cast<const MaskedLatVolMesh *const>(mesh_);
      //return (m ? i_*j_*k_ - m->num_masked_cells() : i_*j_*k_); 
      return (mesh_ ? i_*j_*k_ - mesh_->num_masked_cells() : i_*j_*k_); 
    } 
  };


  struct NodeSize : public MaskedLatIndex
  {
  public:
    NodeSize() : MaskedLatIndex() {}
    NodeSize(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k) 
      : MaskedLatIndex(m, i, j, k) {}
    operator unsigned () const 
    { 
      //      const MaskedLatVolMesh *const m = 
      //	dynamic_cast<const MaskedLatVolMesh *const>(mesh_);
      //return (m ? i_*j_*k_ - m->num_masked_nodes() : i_*j_*k_); 
      return (mesh_ ? i_*j_*k_ - mesh_->num_masked_nodes() : i_*j_*k_); 
    } 
  };


  struct NodeIter : public MaskedLatIndex
  {
  public:
    NodeIter() : MaskedLatIndex() {}
    NodeIter(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i, j, k) {}

    const NodeIndex &operator *() const 
    { 
      return (const NodeIndex&)(*this); 
    }

    operator unsigned() const 
    { 
      if (mesh_ == 0) 
	return i_*j_*k_; 
      else 
	return i_ + ni()*j_ + ni()*nj()*k_;
    }

    bool operator ==(const LatIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const LatIter &a) const
    {
      return !(*this == a);
    }

    NodeIter &operator++()
    {
      do {
	i_++;
	if (i_ >= mesh_->min_i_+mesh_->get_ni())	{
	  i_ = mesh_->min_i_;
	  j_++;
	  if (j_ >=  mesh_->min_j_+mesh_->get_nj()) {
	    j_ = mesh_->min_j_;
	    k_++;
	  }
	}
      } while (!mesh_->check_valid(*this));
      return *this;
    }

    NodeIter &operator--()
    {
      do { 
	if (i_ == mesh_->min_i_) {
	  i_ = mesh_->min_i_ + mesh_->get_ni();
	  if (j_ == mesh_->min_j_) {
	    j_ = mesh_->min_j_ + mesh_->get_nj();
	    ASSERTMSG(k_ == mesh_->min_k_, "Cant --() from first node!");
	    k_--;
	  }
	  else {
	    j_--;
	  }
	}
	else {
	  i_--;
	}
      } while (!mesh_->check_valid(*this));
      return *this;
    }
  private:
    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
    NodeIter operator--(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }

  };

  struct CellIter : public MaskedLatIndex
  {
  public:
    CellIter() : MaskedLatIndex() {}
    CellIter(const MaskedLatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : MaskedLatIndex(m, i, j, k) {}

    const CellIndex &operator *() const 
    { 
      return (const CellIndex&)(*this); 
    }

    operator unsigned() const 
    { 
      if (mesh_ == 0) 
	return (i_-1)*(j_-1)*(k_-1); 
      else 
	return i_ + (ni()-1)*j_ + (ni()-1)*(nj()-1)*k_;
    }

    bool operator ==(const LatIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const LatIter &a) const
    {
      return !(*this == a);
    }

    CellIter &operator++()
    {
      do {
	i_++;
	if (i_ >= mesh_->min_i_+mesh_->get_ni()-1)	{
	  i_ = mesh_->min_i_;
	  j_++;
	  if (j_ >=  mesh_->min_j_+mesh_->get_nj()-1) {
	    j_ = mesh_->min_j_;
	    k_++;
	  }
	}
      } while (!mesh_->check_valid(*this));
      return *this;
    }

    CellIter &operator--()
    {
      do { 
	if (i_ == mesh_->min_i_) {
	  i_ = mesh_->min_i_ + mesh_->get_ni()-1;
	  if (j_ == mesh_->min_j_) {
	    j_ = mesh_->min_j_ + mesh_->get_nj()-1;
	    ASSERTMSG(k_ == mesh_->min_k_-1, "Cant --() from first node!");
	    k_--;
	  }
	  else {
	    j_--;
	  }
	}
	else {
	  i_--;
	}
      } while (!mesh_->check_valid(*this));
      return *this;
    }
  private:
    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
    CellIter operator--(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }

  };



  struct Node {
    typedef NodeIndex          index_type;
    typedef NodeIter           iterator;
    typedef NodeSize           size_type;
    typedef vector<index_type> array_type;
    typedef RangeNodeIter      range_iter;
  };			
  			
  struct Edge {		
    typedef EdgeIndex<unsigned int>          index_type;
    typedef EdgeIterator<unsigned int>       iterator;
    typedef EdgeIndex<unsigned int>          size_type;
    typedef vector<index_type>               array_type;
  };			
			
  struct Face {		
    typedef FaceIndex<unsigned int>          index_type;
    typedef FaceIterator<unsigned int>       iterator;
    typedef FaceIndex<unsigned int>          size_type;
    typedef vector<index_type>               array_type;
  };			
			
  struct Cell {		
    typedef CellIndex          index_type;
    typedef CellIter           iterator;
    typedef CellSize           size_type;
    typedef vector<index_type> array_type;
    typedef RangeCellIter      range_iter;
  };

  typedef Cell Elem;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;



  MaskedLatVolMesh();
  MaskedLatVolMesh(unsigned int x, unsigned int y, unsigned int z);
  MaskedLatVolMesh(const MaskedLatVolMesh &copy);
  virtual MaskedLatVolMesh *clone() { return new MaskedLatVolMesh(*this); }
  virtual ~MaskedLatVolMesh() {}

  //! get the mesh statistics
  virtual BBox get_bounding_box() const;

  void mask_cell(Cell::index_type);
  void unmask_cell(Cell::index_type);
  

  unsigned num_masked_nodes() const;
  unsigned num_masked_edges() const;
  unsigned num_masked_faces() const;
  unsigned num_masked_cells() const;
		   

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

  // returns 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
			      Cell::index_type &idx) const;
  // return 26 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Node::index_type> > &nbrs, 
			      Node::index_type &idx) const;

  // return 8 pairs in ijk order
  void	get_neighbors_stencil(vector<pair<bool,Cell::index_type> > &nbrs, 
			      Node::index_type &idx) const;


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

  void get_weights(const Point &, Cell::array_type &, vector<double> &);

  void get_point(Point &point, const Node::index_type &index) const
  { get_center(point, index); }

  void get_random_point(Point &p, const Elem::index_type &ei, int seed=0) const
  { ASSERTFAIL("not implemented") }

  //! the double return val is the volume of the Hex.
  double get_gradient_basis(Cell::index_type ci, Vector& g0, Vector& g1,
			    Vector& g2, Vector& g3, Vector& g4,
			    Vector& g5, Vector& g6, Vector& g7)
  { ASSERTFAIL("not implemented") }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

private:
  set<unsigned>	masked_cells_;
  unsigned	masked_nodes_count_;
  unsigned	masked_edges_count_;
  unsigned	masked_faces_count_;

  void		convert_edge_indexing(Edge::index_type, 
				      unsigned &, unsigned &, unsigned &, unsigned &) const;
  bool		update_count(Cell::index_type, bool masking);
  unsigned	num_missing_faces(Cell::index_type);
  bool		check_valid(Node::index_type idx) const;
  bool		check_valid(Edge::index_type idx) const;
  bool		check_valid(Face::index_type idx) const;
  bool		check_valid(Cell::index_type idx) const;
  bool		check_valid(Node::iterator idx) const;
  bool		check_valid(Edge::iterator idx) const;
  bool		check_valid(Face::iterator idx) const;
  bool		check_valid(Cell::iterator idx) const;
  inline bool	check_valid(unsigned i, unsigned j, unsigned k) const
  { return masked_cells_.find(unsigned(Cell::index_type(this,i,j,k))) == masked_cells_.end(); }

  
  // returns a MaskedLatVolMesh
  static Persistent *maker() { return new MaskedLatVolMesh(); }
};

typedef LockingHandle<MaskedLatVolMesh> MaskedLatVolMeshHandle;

const TypeDescription* get_type_description(MaskedLatVolMesh *);
} // namespace SCIRun

#endif // SCI_project_MaskedLatVolMesh_h
