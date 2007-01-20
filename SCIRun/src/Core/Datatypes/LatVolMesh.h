/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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


#ifndef CORE_DATATYPES_LATVOLMESH_H
#define CORE_DATATYPES_LATVOLMESH_H 1

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Containers/StackVector.h>
#include <Core/Geometry/BBox.h>

namespace SCIRun {

using std::string;

template <class Basis>
class LatVolMesh : public Mesh
{
public:
  typedef LockingHandle<LatVolMesh<Basis> > handle_type;
  typedef Basis                             basis_type;

  struct LatIndex;
  friend struct LatIndex;

  struct LatIndex
  {
  public:
    LatIndex() : i_(0), j_(0), k_(0), mesh_(0) {}
    LatIndex(const LatVolMesh *m, unsigned i, unsigned j,
             unsigned k) : i_(i), j_(j), k_(k), mesh_(m) {}

    operator unsigned() const {
      ASSERT(mesh_);
      return i_ + ni()*j_ + ni()*nj()*k_;
    }

    std::ostream& str_render(std::ostream& os) const
    {
      os << "[" << i_ << "," << j_ << "," << k_ << "]";
      return os;
    }

    // Make sure mesh_ is valid before calling these convience accessors
    unsigned int ni() const { ASSERT(mesh_); return mesh_->get_ni(); }
    unsigned int nj() const { ASSERT(mesh_); return mesh_->get_nj(); }
    unsigned int nk() const { ASSERT(mesh_); return mesh_->get_nk(); }

    unsigned i_, j_, k_;

    // Needs to be here so we can compute a sensible index.
    const LatVolMesh *mesh_;
  };

  struct CellIndex : public LatIndex
  {
    CellIndex() : LatIndex() {}
    CellIndex(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : LatIndex(m, i,j,k) {}

    operator unsigned() const {
      ASSERT(this->mesh_);
      return (this->i_ + (this->ni()-1)*this->j_ +
              (this->ni()-1)*(this->nj()-1)*this->k_);
    }
  };

  struct NodeIndex : public LatIndex
  {
    NodeIndex() : LatIndex() {}
    NodeIndex(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : LatIndex(m, i,j,k) {}
    static string type_name(int i=-1) {
      ASSERT(i < 1);
      return LatVolMesh<Basis>::type_name(-1) + "::NodeIndex";
    }
  };


  struct LatSize
  {
  public:
    LatSize() : i_(0), j_(0), k_(0) {}
    LatSize(unsigned int i, unsigned int j, unsigned int k) : i_(i), j_(j), k_(k) {}

    operator unsigned() const { return i_*j_*k_; }

    std::ostream& str_render(std::ostream& os) const {
      os << i_*j_*k_ << " (" << i_ << " x " << j_ << " x " << k_ << ")";
      return os;
    }

    unsigned i_, j_, k_;
  };

  struct CellSize : public LatSize
  {
    CellSize() : LatSize() {}
    CellSize(unsigned int i, unsigned int j, unsigned int k) : LatSize(i,j,k) {}
  };

  struct NodeSize : public LatSize
  {
    NodeSize() : LatSize() {}
    NodeSize(unsigned int i, unsigned int j, unsigned int k) : LatSize(i,j,k) {}
  };


  struct LatIter : public LatIndex
  {
    LatIter() : LatIndex() {}
    LatIter(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : LatIndex(m, i, j, k) {}

    const LatIndex &operator *() { return *this; }

    bool operator ==(const LatIter &a) const
    {
      return (this->i_ == a.i_ && this->j_ == a.j_ &&
              this->k_ == a.k_ && this->mesh_ == a.mesh_);
    }

    bool operator !=(const LatIter &a) const
    {
      return !(*this == a);
    }
  };


  struct NodeIter : public LatIter
  {
    NodeIter() : LatIter() {}
    NodeIter(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : LatIter(m, i, j, k) {}

    const NodeIndex &operator *() const { return (const NodeIndex&)(*this); }

    NodeIter &operator++()
    {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_+this->mesh_->get_ni())        {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
        if (this->j_ >=  this->mesh_->min_j_+this->mesh_->get_nj()) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
        }
      }
      return *this;
    }

  private:

    NodeIter operator++(int)
    {
      NodeIter result(*this);
      operator++();
      return result;
    }
  };


  struct CellIter : public LatIter
  {
    CellIter() : LatIter() {}
    CellIter(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k)
      : LatIter(m, i, j, k) {}

    const CellIndex &operator *() const { return (const CellIndex&)(*this); }

    operator unsigned() const {
      ASSERT(this->mesh_);
      return this->i_ + (this->ni()-1)*this->j_ + (this->ni()-1)*(this->nj()-1)*this->k_;;
    }

    CellIter &operator++()
    {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_+this->ni()-1) {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
        if (this->j_ >= this->mesh_->min_j_+this->nj()-1) {
          this->j_ = this->mesh_->min_j_;
          this->k_++;
        }
      }
      return *this;
    }

  private:

    CellIter operator++(int)
    {
      CellIter result(*this);
      operator++();
      return result;
    }
  };

  //////////////////////////////////////////////////////////////////
  // Range Iterators
  //
  // These iterators are designed to loop over a sub-set of the mesh
  //

  struct RangeNodeIter : public NodeIter
  {
    RangeNodeIter() : NodeIter() {}
    // Pre: min, and max are both valid iterators over this mesh
    //      min.A <= max.A where A is (i_, j_, k_)
    RangeNodeIter(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k,
                  unsigned int max_i, unsigned int max_j, unsigned int max_k)
      : NodeIter(m, i, j, k), min_i_(i), min_j_(j), min_k_(k),
        max_i_(max_i), max_j_(max_j), max_k_(max_k)
    {}

    const NodeIndex &operator *() const { return (const NodeIndex&)(*this); }

    RangeNodeIter &operator++()
    {
      this->i_++;
      // Did i_ loop over the line
      // mesh_->min_x is the starting point of the x range for the mesh
      // min_i_ is the starting point of the range on x
      // max_i_ is the ending point of the range on x
      if (this->i_ >= this->mesh_->min_i_ + max_i_) {
        // set i_ to the beginning of the range
        this->i_ = min_i_;
        this->j_++;
        // Did j_ loop over the face
        // mesh_->min_j_ is the starting point of the y range for the mesh
        // min_j is the starting point of the range on y
        // max_j is the ending point of the range on y
        if (this->j_ >= this->mesh_->min_j_ + max_j_) {
          this->j_ = min_j_;
          this->k_++;
        }
      }
      return *this;
    }

    void end(NodeIter &end_iter) {
      // This tests is designed for a slice in the xy plane.  If z (or k)
      // is equal then you have this condition.  When this happens you
      // need to increment k so that you will iterate over the xy values.
      if (min_k_ != max_k_)
        end_iter = NodeIter(this->mesh_, min_i_, min_j_, max_k_);
      else {
        // We need to check to see if the min and max extents are the same.
        // If they are then set the end iterator such that it will be equal
        // to the beginning.  When they are the same anj for() loop using
        // these iterators [for(;iter != end_iter; iter++)] will never enter.
        if (min_i_ != max_i_ || min_j_ != max_j_)
          end_iter = NodeIter(this->mesh_, min_i_, min_j_, max_k_ + 1);
        else
          end_iter = NodeIter(this->mesh_, min_i_, min_j_, max_k_);
      }
    }

  private:
    // The minimum extents
    unsigned int min_i_, min_j_, min_k_;
    // The maximum extents
    unsigned int max_i_, max_j_, max_k_;

    RangeNodeIter operator++(int)
    {
      RangeNodeIter result(*this);
      operator++();
      return result;
    }
  };

  struct RangeCellIter : public CellIter
  {
    RangeCellIter() : CellIter() {}
    // Pre: min, and max are both valid iterators over this mesh
    //      min.A <= max.A where A is (i_, j_, k_)
    RangeCellIter(const LatVolMesh *m, unsigned int i, unsigned int j, unsigned int k,
                  unsigned int max_i, unsigned int max_j, unsigned int max_k)
      : CellIter(m, i, j, k), min_i_(i), min_j_(j), min_k_(k),
        max_i_(max_i), max_j_(max_j), max_k_(max_k)
    {}

    const CellIndex &operator *() const { return (const CellIndex&)(*this); }

    RangeCellIter &operator++()
    {
      this->i_++;
      // Did i_ loop over the line
      // mesh_->min_x is the starting point of the x range for the mesh
      // min_i_ is the starting point of the range on x
      // max_i_ is the ending point of the range on x
      if (this->i_ >= this->mesh_->min_i_ + max_i_) {
        // set i_ to the beginning of the range
        this->i_ = min_i_;
        this->j_++;
        // Did j_ loop over the face
        // mesh_->min_j_ is the starting point of the y range for the mesh
        // min_j is the starting point of the range on y
        // max_j is the ending point of the range on y
        if (this->j_ >= this->mesh_->min_j_ + max_j_) {
          this->j_ = min_j_;
          this->k_++;
        }
      }
      return *this;
    }

    void end(CellIter &end_iter) {
      // This tests is designed for a slice in the xy plane.  If z (or k)
      // is equal then you have this condition.  When this happens you
      // need to increment k so that you will iterate over the xy values.
      if (min_k_ != max_k_)
        end_iter = CellIter(this->mesh_, min_i_, min_j_, max_k_);
      else {
        // We need to check to see if the min and max extents are the same.
        // If they are then set the end iterator such that it will be equal
        // to the beginning.  When they are the same anj for() loop using
        // these iterators [for(;iter != end_iter; iter++)] will never enter.
        if (min_i_ != max_i_ || min_j_ != max_j_)
          end_iter = CellIter(this->mesh_, min_i_, min_j_, max_k_ + 1);
        else
          end_iter = CellIter(this->mesh_, min_i_, min_j_, max_k_);
      }
    }

  private:
    // The minimum extents
    unsigned int min_i_, min_j_, min_k_;
    // The maximum extents
    unsigned int max_i_, max_j_, max_k_;

    RangeCellIter operator++(int)
    {
      RangeCellIter result(*this);
      operator++();
      return result;
    }
  };

  //typedef LatIndex        under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex                  index_type;
    typedef NodeIter                   iterator;
    typedef NodeSize                   size_type;
    typedef StackVector<index_type, 8> array_type;
    typedef RangeNodeIter              range_iter;
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
  typedef Face DElem;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;

  friend class RangeCellIter;
  friend class RangeNodeIter;

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const LatVolMesh<Basis>& msh,
                const typename Cell::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned int node0_index() const {
      return (index_.i_ + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned int node1_index() const {
      return (index_.i_+ 1 + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned int node2_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);

    }
    inline
    unsigned int node3_index() const {
      return (index_.i_ + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*index_.k_);
    }
    inline
    unsigned int node4_index() const {
      return (index_.i_ + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }
    inline
    unsigned int node5_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*index_.j_ +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }

    inline
    unsigned int node6_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }
    inline
    unsigned int node7_index() const {
      return (index_.i_ + mesh_.get_ni()*(index_.j_ + 1) +
              mesh_.get_ni()*mesh_.get_nj()*(index_.k_ + 1));
    }

    inline
    const Point node0() const {
      Point p(index_.i_, index_.j_, index_.k_);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node1() const {
      Point p(index_.i_ + 1, index_.j_, index_.k_);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node2() const {
      Point p(index_.i_ + 1, index_.j_ + 1, index_.k_);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node3() const {
      Point p(index_.i_, index_.j_ + 1, index_.k_);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node4() const {
      Point p(index_.i_, index_.j_, index_.k_+ 1);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node5() const {
      Point p(index_.i_ + 1, index_.j_, index_.k_+ 1);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node6() const {
      Point p(index_.i_ + 1, index_.j_ + 1, index_.k_+ 1);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node7() const {
      Point p(index_.i_, index_.j_ + 1, index_.k_+ 1);
      return mesh_.transform_.project(p);
    }

  private:
    const LatVolMesh<Basis>          &mesh_;
    const typename Cell::index_type  index_;
  };


  LatVolMesh() :
    min_i_(0),
    min_j_(0),
    min_k_(0),
    ni_(1),
    nj_(1),
    nk_(1)
  {}
  LatVolMesh(unsigned int x, unsigned int y, unsigned int z,
             const Point &min, const Point &max);
  LatVolMesh(LatVolMesh* /* mh */,  // FIXME: Is this constructor broken?
             unsigned int mx, unsigned int my, unsigned int mz,
             unsigned int x, unsigned int y, unsigned int z) :
    min_i_(mx),
    min_j_(my),
    min_k_(mz),
    ni_(x),
    nj_(y),
    nk_(z)
  {}
  LatVolMesh(const LatVolMesh &copy) :
    min_i_(copy.min_i_),
    min_j_(copy.min_j_),
    min_k_(copy.min_k_),
    ni_(copy.get_ni()),
    nj_(copy.get_nj()),
    nk_(copy.get_nk()),
    transform_(copy.transform_),
    basis_(copy.basis_)
  {}
  virtual LatVolMesh *clone() { return new LatVolMesh(*this); }
  virtual ~LatVolMesh() {}

  virtual int basis_order() { return (basis_.polynomial_order()); }

  Basis &get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned int which_edge,
                       unsigned int div_per_unit) const
  {
    // Needs to match unit_edges in Basis/HexTrilinearLgn.cc
    // compare get_nodes order to the basis order
    int emap[] = {0, 2, 8, 10, 3, 1, 11, 9, 4, 5, 7, 6};
    basis_.approx_edge(emap[which_edge], div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type ci,
                       unsigned int which_face,
                       unsigned int div_per_unit) const
  {
    // Needs to match unit_edges in Basis/HexTrilinearLgn.cc
    // compare get_nodes order to the basis order
    int fmap[] = {0, 5, 4, 2, 1, 3};
    basis_.approx_face(fmap[which_face], div_per_unit, coords);
  }

  bool get_coords(vector<double> &coords,
                  const Point &p,
                  typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    return basis_.get_coords(coords, p, ed);
  }

  void interpolate(Point &pt, const vector<double> &coords,
                   typename Elem::index_type idx) const
  {
    ElemData ed(*this, idx);
    pt = basis_.interpolate(coords, ed);
  }

  // get the Jacobian matrix
  template<class VECTOR>
  void derivate(const vector<double> &coords,
                typename Cell::index_type idx,
                VECTOR &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  //! get the mesh statistics
  unsigned int get_min_i() const { return min_i_; }
  unsigned int get_min_j() const { return min_j_; }
  unsigned int get_min_k() const { return min_k_; }
  bool get_min(vector<unsigned int>&) const;
  unsigned int get_ni() const { return ni_; }
  unsigned int get_nj() const { return nj_; }
  unsigned int get_nk() const { return nk_; }
  bool get_dim(vector<unsigned int>&) const;
  Vector diagonal() const;

  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);
  virtual void get_canonical_transform(Transform &t);

  //! set the mesh statistics
  void set_min_i(unsigned int i) {min_i_ = i; }
  void set_min_j(unsigned int j) {min_j_ = j; }
  void set_min_k(unsigned int k) {min_k_ = k; }
  void set_min(vector<unsigned int> mins);
  void set_ni(unsigned int i) { ni_ = i; }
  void set_nj(unsigned int j) { nj_ = j; }
  void set_nk(unsigned int k) { nk_ = k; }
  void set_dim(vector<unsigned int> dims);

  void begin(typename Node::iterator &) const;
  void begin(typename Edge::iterator &) const;
  void begin(typename Face::iterator &) const;
  void begin(typename Cell::iterator &) const;

  void end(typename Node::iterator &) const;
  void end(typename Edge::iterator &) const;
  void end(typename Face::iterator &) const;
  void end(typename Cell::iterator &) const;

  void size(typename Node::size_type &) const;
  void size(typename Edge::size_type &) const;
  void size(typename Face::size_type &) const;
  void size(typename Cell::size_type &) const;

  void to_index(typename Node::index_type &index, unsigned int i) const;
  void to_index(typename Edge::index_type &index, unsigned int i) const
  { index = i; }
  void to_index(typename Face::index_type &index, unsigned int i) const
  { index = i; }
  void to_index(typename Cell::index_type &index, unsigned int i) const;

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const;
  void get_nodes(typename Node::array_type &,
                 const typename Cell::index_type &) const;
  void get_edges(typename Edge::array_type &,
                 typename Face::index_type) const;
  void get_edges(typename Edge::array_type &,
                 const typename Cell::index_type &) const;
  void get_faces(typename Face::array_type &,
                 const typename Cell::index_type &) const;

  //! get the parent element(s) of the given index
  void get_elems(typename Elem::array_type &result,
                 const typename Node::index_type &idx) const;
  void get_elems(typename Elem::array_type &result,
                 const typename Edge::index_type &idx) const;
  void get_elems(typename Elem::array_type &result,
                 const typename Face::index_type &idx) const;


  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  const typename Elem::index_type &idx) const
  {
    get_faces(result, idx);
  }

  //! return all cell_indecies that overlap the BBox in arr.
  void get_cells(typename Cell::array_type &arr, const BBox &box);
  //! returns the min and max indices that fall within or on the BBox
  void get_cells(typename Cell::index_type &begin,
                 typename Cell::index_type &end,
                 const BBox &bbox);
  void get_nodes(typename Node::index_type &begin,
                 typename Node::index_type &end,
                 const BBox &bbox);

  bool get_neighbor(typename Cell::index_type &neighbor,
                    const typename Cell::index_type &from,
                    typename Face::index_type face) const;

  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, typename Edge::index_type) const;
  void get_center(Point &, typename Face::index_type) const;
  void get_center(Point &, const typename Cell::index_type &) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const typename Node::index_type &idx) const;
  double get_size(typename Edge::index_type idx) const;
  double get_size(typename Face::index_type idx) const;
  double get_size(const typename Cell::index_type &idx) const;
  double get_length(typename Edge::index_type idx) const
  { return get_size(idx); };
  double get_area(typename Face::index_type idx) const
  { return get_size(idx); };
  double get_volume(const typename Cell::index_type &i) const
  { return get_size(i); };

  int get_valence(const typename Node::index_type &idx) const;
  int get_valence(typename Edge::index_type idx) const;
  int get_valence(typename Face::index_type idx) const;
  int get_valence(const typename Cell::index_type &idx) const;

  bool locate(typename Node::index_type &, const Point &) const;
  bool locate(typename Edge::index_type &, const Point &) const
  { return false; }
  bool locate(typename Face::index_type &, const Point &) const
  { return false; }
  bool locate(typename Cell::index_type &, const Point &) const;

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point & , typename Face::array_type & , double * )
  {ASSERTFAIL("LatVolMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &p, typename Cell::array_type &l, double *w);

  void get_point(Point &p, const typename Node::index_type &i) const
  { get_center(p, i); }

  void get_normal(Vector &, const typename Node::index_type &) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &, vector<double> &, typename Elem::index_type,
                  unsigned int)
  { ASSERTFAIL("This mesh type does not have element normals."); }
  void get_random_point(Point &,
                        const typename Elem::index_type &,
                        MusilRNG &rng) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_idlv;
  static MeshTypeID mesh_idlv;
  
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;
  static const TypeDescription* cell_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* node_type_description();
  static const TypeDescription* elem_type_description()
  { return cell_type_description(); }
  static const TypeDescription* cell_index_type_description();
  static const TypeDescription* node_index_type_description();

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans)
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 3; }
  virtual int topology_geometry() const { return (STRUCTURED | REGULAR); }

  // returns a LatVolMesh
  static Persistent *maker() { return scinew LatVolMesh(); }
  static MeshHandle mesh_maker() { return scinew LatVolMesh(); }
  static MeshHandle latvol_maker(unsigned int x, unsigned int y, unsigned int z, const Point& min, const Point& max) { return scinew LatVolMesh(x,y,z,min,max); }

protected:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned int min_i_, min_j_, min_k_;
  //! the Node::index_type space extents of a LatVolMesh
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned int ni_, nj_, nk_;

  Transform transform_;
  Basis     basis_;
  
  
public:
  //! VIRTUAL INTERFACE FUNCTIONS
  
  virtual bool has_virtual_interface() const;
  
  virtual void size(Mesh::VNode::size_type& size) const;
  virtual void size(Mesh::VEdge::size_type& size) const;
  virtual void size(Mesh::VFace::size_type& size) const;
  virtual void size(Mesh::VCell::size_type& size) const;
  virtual void size(Mesh::VElem::size_type& size) const;
  virtual void size(Mesh::VDElem::size_type& size) const;
  
  virtual void get_nodes(VNode::array_type& nodes, VEdge::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VFace::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VCell::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VElem::index_type i) const;
  virtual void get_nodes(VNode::array_type& nodes, VDElem::index_type i) const;
  
  virtual void get_edges(VEdge::array_type& edges, VFace::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VCell::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VElem::index_type i) const;
  virtual void get_edges(VEdge::array_type& edges, VDElem::index_type i) const;

  virtual void get_faces(VFace::array_type& faces, VCell::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VElem::index_type i) const;
  virtual void get_faces(VFace::array_type& faces, VDElem::index_type i) const;

  virtual void get_cells(VCell::array_type& cells, VNode::index_type i) const;  
  virtual void get_cells(VCell::array_type& cells, VEdge::index_type i) const;  
  virtual void get_cells(VCell::array_type& cells, VFace::index_type i) const;  
  virtual void get_cells(VCell::array_type& cells, VElem::index_type i) const;  
  virtual void get_cells(VCell::array_type& cells, VDElem::index_type i) const;  
  
  virtual void get_elems(VElem::array_type& elems, VNode::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VEdge::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VFace::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VCell::index_type i) const;
  virtual void get_elems(VElem::array_type& elems, VDElem::index_type i) const;

  virtual void get_delems(VDElem::array_type& delems, VFace::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VCell::index_type i) const;
  virtual void get_delems(VDElem::array_type& delems, VElem::index_type i) const;

  //! Get the center of a certain mesh element
  virtual void get_center(Point &point, VNode::index_type i) const;
  virtual void get_center(Point &point, VEdge::index_type i) const;
  virtual void get_center(Point &point, VFace::index_type i) const;
  virtual void get_center(Point &point, VCell::index_type i) const;
  virtual void get_center(Point &point, VElem::index_type i) const;
  virtual void get_center(Point &point, VDElem::index_type i) const;

  virtual double get_size(VNode::index_type i) const;
  virtual double get_size(VEdge::index_type i) const;
  virtual double get_size(VFace::index_type i) const;
  virtual double get_size(VCell::index_type i) const;
  virtual double get_size(VElem::index_type i) const;
  virtual double get_size(VDElem::index_type i) const;
  
  virtual void get_weights(const Point& p,VNode::array_type& nodes,
                                                vector<double>& weights) const;
  virtual void get_weights(const Point& p,VElem::array_type& elems,
                                                vector<double>& weights) const;
                                                  
  virtual bool locate(VNode::index_type &i, const Point &point) const;
  virtual bool locate(VElem::index_type &i, const Point &point) const;

  virtual bool get_coords(vector<double> &coords, const Point &point, 
                                                    VElem::index_type i) const;  
  virtual void interpolate(Point &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;
  virtual void derivate(vector<Point> &p, const vector<double> &coords, 
                                                    VElem::index_type i) const;

  virtual void get_random_point(Point &p, VElem::index_type i,MusilRNG &rng) const;

  virtual bool get_neighbor(VElem::index_type &neighbor, 
                       VElem::index_type from, VDElem::index_type delem) const;
  virtual void get_neighbors(VElem::array_type &elems, 
                                                    VElem::index_type i) const;
  virtual void get_neighbors(VNode::array_type &nodes, 
                                                    VNode::index_type i) const;

  virtual void pwl_approx_edge(vector<vector<double> > &coords, 
                               VElem::index_type ci, unsigned int which_edge, 
                               unsigned int div_per_unit) const;
  virtual void pwl_approx_face(vector<vector<vector<double> > > &coords, 
                               VElem::index_type ci, unsigned int which_face, 
                              unsigned int div_per_unit) const;
                              
  virtual void get_dimensions(Mesh::dimension_type& dims);
};


template <class Basis>
const TypeDescription* get_type_description(LatVolMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("LatVolMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((LatVolMesh<Basis> *)0);
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::node_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::NodeIndex",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
const TypeDescription*
LatVolMesh<Basis>::cell_index_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((LatVolMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::CellIndex",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template <class Basis>
PersistentTypeID
LatVolMesh<Basis>::type_idlv(type_name(-1), "Mesh", LatVolMesh<Basis>::maker);

template <class Basis>
MeshTypeID
LatVolMesh<Basis>::mesh_idlv(type_name(-1), LatVolMesh<Basis>::mesh_maker, LatVolMesh<Basis>::latvol_maker);


template <class Basis>
LatVolMesh<Basis>::LatVolMesh(unsigned int i, unsigned int j, unsigned int k,
                              const Point &min, const Point &max)
  : min_i_(0), min_j_(0), min_k_(0),
    ni_(i), nj_(j), nk_(k)
{
  transform_.pre_scale(Vector(1.0 / (i-1.0), 1.0 / (j-1.0), 1.0 / (k-1.0)));
  transform_.pre_scale(max - min);

  transform_.pre_translate(min.asVector());
  transform_.compute_imat();
}


template <class Basis>
void
LatVolMesh<Basis>::get_random_point(Point &p,
                                    const typename Elem::index_type &ei,
                                    MusilRNG &rng) const
{
  // build the three principal edge vectors
  typename Node::array_type ra;
  get_nodes(ra,ei);
  Point p0,p1,p2,p3;
  get_point(p0,ra[0]);
  get_point(p1,ra[1]);
  get_point(p2,ra[3]);
  get_point(p3,ra[4]);
  Vector v0(p1-p0);
  Vector v1(p2-p0);
  Vector v2(p3-p0);

  // Choose a random point in the cell.
  const double t = rng();
  const double u = rng();
  const double v = rng();

  p = p0+(v0*t)+(v1*u)+(v2*v);
}


template <class Basis>
BBox
LatVolMesh<Basis>::get_bounding_box() const
{
  Point p0(min_i_,         min_j_,         min_k_);
  Point p1(min_i_ + ni_-1, min_j_,         min_k_);
  Point p2(min_i_ + ni_-1, min_j_ + nj_-1, min_k_);
  Point p3(min_i_,         min_j_ + nj_-1, min_k_);
  Point p4(min_i_,         min_j_,         min_k_ + nk_-1);
  Point p5(min_i_ + ni_-1, min_j_,         min_k_ + nk_-1);
  Point p6(min_i_ + ni_-1, min_j_ + nj_-1, min_k_ + nk_-1);
  Point p7(min_i_,         min_j_ + nj_-1, min_k_ + nk_-1);

  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  result.extend(transform_.project(p2));
  result.extend(transform_.project(p3));
  result.extend(transform_.project(p4));
  result.extend(transform_.project(p5));
  result.extend(transform_.project(p6));
  result.extend(transform_.project(p7));
  return result;
}


template <class Basis>
Vector
LatVolMesh<Basis>::diagonal() const
{
  return get_bounding_box().diagonal();
}


template <class Basis>
void
LatVolMesh<Basis>::transform(const Transform &t)
{
  transform_.pre_trans(t);
}


template <class Basis>
void
LatVolMesh<Basis>::get_canonical_transform(Transform &t)
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, nj_ - 1.0, nk_ - 1.0));
}


template <class Basis>
bool
LatVolMesh<Basis>::get_min(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(min_i_);
  array.push_back(min_j_);
  array.push_back(min_k_);

  return true;
}


template <class Basis>
bool
LatVolMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(3);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);
  array.push_back(nk_);

  return true;
}


template <class Basis>
void
LatVolMesh<Basis>::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
  min_j_ = min[1];
  min_k_ = min[2];
}


template <class Basis>
void
LatVolMesh<Basis>::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
  nj_ = dim[1];
  nk_ = dim[2];
}


// Note: This code does not respect boundaries of the mesh
template <class Basis>
void
LatVolMesh<Basis>::get_nodes(typename Node::array_type &array,
                             typename Edge::index_type idx) const
{
  array.resize(2);
  // The (const unsigned int) on the next line is due
  // to a bug in the OSX 10.4 compiler that gives xidx
  // the wrong type
  const unsigned int xidx = (const unsigned int)idx;
  if (xidx < (ni_ - 1) * nj_ * nk_)
  {
    const int i = xidx % (ni_ - 1);
    const int jk = xidx / (ni_ - 1);
    const int j = jk % nj_;
    const int k = jk / nj_;

    array[0] = typename Node::index_type(this, i+0, j, k);
    array[1] = typename Node::index_type(this, i+1, j, k);
  }
  else
  {
    const unsigned int yidx = idx - (ni_ - 1) * nj_ * nk_;
    if (yidx < (ni_ * (nj_ - 1) * nk_))
    {
      const int j = yidx % (nj_ - 1);
      const int ik = yidx / (nj_ - 1);
      const int i = ik / nk_;
      const int k = ik % nk_;

      array[0] = typename Node::index_type(this, i, j+0, k);
      array[1] = typename Node::index_type(this, i, j+1, k);
    }
    else
    {
      const unsigned int zidx = yidx - (ni_ * (nj_ - 1) * nk_);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % ni_;
      const int j = ij / ni_;

      array[0] = typename Node::index_type(this, i, j, k+0);
      array[1] = typename Node::index_type(this, i, j, k+1);
    }
  }
}


// Note: This code does not respect boundaries of the mesh
template <class Basis>
void
LatVolMesh<Basis>::get_nodes(typename Node::array_type &array,
                             typename Face::index_type idx) const
{
  array.resize(4);
  // The (const unsigned int) on the next line is due
  // to a bug in the OSX 10.4 compiler that gives xidx
  // the wrong type
  const unsigned int xidx = (const unsigned int)idx;
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    const int i = xidx % (ni_ - 1);
    const int jk = xidx / (ni_ - 1);
    const int j = jk % (nj_ - 1);
    const int k = jk / (nj_ - 1);
    array[0] = typename Node::index_type(this, i+0, j+0, k);
    array[1] = typename Node::index_type(this, i+1, j+0, k);
    array[2] = typename Node::index_type(this, i+1, j+1, k);
    array[3] = typename Node::index_type(this, i+0, j+1, k);
  }
  else
  {
    const unsigned int yidx = idx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      const int j = yidx % (nj_ - 1);
      const int ik = yidx / (nj_ - 1);
      const int k = ik % (nk_ - 1);
      const int i = ik / (nk_ - 1);
      array[0] = typename Node::index_type(this, i, j+0, k+0);
      array[1] = typename Node::index_type(this, i, j+1, k+0);
      array[2] = typename Node::index_type(this, i, j+1, k+1);
      array[3] = typename Node::index_type(this, i, j+0, k+1);
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % (ni_ - 1);
      const int j = ij / (ni_ - 1);
      array[0] = typename Node::index_type(this, i+0, j, k+0);
      array[1] = typename Node::index_type(this, i+0, j, k+1);
      array[2] = typename Node::index_type(this, i+1, j, k+1);
      array[3] = typename Node::index_type(this, i+1, j, k+0);
    }
  }
}


// Note: This code does not respect boundaries of the mesh.
template <class Basis>
void
LatVolMesh<Basis>::get_nodes(typename Node::array_type &array,
                             const typename Cell::index_type &idx) const
{
  array.resize(8);
  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;   array[0].k_ = idx.k_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;   array[1].k_ = idx.k_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1; array[2].k_ = idx.k_;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1; array[3].k_ = idx.k_;
  array[4].i_ = idx.i_;   array[4].j_ = idx.j_;   array[4].k_ = idx.k_+1;
  array[5].i_ = idx.i_+1; array[5].j_ = idx.j_;   array[5].k_ = idx.k_+1;
  array[6].i_ = idx.i_+1; array[6].j_ = idx.j_+1; array[6].k_ = idx.k_+1;
  array[7].i_ = idx.i_;   array[7].j_ = idx.j_+1; array[7].k_ = idx.k_+1;

  array[0].mesh_ = this;
  array[1].mesh_ = this;
  array[2].mesh_ = this;
  array[3].mesh_ = this;
  array[4].mesh_ = this;
  array[5].mesh_ = this;
  array[6].mesh_ = this;
  array[7].mesh_ = this;
}


template <class Basis>
void
LatVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                             typename Face::index_type idx) const
{
  array.resize(4);
  const unsigned int num_i_faces = (ni_-1)*(nj_-1)*nk_;  // lie in ij plane ijk
  const unsigned int num_j_faces = ni_*(nj_-1)*(nk_-1);  // lie in jk plane jki
  const unsigned int num_k_faces = (ni_-1)*nj_*(nk_-1);  // lie in ki plane kij

  const unsigned int num_i_edges = (ni_-1)*nj_*nk_; // ijk
  const unsigned int num_j_edges = ni_*(nj_-1)*nk_; // jki

  unsigned int facei, facej, facek;
  unsigned int face = idx;

  if (face < num_i_faces)
  {
    facei = face % (ni_-1);
    facej = (face / (ni_-1)) % (nj_-1);
    facek = face / ((ni_-1)*(nj_-1));
    array[0] = facei+facej*(ni_-1)+facek*(ni_-1)*(nj_);
    array[1] = facei+(facej+1)*(ni_-1)+facek*(ni_-1)*(nj_);
    array[2] = num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1);
    array[3] = num_i_edges + (facei+1)*(nj_-1)*(nk_)+facej+facek*(nj_-1);
  }
  else if (face - num_i_faces < num_j_faces)
  {
    face -= num_i_faces;
    facei = face / ((nj_-1) *(nk_-1));
    facej = face % (nj_-1);
    facek = (face / (nj_-1)) % (nk_-1);
    array[0] = num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1);
    array[1] = num_i_edges + facei*(nj_-1)*(nk_)+facej+(facek+1)*(nj_-1);
    array[2] = (num_i_edges + num_j_edges +
                facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
    array[3] = (num_i_edges + num_j_edges +
                facei*(nk_-1)+(facej+1)*(ni_)*(nk_-1)+facek);

  }
  else if (face - num_i_faces - num_j_faces < num_k_faces)
  {
    face -= (num_i_faces + num_j_faces);
    facei = (face / (nk_-1)) % (ni_-1);
    facej = face / ((ni_-1) * (nk_-1));
    facek = face % (nk_-1);
    array[0] = facei+facej*(ni_-1)+facek*(ni_-1)*(nj_);
    array[1] = facei+facej*(ni_-1)+(facek+1)*(ni_-1)*(nj_);
    array[2] = (num_i_edges + num_j_edges +
                facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
    array[3] = (num_i_edges + num_j_edges +
                (facei+1)*(nk_-1)+facej*(ni_)*(nk_-1)+facek);
  }
  else {
    ASSERTFAIL(
          "LatVolMesh<Basis>::get_edges(Edge, Face) Face idx out of bounds");
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_edges(typename Edge::array_type &array,
                             const typename Cell::index_type &idx) const
{
  array.resize(12);
  const unsigned int j_start= (ni_-1)*nj_*nk_;
  const unsigned int k_start = ni_*(nj_-1)*nk_ + j_start;

  array[0] = idx.i_ + idx.j_*(ni_-1)     + idx.k_*(ni_-1)*(nj_);
  array[1] = idx.i_ + (idx.j_+1)*(ni_-1) + idx.k_*(ni_-1)*(nj_);
  array[2] = idx.i_ + idx.j_*(ni_-1)     + (idx.k_+1)*(ni_-1)*(nj_);
  array[3] = idx.i_ + (idx.j_+1)*(ni_-1) + (idx.k_+1)*(ni_-1)*(nj_);

  array[4] = j_start + idx.i_*(nj_-1)*(nk_)     + idx.j_ + idx.k_*(nj_-1);
  array[5] = j_start + (idx.i_+1)*(nj_-1)*(nk_) + idx.j_ + idx.k_*(nj_-1);
  array[6] = j_start + idx.i_*(nj_-1)*(nk_)     + idx.j_ + (idx.k_+1)*(nj_-1);
  array[7] = j_start + (idx.i_+1)*(nj_-1)*(nk_) + idx.j_ + (idx.k_+1)*(nj_-1);

  array[8] =  k_start + idx.i_*(nk_-1)     + idx.j_*(ni_)*(nk_-1)     + idx.k_;
  array[9] =  k_start + (idx.i_+1)*(nk_-1) + idx.j_*(ni_)*(nk_-1)     + idx.k_;
  array[10] = k_start + idx.i_*(nk_-1)     + (idx.j_+1)*(ni_)*(nk_-1) + idx.k_;
  array[11] = k_start + (idx.i_+1)*(nk_-1) + (idx.j_+1)*(ni_)*(nk_-1) + idx.k_;

}


template <class Basis>
void
LatVolMesh<Basis>::get_elems(typename Cell::array_type &result,
                             const typename Edge::index_type &eidx) const
{
  result.reserve(4);
  result.clear();
  const unsigned int offset1 = (ni_-1)*nj_*nk_;
  const unsigned int offset2 = offset1 + ni_*(nj_-1)*nk_;
  unsigned int idx = eidx;

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_)*(ni_-1)); idx -= k*(nj_)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (j > 0)
    {
      if (k < (nk_-1)) result.push_back(CellIndex(this,i,j-1,k));
      if (k > 0) result.push_back(CellIndex(this,i,j-1,k-1));
    }
    
    if (j < (nj_-1))
    {
      if (k < (nk_-1)) result.push_back(CellIndex(this,i,j,k));
      if (k > 0) result.push_back(CellIndex(this,i,j,k-1));
    }
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_)); idx -= j*(nk_-1)*(ni_);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (i > 0)
    {
      if (j < (nj_-1)) result.push_back(CellIndex(this,i-1,j,k));    
      if (j > 0) result.push_back(CellIndex(this,i-1,j-1,k));
    }
    
    if (i < (ni_-1))
    {
      if (j < (nj_-1)) result.push_back(CellIndex(this,i,j,k));    
      if (j > 0) result.push_back(CellIndex(this,i,j-1,k));
    }
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_)*(nj_-1)); idx -= i*(nk_)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (k > 0)
    {
      if (i < (nk_-1)) result.push_back(CellIndex(this,i,j,k-1));
      if (i > 0) result.push_back(CellIndex(this,i-1,j,k-1));
    }

    if (k < (nk_-1))
    {
      if (i < (ni_-1)) result.push_back(CellIndex(this,i,j,k));        
      if (i > 0) result.push_back(CellIndex(this,i-1,j,k));
    }
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_faces(typename Face::array_type &array,
                             const typename Cell::index_type &idx) const
{
  array.resize(6);

  const unsigned int i = idx.i_;
  const unsigned int j = idx.j_;
  const unsigned int k = idx.k_;

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = i + (j + k * (nj_-1)) * (ni_-1);
  array[1] = i + (j + (k+1) * (nj_-1)) * (ni_-1);

  array[2] = offset1 + j + (k + i * (nk_-1)) * (nj_-1);
  array[3] = offset1 + j + (k + (i+1) * (nk_-1)) * (nj_-1);

  array[4] = offset2 + k + (i + j * (ni_-1)) * (nk_-1);
  array[5] = offset2 + k + (i + (j+1) * (ni_-1)) * (nk_-1);
}


template <class Basis>
void
LatVolMesh<Basis>::get_elems(typename Cell::array_type &result,
                             const typename Face::index_type &fidx) const
{
  result.reserve(2);
  result.clear();
  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);
  unsigned int idx = fidx;

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_-1)*(ni_-1)); idx -= k*(nj_-1)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (k < (nk_-1)) result.push_back(CellIndex(this,i,j,k));
    if (k > 0) result.push_back(CellIndex(this,i,j,k-1));
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_-1)); idx -= j*(nk_-1)*(ni_-1);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (j < (nj_-1)) result.push_back(CellIndex(this,i,j,k));    
    if (j > 0) result.push_back(CellIndex(this,i,j-1,k));
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_-1)*(nj_-1)); idx -= i*(nk_-1)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (i < (ni_-1)) result.push_back(CellIndex(this,i,j,k));        
    if (i > 0) result.push_back(CellIndex(this,i-1,j,k));
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_elems(typename Cell::array_type &result,
                             const typename Node::index_type &idx) const
{
  result.reserve(8);
  result.clear();
  const unsigned int i0 = idx.i_ ? idx.i_ - 1 : 0;
  const unsigned int j0 = idx.j_ ? idx.j_ - 1 : 0;
  const unsigned int k0 = idx.k_ ? idx.k_ - 1 : 0;

  const unsigned int i1 = idx.i_ < ni_-1 ? idx.i_+1 : ni_-1;
  const unsigned int j1 = idx.j_ < nj_-1 ? idx.j_+1 : nj_-1;
  const unsigned int k1 = idx.k_ < nk_-1 ? idx.k_+1 : nk_-1;

  unsigned int i, j, k;
  for (k = k0; k < k1; k++)
    for (j = j0; j < j1; j++)
      for (i = i0; i < i1; i++)
        result.push_back(typename Cell::index_type(this, i, j, k));
}


//! return all cell_indecies that overlap the BBox in arr.

template <class Basis>
void
LatVolMesh<Basis>::get_cells(typename Cell::array_type &arr, const BBox &bbox)
{
  // Get our min and max
  typename Cell::index_type min, max;
  get_cells(min, max, bbox);

  // Clear the input array.  Limited to range of ints.
  arr.clear();

  // Loop over over min to max and fill the array
  unsigned int i, j, k;
  for (i = min.i_; i <= max.i_; i++) {
    for (j = min.j_; j <= max.j_; j++) {
      for (k = min.k_; k <= max.k_; k++) {
        arr.push_back(typename Cell::index_type(this, i,j,k));
      }
    }
  }
}


//! Returns the min and max indices that fall within or on the BBox.

// If the max index lies "in front of" (meaning that any of the
// indexes are negative) then the max will be set to [0,0,0] and the
// min to [1,1,1] in the hopes that they will be used something like
// this: for(unsigned int i = min.i_; i <= max.i_; i++)....  Otherwise
// you can expect the min and max to be set clamped to the boundaries.
template <class Basis>
void
LatVolMesh<Basis>::get_cells(typename Cell::index_type &begin, typename Cell::index_type &end,
                             const BBox &bbox) {

  const Point minp = transform_.unproject(bbox.min());
  int mini = (int)floor(minp.x());
  int minj = (int)floor(minp.y());
  int mink = (int)floor(minp.z());
  if (mini < 0) { mini = 0; }
  if (minj < 0) { minj = 0; }
  if (mink < 0) { mink = 0; }

  const Point maxp = transform_.unproject(bbox.max());
  int maxi = (int)floor(maxp.x());
  int maxj = (int)floor(maxp.y());
  int maxk = (int)floor(maxp.z());
  if (maxi >= (int)(ni_ - 1)) { maxi = ni_ - 2; }
  if (maxj >= (int)(nj_ - 1)) { maxj = nj_ - 2; }
  if (maxk >= (int)(nk_ - 1)) { maxk = nk_ - 2; }
  // We also need to protect against when any of these guys are
  // negative.  In this case we should not have any iteration.  We
  // can't however express negative numbers with unsigned ints (in the
  // case of index_type).
  if (maxi < 0 || maxj < 0 || maxk < 0) {
    // We should create a range which will not be iterated over
    mini = minj = mink = 1;
    maxi = maxj = maxk = 0;
  }

  begin = typename Cell::index_type(this, mini, minj, mink);
  end   = typename Cell::index_type(this, maxi, maxj, maxk);
}


template <class Basis>
bool
LatVolMesh<Basis>::get_neighbor(typename Cell::index_type &neighbor,
                                const typename Cell::index_type &from,
                                typename Face::index_type face) const
{
  // The (const unsigned int) on the next line is due
  // to a bug in the OSX 10.4 compiler that gives xidx
  // the wrong type
  const unsigned int xidx = (const unsigned int)face;
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    const unsigned int jk = xidx / (ni_ - 1);
    const unsigned int k = jk / (nj_ - 1);

    if (k == from.k_ && k > 0)
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k-1;
      neighbor.mesh_ = this;
      return true;
    }
    else if (k == (from.k_+1) && k < (nk_-1))
    {
      neighbor.i_ = from.i_;
      neighbor.j_ = from.j_;
      neighbor.k_ = k;
      neighbor.mesh_ = this;
      return true;
    }
  }
  else
  {
    const unsigned int yidx = xidx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      const unsigned int ik = yidx / (nj_ - 1);
      const unsigned int i = ik / (nk_ - 1);

      if (i == from.i_ && i > 0)
      {
        neighbor.i_ = i-1;
        neighbor.j_ = from.j_;
        neighbor.k_ = from.k_;
        neighbor.mesh_ = this;
        return true;
      }
      else if (i == (from.i_+1) && i < (ni_-1))
      {
        neighbor.i_ = i;
        neighbor.j_ = from.j_;
        neighbor.k_ = from.k_;
        neighbor.mesh_ = this;
        return true;
      }
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      const unsigned int ij = zidx / (nk_ - 1);
      const unsigned int j = ij / (ni_ - 1);

      if (j == from.j_ && j > 0)
      {
        neighbor.i_ = from.i_;
        neighbor.j_ = j-1;
        neighbor.k_ = from.k_;
        neighbor.mesh_ = this;
        return true;
      }
      else if (j == (from.j_+1) && j < (nj_-1))
      {
        neighbor.i_ = from.i_;
        neighbor.j_ = j;
        neighbor.k_ = from.k_;
        neighbor.mesh_ = this;
        return true;
      }
    }
  }
  return false;
}


template <class Basis>
void
LatVolMesh<Basis>::get_nodes(typename Node::index_type &begin,
                             typename Node::index_type &end,
                             const BBox &bbox)
{
  // get the min and max points of the bbox and make sure that they lie
  // inside the mesh boundaries.
  BBox mesh_boundary = get_bounding_box();
  // crop by min boundary
  Point min = Max(bbox.min(), mesh_boundary.min());
  Point max = Max(bbox.max(), mesh_boundary.min());
  // crop by max boundary
  min = Min(min, mesh_boundary.max());
  max = Min(max, mesh_boundary.max());
  typename Node::index_type min_index, max_index;

  // If one of the locates return true, then we have a valid iteration
  bool min_located = locate(min_index, min);
  bool max_located = locate(max_index, max);
  if (!min_located && !max_located)
  {
    // first check to see if there is a bbox overlap
    BBox box;
    box.extend(min);
    box.extend(max);
    if ( box.overlaps(mesh_boundary) )
    {
      Point r = transform_.unproject(min);
      double rx = floor(r.x());
      double ry = floor(r.y());
      double rz = floor(r.z());
      min_index.i_ = (unsigned int)Max(rx, 0.0);
      min_index.j_ = (unsigned int)Max(ry, 0.0);
      min_index.k_ = (unsigned int)Max(rz, 0.0);
      r = transform_.unproject(max);
      rx = floor(r.x());
      ry = floor(r.y());
      rz = floor(r.z());
      max_index.i_ = (unsigned int)Min(rx, (double)ni_ );
      max_index.j_ = (unsigned int)Min(ry, (double)nj_ );
      max_index.k_ = (unsigned int)Min(rz, (double)nk_ );
    }
    else
    {
      // Set the min and max extents of the range iterator to be the
      // same thing.  When they are the same end_iter will be set to
      // the starting state of the range iterator, thereby causing any
      // for loop using these iterators [for(;iter != end_iter;
      // iter++)] to never enter.
      min_index = typename Node::index_type(this, 0,0,0);
      max_index = typename Node::index_type(this, 0,0,0);
    }
  }
  else if ( !min_located )
  {
    const Point r = transform_.unproject(min);
    const double rx = floor(r.x());
    const double ry = floor(r.y());
    const double rz = floor(r.z());
    min_index.i_ = (unsigned int)Max(rx, 0.0);
    min_index.j_ = (unsigned int)Max(ry, 0.0);
    min_index.k_ = (unsigned int)Max(rz, 0.0);
  }
  else
  { //  !max_located
    const Point r = transform_.unproject(max);
    const double rx = floor(r.x());
    const double ry = floor(r.y());
    const double rz = floor(r.z());
    max_index.i_ = (unsigned int)Min(rx, (double) ni_ );
    max_index.j_ = (unsigned int)Min(ry, (double) nj_ );
    max_index.k_ = (unsigned int)Min(rz, (double) nk_ );
  }

  begin = min_index;
  end   = max_index;
}


template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &result,
                              typename Edge::index_type idx) const
{
  typename Node::array_type arr;
  get_nodes(arr, idx);
  Point p1;
  get_center(result, arr[0]);
  get_center(p1, arr[1]);

  result.asVector() += p1.asVector();
  result.asVector() *= 0.5;
}


template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &result,
                              typename Face::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  ASSERT(nodes.size() == 4);
  typename Node::array_type::iterator nai = nodes.begin();
  get_point(result, *nai);
  ++nai;
  Point pp;
  while (nai != nodes.end())
  {
    get_point(pp, *nai);
    result.asVector() += pp.asVector();
    ++nai;
  }
  result.asVector() *= (1.0 / 4.0);
}


template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &result,
                              const typename Cell::index_type &idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, idx.k_ + 0.5);
  result = transform_.project(p);
}


template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &result,
                              const typename Node::index_type &idx) const
{
  Point p(idx.i_, idx.j_, idx.k_);
  result = transform_.project(p);
}


template <class Basis>
double
LatVolMesh<Basis>::get_size(const typename Node::index_type &idx) const
{
  return 0.0;
}


template <class Basis>
double
LatVolMesh<Basis>::get_size(typename Edge::index_type idx) const
{
  typename Node::array_type arr;
  get_nodes(arr, idx);
  Point p0, p1;
  get_center(p0, arr[0]);
  get_center(p1, arr[1]);

  return (p1.asVector() - p0.asVector()).length();
}


template <class Basis>
double
LatVolMesh<Basis>::get_size(typename Face::index_type idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[2]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  return (v0.length() * v1.length());
}


template <class Basis>
double
LatVolMesh<Basis>::get_size(const typename Cell::index_type &idx) const
{
  typename Node::array_type nodes;
  get_nodes(nodes, idx);
  Point p0, p1, p2, p3;
  get_point(p0, nodes[0]);
  get_point(p1, nodes[1]);
  get_point(p2, nodes[3]);
  get_point(p3, nodes[4]);
  Vector v0 = p1 - p0;
  Vector v1 = p2 - p0;
  Vector v2 = p3 - p0;
  return (v0.length() * v1.length() * v2.length());
}


template <class Basis>
bool
LatVolMesh<Basis>::locate(typename Cell::index_type &cell, const Point &p) const
{
  if (basis_.polynomial_order() > 1) return elem_locate(cell, *this, p);
  const Point r = transform_.unproject(p);

  double ii = r.x();
  double jj = r.y();
  double kk = r.z();

  if (ii>(ni_-1) && (ii-(MIN_ELEMENT_VAL))<(ni_-1)) ii=ni_-1-(MIN_ELEMENT_VAL);
  if (jj>(nj_-1) && (jj-(MIN_ELEMENT_VAL))<(nj_-1)) jj=nj_-1-(MIN_ELEMENT_VAL);
  if (kk>(nk_-1) && (kk-(MIN_ELEMENT_VAL))<(nk_-1)) kk=nk_-1-(MIN_ELEMENT_VAL);
  if (ii<0 && ii>(-MIN_ELEMENT_VAL)) ii=0;
  if (jj<0 && jj>(-MIN_ELEMENT_VAL)) jj=0;
  if (kk<0 && kk>(-MIN_ELEMENT_VAL)) kk=0;
	
  const int i = static_cast<int>(floor(ii));
  const int j = static_cast<int>(floor(jj));
  const int k = static_cast<int>(floor(kk));

  if (i < (int)(ni_-1) && i >= 0 &&
      j < (int)(nj_-1) && j >= 0 &&
      k < (int)(nk_-1) && k >= 0)
  {
    cell.i_ = i;
    cell.j_ = j;
    cell.k_ = k;
    cell.mesh_ = this;
    return true;
  }
  cell.i_ = (unsigned int)Max(Min(ii,(double)(ni_-1)), 0.0);
  cell.j_ = (unsigned int)Max(Min(jj,(double)(nj_-1)), 0.0);
  cell.k_ = (unsigned int)Max(Min(kk,(double)(nk_-1)), 0.0);
  cell.mesh_ = this;
  return false;
}


template <class Basis>
bool
LatVolMesh<Basis>::locate(typename Node::index_type &node, const Point &p) const
{
  const Point r = transform_.unproject(p);

  const double rx = floor(r.x() + 0.5);
  const double ry = floor(r.y() + 0.5);
  const double rz = floor(r.z() + 0.5);


  // Clamp in double space to avoid overflow errors.
  if (rx < 0.0  || ry < 0.0  || rz < 0.0 ||
      rx >= ni_ || ry >= nj_ || rz >= nk_)
  {
    node.i_ = (unsigned int)Max(Min(rx,(double)(ni_-1)), 0.0);
    node.j_ = (unsigned int)Max(Min(ry,(double)(nj_-1)), 0.0);
    node.k_ = (unsigned int)Max(Min(rz,(double)(nk_-1)), 0.0);
    node.mesh_ = this;
    return false;
  }

  // Nodes over 2 billion might suffer roundoff error.
  node.i_ = (unsigned int)rx;
  node.j_ = (unsigned int)ry;
  node.k_ = (unsigned int)rz;
  node.mesh_ = this;

  return true;
}


template <class Basis>
int
LatVolMesh<Basis>::get_weights(const Point &p,
                               typename Node::array_type &locs,
                               double *w)
{
  typename Cell::index_type idx;
  if (locate(idx, p)) {
    get_nodes(locs, idx);
    vector<double> coords(3);
    if (get_coords(coords, p, idx)) {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
LatVolMesh<Basis>::get_weights(const Point &p, typename Cell::array_type &l,
                              double *w)
{
  typename Cell::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


template <class Basis>
const string
find_type_name(typename LatVolMesh<Basis>::NodeIndex *)
{
  static string name = LatVolMesh<Basis>::type_name(-1) + "::NodeIndex";
  return name;
}


template <class Basis>
const string
find_type_name(typename LatVolMesh<Basis>::CellIndex *)
{
  static string name = LatVolMesh<Basis>::type_name(-1) + "::CellIndex";
  return name;
}


#define LATVOLMESH_VERSION 4

template <class Basis>
void
LatVolMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), LATVOLMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);
  Pio(stream, nk_);

  if (version < 2 && stream.reading())
  {
    Point min, max;
    Pio(stream, min);
    Pio(stream, max);
    transform_.pre_scale(Vector(1.0 / (ni_ - 1.0),
                                1.0 / (nj_ - 1.0),
                                1.0 / (nk_ - 1.0)));
    transform_.pre_scale(max - min);
    transform_.pre_translate(Vector(min));
    transform_.compute_imat();
  } else if (version < 3 && stream.reading() ) {
    Pio_old(stream, transform_);
  }
  else
  {
    Pio(stream, transform_);
  }

  if (version >= 4) {
    basis_.io(stream);
  }

  stream.end_class();
}


template <class Basis>
const string
LatVolMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("LatVolMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template <class Basis>
void
LatVolMesh<Basis>::begin(typename LatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_, min_k_);
}


template <class Basis>
void
LatVolMesh<Basis>::end(typename LatVolMesh<Basis>::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_, min_k_ + nk_);
}


template <class Basis>
void
LatVolMesh<Basis>::size(typename LatVolMesh<Basis>::Node::size_type &s) const
{
  s = typename Node::size_type(ni_,nj_,nk_);
}


template <class Basis>
void
LatVolMesh<Basis>::to_index(typename LatVolMesh<Basis>::Node::index_type &idx,
                            unsigned int a) const
{
  const unsigned int i = a % ni_;
  const unsigned int jk = a / ni_;
  const unsigned int j = jk % nj_;
  const unsigned int k = jk / nj_;
  idx = typename Node::index_type(this, i, j, k);
}


template <class Basis>
void
LatVolMesh<Basis>::begin(typename LatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = typename Edge::iterator(0);
}


template <class Basis>
void
LatVolMesh<Basis>::end(typename LatVolMesh<Basis>::Edge::iterator &itr) const
{
  itr = ((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1));
}


template <class Basis>
void
LatVolMesh<Basis>::size(typename LatVolMesh<Basis>::Edge::size_type &s) const
{
  s = ((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1));
}


template <class Basis>
void
LatVolMesh<Basis>::begin(typename LatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = typename Face::iterator(0);
}


template <class Basis>
void
LatVolMesh<Basis>::end(typename LatVolMesh<Basis>::Face::iterator &itr) const
{
  itr = (ni_-1) * (nj_-1) * nk_ +
    ni_ * (nj_ - 1 ) * (nk_ - 1) +
    (ni_ - 1) * nj_ * (nk_ - 1);
}


template <class Basis>
void
LatVolMesh<Basis>::size(typename LatVolMesh<Basis>::Face::size_type &s) const
{
  s =  (ni_-1) * (nj_-1) * nk_ +
    ni_ * (nj_ - 1 ) * (nk_ - 1) +
    (ni_ - 1) * nj_ * (nk_ - 1);
}


template <class Basis>
void
LatVolMesh<Basis>::begin(typename LatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this,  min_i_, min_j_, min_k_);
}


template <class Basis>
void
LatVolMesh<Basis>::end(typename LatVolMesh<Basis>::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(this, min_i_, min_j_, min_k_ + nk_-1);
}


template <class Basis>
void
LatVolMesh<Basis>::size(typename LatVolMesh<Basis>::Cell::size_type &s) const
{
  s = typename Cell::size_type(ni_-1, nj_-1,nk_-1);
}


template <class Basis>
void
LatVolMesh<Basis>::to_index(typename LatVolMesh<Basis>::Cell::index_type &idx,
                            unsigned int a) const
{
  const unsigned int i = a % (ni_-1);
  const unsigned int jk = a / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  idx = typename Cell::index_type(this, i, j, k);
}


template <class Basis>
int
LatVolMesh<Basis>::get_valence(const typename Node::index_type &i) const
{
  return (((i.i_ == 0 || i.i_ == ni_) ? 1 : 2) +
          ((i.j_ == 0 || i.j_ == nj_) ? 1 : 2) +
          ((i.k_ == 0 || i.k_ == nk_) ? 1 : 2));
}


template <class Basis>
int
LatVolMesh<Basis>::get_valence(
                         typename LatVolMesh<Basis>::Edge::index_type i) const
{
  return 1;
}


template <class Basis>
int
LatVolMesh<Basis>::get_valence(
                         typename LatVolMesh<Basis>::Face::index_type i) const
{
  return 1;
}


template <class Basis>
int
LatVolMesh<Basis>::get_valence(const typename Cell::index_type &i) const
{
  return 1;
}


// VIRTUAL IMPLEMENTATION OF FUNCTIONS

template <class Basis>
bool
LatVolMesh<Basis>::has_virtual_interface() const
{
  return (true);
}


template <class Basis>
void
LatVolMesh<Basis>::size(VNode::size_type& sz) const
{
  sz = VNode::size_type(ni_*nj_*nk_);
}

template <class Basis>
void
LatVolMesh<Basis>::size(VEdge::size_type& sz) const
{
  sz = VEdge::size_type(((ni_-1) * nj_ * nk_) + (ni_ * (nj_-1) * nk_) + (ni_ * nj_ * (nk_-1)));
}

template <class Basis>
void
LatVolMesh<Basis>::size(VFace::size_type& sz) const
{
  sz = VFace::size_type((ni_-1) * (nj_-1) * nk_ + ni_ * (nj_ - 1 ) * (nk_ - 1) + (ni_ - 1) * nj_ * (nk_ - 1));
}

template <class Basis>
void
LatVolMesh<Basis>::size(VCell::size_type& sz) const
{
  sz = VCell::size_type((ni_-1)*(nj_-1)*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::size(VDElem::size_type& sz) const
{
  sz = VDElem::size_type((ni_-1) * (nj_-1) * nk_ + ni_ * (nj_ - 1 ) * (nk_ - 1) + (ni_ - 1) * nj_ * (nk_ - 1));
}

template <class Basis>
void
LatVolMesh<Basis>::size(VElem::size_type& sz) const
{
  sz = VElem::size_type((ni_-1)*(nj_-1)*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::get_nodes(VNode::array_type &array,
                             VEdge::index_type idx) const
{
  array.resize(2);
  const unsigned int xidx = static_cast<unsigned int>(idx);
  
  if (xidx < (ni_ - 1) * nj_ * nk_)
  {
    const int i = xidx % (ni_ - 1);
    const int jk = xidx / (ni_ - 1);
    const int j = jk % nj_;
    const int k = jk / nj_;
    const int a = i+j*ni_+(ni_*nj_)*k;
    array[0] = VNode::index_type(a);
    array[1] = VNode::index_type(a+1);
  }
  else
  {
    const unsigned int yidx = xidx - (ni_ - 1) * nj_ * nk_;
    if (yidx < (ni_ * (nj_ - 1) * nk_))
    {
      const int j = yidx % (nj_ - 1);
      const int ik = yidx / (nj_ - 1);
      const int i = ik / nk_;
      const int k = ik % nk_;

      const int a = i+j*ni_+(ni_*nj_)*k;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+ni_);
    }
    else
    {
      const unsigned int zidx = yidx - (ni_ * (nj_ - 1) * nk_);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % ni_;
      const int j = ij / ni_;

      const int a = i+j*ni_+(ni_*nj_)*k;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+(nj_*ni_));
    }
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_nodes(VNode::array_type &array,
                             VFace::index_type idx) const
{
  array.resize(4);

  const unsigned int xidx = static_cast<unsigned int>(idx);
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    const unsigned int i = xidx % (ni_ - 1);
    const unsigned int jk = xidx / (ni_ - 1);
    const unsigned int j = jk % (nj_ - 1);
    const unsigned int k = jk / (nj_ - 1);
    const unsigned int nij_ = ni_*nj_; 
    const unsigned int a = i+j*ni_+k*nij_;
    array[0] = VNode::index_type(a);
    array[1] = VNode::index_type(a+1);
    array[2] = VNode::index_type(a+ni_+1);
    array[3] = VNode::index_type(a+ni_);
  }
  else
  {
    const unsigned int yidx = xidx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      const unsigned int j = yidx % (nj_ - 1);
      const unsigned int ik = yidx / (nj_ - 1);
      const unsigned int k = ik % (nk_ - 1);
      const unsigned int i = ik / (nk_ - 1);
      
      const unsigned int nij_ = ni_*nj_; 
      const unsigned int a = i+j*ni_+k*nij_;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+ni_);
      array[2] = VNode::index_type(a+ni_+nij_);
      array[3] = VNode::index_type(a+nij_);    
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % (ni_ - 1);
      const int j = ij / (ni_ - 1);

      const unsigned int nij_ = ni_*nj_; 
      const unsigned int a = i+j*ni_+k*nij_;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+nij_);
      array[2] = VNode::index_type(a+nij_+1);
      array[3] = VNode::index_type(a+1); 
    }
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_nodes(VNode::array_type &array,
                             VCell::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);

  array.resize(8);
  const unsigned int nij_ = ni_*nj_;
  const unsigned int a = i+ni_*j+nij_*k;
  array[0] = VNode::index_type(a);
  array[1] = VNode::index_type(a+1);
  array[2] = VNode::index_type(a+1+ni_);
  array[3] = VNode::index_type(a+ni_);
  array[4] = VNode::index_type(a+nij_);
  array[5] = VNode::index_type(a+1+nij_);
  array[6] = VNode::index_type(a+1+ni_+nij_);
  array[7] = VNode::index_type(a+ni_+nij_);
}


template <class Basis>
void
LatVolMesh<Basis>::get_nodes(VNode::array_type &array,
                             VElem::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);

  array.resize(8);
  const unsigned int nij_ = ni_*nj_;
  const unsigned int a = i+ni_*j+nij_*k;
  array[0] = VNode::index_type(a);
  array[1] = VNode::index_type(a+1);
  array[2] = VNode::index_type(a+1+ni_);
  array[3] = VNode::index_type(a+ni_);
  array[4] = VNode::index_type(a+nij_);
  array[5] = VNode::index_type(a+1+nij_);
  array[6] = VNode::index_type(a+1+ni_+nij_);
  array[7] = VNode::index_type(a+ni_+nij_);
}

template <class Basis>
void
LatVolMesh<Basis>::get_nodes(VNode::array_type &array,
                             VDElem::index_type idx) const
{
  array.resize(4);

  const unsigned int xidx = static_cast<unsigned int>(idx);
  if (xidx < (ni_ - 1) * (nj_ - 1) * nk_)
  {
    const unsigned int i = xidx % (ni_ - 1);
    const unsigned int jk = xidx / (ni_ - 1);
    const unsigned int j = jk % (nj_ - 1);
    const unsigned int k = jk / (nj_ - 1);
    const unsigned int nij_ = ni_*nj_; 
    const unsigned int a = i+j*ni_+k*nij_;
    array[0] = VNode::index_type(a);
    array[1] = VNode::index_type(a+1);
    array[2] = VNode::index_type(a+ni_+1);
    array[3] = VNode::index_type(a+ni_);
  }
  else
  {
    const unsigned int yidx = xidx - (ni_ - 1) * (nj_ - 1) * nk_;
    if (yidx < ni_ * (nj_ - 1) * (nk_ - 1))
    {
      const unsigned int j = yidx % (nj_ - 1);
      const unsigned int ik = yidx / (nj_ - 1);
      const unsigned int k = ik % (nk_ - 1);
      const unsigned int i = ik / (nk_ - 1);
      
      const unsigned int nij_ = ni_*nj_; 
      const unsigned int a = i+j*ni_+k*nij_;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+ni_);
      array[2] = VNode::index_type(a+ni_+nij_);
      array[3] = VNode::index_type(a+nij_);    
    }
    else
    {
      const unsigned int zidx = yidx - ni_ * (nj_ - 1) * (nk_ - 1);
      const int k = zidx % (nk_ - 1);
      const int ij = zidx / (nk_ - 1);
      const int i = ij % (ni_ - 1);
      const int j = ij / (ni_ - 1);

      const unsigned int nij_ = ni_*nj_; 
      const unsigned int a = i+j*ni_+k*nij_;
      array[0] = VNode::index_type(a);
      array[1] = VNode::index_type(a+nij_);
      array[2] = VNode::index_type(a+nij_+1);
      array[3] = VNode::index_type(a+1); 
    }
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_edges(VEdge::array_type &array,
                             VFace::index_type idx) const
{
  array.resize(4);
  const unsigned int num_i_faces = (ni_-1)*(nj_-1)*nk_;  // lie in ij plane ijk
  const unsigned int num_j_faces = ni_*(nj_-1)*(nk_-1);  // lie in jk plane jki
  const unsigned int num_k_faces = (ni_-1)*nj_*(nk_-1);  // lie in ki plane kij

  const unsigned int num_i_edges = (ni_-1)*nj_*nk_; // ijk
  const unsigned int num_j_edges = ni_*(nj_-1)*nk_; // jki

  unsigned int facei, facej, facek;
  unsigned int face = static_cast<unsigned int>(idx);

  if (face < num_i_faces)
  {
    facei = face % (ni_-1);
    facej = (face / (ni_-1)) % (nj_-1);
    facek = face / ((ni_-1)*(nj_-1));
    array[0] = VEdge::index_type(facei+facej*(ni_-1)+facek*(ni_-1)*(nj_));
    array[1] = VEdge::index_type(facei+(facej+1)*(ni_-1)+facek*(ni_-1)*(nj_));
    array[2] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1));
    array[3] = VEdge::index_type(num_i_edges + (facei+1)*(nj_-1)*(nk_)+facej+facek*(nj_-1));
  }
  else if (face - num_i_faces < num_j_faces)
  {
    face -= num_i_faces;
    facei = face / ((nj_-1) *(nk_-1));
    facej = face % (nj_-1);
    facek = (face / (nj_-1)) % (nk_-1);
    array[0] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1));
    array[1] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+(facek+1)*(nj_-1));
    array[2] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
    array[3] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+(facej+1)*(ni_)*(nk_-1)+facek));
  }
  else if (face - num_i_faces - num_j_faces < num_k_faces)
  {
    face -= (num_i_faces + num_j_faces);
    facei = (face / (nk_-1)) % (ni_-1);
    facej = face / ((ni_-1) * (nk_-1));
    facek = face % (nk_-1);
    array[0] = VEdge::index_type(facei+facej*(ni_-1)+facek*(ni_-1)*(nj_));
    array[1] = VEdge::index_type(facei+facej*(ni_-1)+(facek+1)*(ni_-1)*(nj_));
    array[2] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
    array[3] = VEdge::index_type((num_i_edges + num_j_edges + (facei+1)*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
  }
  else 
  {
    ASSERTFAIL("LatVolMesh<Basis>::get_edges(Edge, Face) Face idx out of bounds");
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_edges(VEdge::array_type &array,
                             VDElem::index_type idx) const
{
  array.resize(4);
  const unsigned int num_i_faces = (ni_-1)*(nj_-1)*nk_;  // lie in ij plane ijk
  const unsigned int num_j_faces = ni_*(nj_-1)*(nk_-1);  // lie in jk plane jki
  const unsigned int num_k_faces = (ni_-1)*nj_*(nk_-1);  // lie in ki plane kij

  const unsigned int num_i_edges = (ni_-1)*nj_*nk_; // ijk
  const unsigned int num_j_edges = ni_*(nj_-1)*nk_; // jki

  unsigned int facei, facej, facek;
  unsigned int face = static_cast<unsigned int>(idx);

  if (face < num_i_faces)
  {
    facei = face % (ni_-1);
    facej = (face / (ni_-1)) % (nj_-1);
    facek = face / ((ni_-1)*(nj_-1));
    array[0] = VEdge::index_type(facei+facej*(ni_-1)+facek*(ni_-1)*(nj_));
    array[1] = VEdge::index_type(facei+(facej+1)*(ni_-1)+facek*(ni_-1)*(nj_));
    array[2] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1));
    array[3] = VEdge::index_type(num_i_edges + (facei+1)*(nj_-1)*(nk_)+facej+facek*(nj_-1));
  }
  else if (face - num_i_faces < num_j_faces)
  {
    face -= num_i_faces;
    facei = face / ((nj_-1) *(nk_-1));
    facej = face % (nj_-1);
    facek = (face / (nj_-1)) % (nk_-1);
    array[0] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+facek*(nj_-1));
    array[1] = VEdge::index_type(num_i_edges + facei*(nj_-1)*(nk_)+facej+(facek+1)*(nj_-1));
    array[2] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
    array[3] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+(facej+1)*(ni_)*(nk_-1)+facek));
  }
  else if (face - num_i_faces - num_j_faces < num_k_faces)
  {
    face -= (num_i_faces + num_j_faces);
    facei = (face / (nk_-1)) % (ni_-1);
    facej = face / ((ni_-1) * (nk_-1));
    facek = face % (nk_-1);
    array[0] = VEdge::index_type(facei+facej*(ni_-1)+facek*(ni_-1)*(nj_));
    array[1] = VEdge::index_type(facei+facej*(ni_-1)+(facek+1)*(ni_-1)*(nj_));
    array[2] = VEdge::index_type((num_i_edges + num_j_edges + facei*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
    array[3] = VEdge::index_type((num_i_edges + num_j_edges + (facei+1)*(nk_-1)+facej*(ni_)*(nk_-1)+facek));
  }
  else 
  {
    ASSERTFAIL("LatVolMesh<Basis>::get_edges(Edge, Face) Face idx out of bounds");
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_edges(VEdge::array_type &array,
                             VCell::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(12);
  const unsigned int j_start= (ni_-1)*nj_*nk_;
  const unsigned int k_start = ni_*(nj_-1)*nk_ + j_start;

  array[0] = VEdge::index_type(i+j*(ni_-1)+k*(ni_-1)*(nj_));
  array[1] = VEdge::index_type(i+(j+1)*(ni_-1)+k*(ni_-1)*(nj_));
  array[2] = VEdge::index_type(i+j*(ni_-1)+(k+1)*(ni_-1)*(nj_));
  array[3] = VEdge::index_type(i+(j+1)*(ni_-1)+(k+1)*(ni_-1)*(nj_));

  array[4] = VEdge::index_type(j_start+i*(nj_-1)*(nk_)+j+k*(nj_-1));
  array[5] = VEdge::index_type(j_start+(i+1)*(nj_-1)*(nk_)+j+k*(nj_-1));
  array[6] = VEdge::index_type(j_start+i*(nj_-1)*(nk_)+j+(k+1)*(nj_-1));
  array[7] = VEdge::index_type(j_start+(i+1)*(nj_-1)*(nk_)+j+(k+1)*(nj_-1));

  array[8] =  VEdge::index_type(k_start+i*(nk_-1)+j*(ni_)*(nk_-1)+k);
  array[9] =  VEdge::index_type(k_start+(i+1)*(nk_-1)+j*(ni_)*(nk_-1)+k);
  array[10] = VEdge::index_type(k_start+i*(nk_-1)+(j+1)*(ni_)*(nk_-1)+k);
  array[11] = VEdge::index_type(k_start+(i+1)*(nk_-1)+(j+1)*(ni_)*(nk_-1)+k);
}

template <class Basis>
void
LatVolMesh<Basis>::get_edges(VEdge::array_type &array,
                             VElem::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(12);
  const unsigned int j_start= (ni_-1)*nj_*nk_;
  const unsigned int k_start = ni_*(nj_-1)*nk_ + j_start;

  array[0] = VEdge::index_type(i+j*(ni_-1)+k*(ni_-1)*(nj_));
  array[1] = VEdge::index_type(i+(j+1)*(ni_-1)+k*(ni_-1)*(nj_));
  array[2] = VEdge::index_type(i+j*(ni_-1)+(k+1)*(ni_-1)*(nj_));
  array[3] = VEdge::index_type(i+(j+1)*(ni_-1)+(k+1)*(ni_-1)*(nj_));

  array[4] = VEdge::index_type(j_start+i*(nj_-1)*(nk_)+j+k*(nj_-1));
  array[5] = VEdge::index_type(j_start+(i+1)*(nj_-1)*(nk_)+j+k*(nj_-1));
  array[6] = VEdge::index_type(j_start+i*(nj_-1)*(nk_)+j+(k+1)*(nj_-1));
  array[7] = VEdge::index_type(j_start+(i+1)*(nj_-1)*(nk_)+j+(k+1)*(nj_-1));

  array[8] =  VEdge::index_type(k_start+i*(nk_-1)+j*(ni_)*(nk_-1)+k);
  array[9] =  VEdge::index_type(k_start+(i+1)*(nk_-1)+j*(ni_)*(nk_-1)+k);
  array[10] = VEdge::index_type(k_start+i*(nk_-1)+(j+1)*(ni_)*(nk_-1)+k);
  array[11] = VEdge::index_type(k_start+(i+1)*(nk_-1)+(j+1)*(ni_)*(nk_-1)+k);
}

template <class Basis>
void
LatVolMesh<Basis>::get_faces(VFace::array_type &array,
                             VCell::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(6);

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = VFace::index_type(i+(j+k*(nj_-1))*(ni_-1));
  array[1] = VFace::index_type(i+(j+(k+1)*(nj_-1))*(ni_-1));

  array[2] = VFace::index_type(offset1+j+(k+i*(nk_-1))*(nj_-1));
  array[3] = VFace::index_type(offset1+j+(k+(i+1)*(nk_-1))*(nj_-1));

  array[4] = VFace::index_type(offset2+k+(i+j*(ni_-1))*(nk_-1));
  array[5] = VFace::index_type(offset2+k+(i+(j+1)*(ni_-1))*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::get_faces(VFace::array_type &array,
                             VElem::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(6);

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = VFace::index_type(i+(j+k*(nj_-1))*(ni_-1));
  array[1] = VFace::index_type(i+(j+(k+1)*(nj_-1))*(ni_-1));

  array[2] = VFace::index_type(offset1+j+(k+i*(nk_-1))*(nj_-1));
  array[3] = VFace::index_type(offset1+j+(k+(i+1)*(nk_-1))*(nj_-1));

  array[4] = VFace::index_type(offset2+k+(i+j*(ni_-1))*(nk_-1));
  array[5] = VFace::index_type(offset2+k+(i+(j+1)*(ni_-1))*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::get_faces(VFace::array_type &array,
                             VDElem::index_type idx) const
{
  array.resize(1); array[0] = static_cast<VFace::index_type>(idx);
}

template <class Basis>
void
LatVolMesh<Basis>::get_cells(VCell::array_type &array,
                             VNode::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);

  const unsigned int ii = xidx % ni_;
  const unsigned int jk = xidx / ni_;
  const unsigned int jj = jk % nj_;
  const unsigned int kk = jk / nj_;
  
  array.reserve(8);
  array.clear();
  const unsigned int i0 = ii ? ii-1 : 0;
  const unsigned int j0 = jj ? jj- 1 : 0;
  const unsigned int k0 = kk ? kk- 1 : 0;

  const unsigned int i1 = ii < ni_-1 ? ii+1 : ni_-1;
  const unsigned int j1 = jj < nj_-1 ? jj+1 : nj_-1;
  const unsigned int k1 = kk < nk_-1 ? kk+1 : nk_-1;

  unsigned int i, j, k;
  for (k = k0; k < k1; k++)
    for (j = j0; j < j1; j++)
      for (i = i0; i < i1; i++)
      {
        array.push_back(VCell::index_type(i+j*(ni_-1)+k*(ni_-1)*(nj_-1)));
      }
}

template <class Basis>
void
LatVolMesh<Basis>::get_cells(VCell::array_type &array,
                             VEdge::index_type eidx) const
{
  array.reserve(4);
  array.clear();
  
  const unsigned int offset1 = (ni_-1)*nj_*nk_;
  const unsigned int offset2 = offset1 + ni_*(nj_-1)*nk_;
  unsigned int idx = eidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);
  
  if (idx < offset1)
  {
    unsigned int k = idx/((nj_)*(ni_-1)); idx -= k*(nj_)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (j > 0)
    {
      if (k < (nk_-1)) array.push_back(VCell::index_type(i+mj*(j-1)+mk*k));
      if (k > 0) array.push_back(VCell::index_type(i+mj*(j-1)+mk*(k-1)));
    }
    if (j < (nj_-1))
    {
      if (k < (nk_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));
      if (k > 0) array.push_back(VCell::index_type(i+mj*j+mk*(k-1)));
    }
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_)); idx -= j*(nk_-1)*(ni_);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (i > 0)
    {
      if (j < (nj_-1)) array.push_back(VCell::index_type((i-1)+mj*j+mk*k));    
      if (j > 0) array.push_back(VCell::index_type((i-1)+mj*(j-1)+mk*k));
    }
    if (i < (ni_-1))
    {
      if (j < (nj_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));    
      if (j > 0) array.push_back(VCell::index_type(i+mj*(j-1)+mk*k));
    }
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_)*(nj_-1)); idx -= i*(nk_)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (k > 0)
    {
      if (i < (nk_-1)) array.push_back(VCell::index_type(i+mj*j+mk*(k-1)));
      if (i > 0) array.push_back(VCell::index_type((i-1)+mj*j+mk*(k-1)));
    }

    if (k < (nk_-1))
    {
      if (i < (ni_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));        
      if (i > 0) array.push_back(VCell::index_type((i-1)+mj*j+mk*k));
    }
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_cells(VCell::array_type &array,
                             VFace::index_type fidx) const
{
  array.reserve(2);
  array.clear();
  
  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);
  unsigned int idx = fidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_-1)*(ni_-1)); idx -= k*(nj_-1)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (k < (nk_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));
    if (k > 0) array.push_back(VCell::index_type(i+mj*j+mk*(k-1)));
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_-1)); idx -= j*(nk_-1)*(ni_-1);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (j < (nj_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));    
    if (j > 0) array.push_back(VCell::index_type(i+mj*(j-1)+mk*k));
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_-1)*(nj_-1)); idx -= i*(nk_-1)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (i < (ni_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));        
    if (i > 0) array.push_back(VCell::index_type((i-1)+mj*j+mk*k));
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_cells(VCell::array_type &array,
                             VElem::index_type idx) const
{
  array.resize(1); array[0] = VCell::index_type(idx);
}

template <class Basis>
void
LatVolMesh<Basis>::get_cells(VCell::array_type &array,
                             VDElem::index_type fidx) const
{
  array.reserve(2);
  array.clear();
  
  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);
  unsigned int idx = fidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_-1)*(ni_-1)); idx -= k*(nj_-1)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (k < (nk_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));
    if (k > 0) array.push_back(VCell::index_type(i+mj*j+mk*(k-1)));
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_-1)); idx -= j*(nk_-1)*(ni_-1);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (j < (nj_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));    
    if (j > 0) array.push_back(VCell::index_type(i+mj*(j-1)+mk*k));
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_-1)*(nj_-1)); idx -= i*(nk_-1)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (i < (ni_-1)) array.push_back(VCell::index_type(i+mj*j+mk*k));        
    if (i > 0) array.push_back(VCell::index_type((i-1)+mj*j+mk*k));
  }
}


template <class Basis>
void
LatVolMesh<Basis>::get_elems(VElem::array_type &array,
                             VNode::index_type idx) const
{
  const unsigned int ii = idx % ni_;
  const unsigned int jk = idx / ni_;
  const unsigned int jj = jk % nj_;
  const unsigned int kk = jk / nj_;
  
  array.reserve(8);
  array.clear();
  const unsigned int i0 = ii ? ii-1 : 0;
  const unsigned int j0 = jj ? jj- 1 : 0;
  const unsigned int k0 = kk ? kk- 1 : 0;

  const unsigned int i1 = ii < ni_-1 ? ii+1 : ni_-1;
  const unsigned int j1 = jj < nj_-1 ? jj+1 : nj_-1;
  const unsigned int k1 = kk < nk_-1 ? kk+1 : nk_-1;

  unsigned int i, j, k;
  for (k = k0; k < k1; k++)
    for (j = j0; j < j1; j++)
      for (i = i0; i < i1; i++)
      {
        array.push_back(VElem::index_type(i+j*(ni_-1)+k*(ni_-1)*(nj_-1)));
      }
}

template <class Basis>
void
LatVolMesh<Basis>::get_elems(VElem::array_type &array,
                             VEdge::index_type eidx) const
{
  array.reserve(4);
  array.clear();
  
  const unsigned int offset1 = (ni_-1)*nj_*nk_;
  const unsigned int offset2 = offset1 + ni_*(nj_-1)*nk_;
  unsigned int idx = eidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);
  
  if (idx < offset1)
  {
    unsigned int k = idx/((nj_)*(ni_-1)); idx -= k*(nj_)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (j > 0)
    {
      if (k < (nk_-1)) array.push_back(VElem::index_type(i+mj*(j-1)+mk*k));
      if (k > 0) array.push_back(VElem::index_type(i+mj*(j-1)+mk*(k-1)));
    }
    if (j < (nj_-1))
    {
      if (k < (nk_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));
      if (k > 0) array.push_back(VElem::index_type(i+mj*j+mk*(k-1)));
    }
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_)); idx -= j*(nk_-1)*(ni_);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (i > 0)
    {
      if (j < (nj_-1)) array.push_back(VElem::index_type((i-1)+mj*j+mk*k));    
      if (j > 0) array.push_back(VElem::index_type((i-1)+mj*(j-1)+mk*k));
    }
    if (i < (ni_-1))
    {
      if (j < (nj_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));    
      if (j > 0) array.push_back(VElem::index_type(i+mj*(j-1)+mk*k));
    }
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_)*(nj_-1)); idx -= i*(nk_)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (k > 0)
    {
      if (i < (nk_-1)) array.push_back(VElem::index_type(i+mj*j+mk*(k-1)));
      if (i > 0) array.push_back(VElem::index_type((i-1)+mj*j+mk*(k-1)));
    }

    if (k < (nk_-1))
    {
      if (i < (ni_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));        
      if (i > 0) array.push_back(VElem::index_type((i-1)+mj*j+mk*k));
    }
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_elems(VElem::array_type &array,
                             VFace::index_type fidx) const
{
  array.reserve(2);
  array.clear();
  
  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);
  unsigned int idx = fidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_-1)*(ni_-1)); idx -= k*(nj_-1)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (k < (nk_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));
    if (k > 0) array.push_back(VElem::index_type(i+mj*j+mk*(k-1)));
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_-1)); idx -= j*(nk_-1)*(ni_-1);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (j < (nj_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));    
    if (j > 0) array.push_back(VElem::index_type(i+mj*(j-1)+mk*k));
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_-1)*(nj_-1)); idx -= i*(nk_-1)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (i < (ni_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));        
    if (i > 0) array.push_back(VElem::index_type((i-1)+mj*j+mk*k));
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_elems(VElem::array_type &array,
                             VCell::index_type idx) const
{
  array.resize(1); array[0] = VElem::index_type(idx);
}

template <class Basis>
void
LatVolMesh<Basis>::get_elems(VElem::array_type &array,
                             VDElem::index_type fidx) const
{
  array.reserve(2);
  array.clear();
  
  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);
  unsigned int idx = fidx;

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);

  if (idx < offset1)
  {
    unsigned int k = idx/((nj_-1)*(ni_-1)); idx -= k*(nj_-1)*(ni_-1);
    unsigned int j = idx/(ni_-1); idx -= j*(ni_-1);
    unsigned int i = idx;

    if (k < (nk_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));
    if (k > 0) array.push_back(VElem::index_type(i+mj*j+mk*(k-1)));
  }
  else if (idx >= offset2)
  {
    idx -= offset2;
    unsigned int j = idx/((nk_-1)*(ni_-1)); idx -= j*(nk_-1)*(ni_-1);
    unsigned int i = idx/(nk_-1); idx -= i*(nk_-1);
    unsigned int k = idx;

    if (j < (nj_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));    
    if (j > 0) array.push_back(VElem::index_type(i+mj*(j-1)+mk*k));
  }
  else
  {
    idx -= offset1;
    unsigned int i = idx/((nk_-1)*(nj_-1)); idx -= i*(nk_-1)*(nj_-1);
    unsigned int k = idx/(nj_-1); idx -= k*(nj_-1);
    unsigned int j = idx;

    if (i < (ni_-1)) array.push_back(VElem::index_type(i+mj*j+mk*k));        
    if (i > 0) array.push_back(VElem::index_type((i-1)+mj*j+mk*k));
  }

}

template <class Basis>
void
LatVolMesh<Basis>::get_delems(VDElem::array_type &array,
                              VCell::index_type idx) const
{
  const unsigned int i = idx % (ni_-1);
  const unsigned int jk = idx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(6);

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = VDElem::index_type(i+(j+k*(nj_-1))*(ni_-1));
  array[1] = VDElem::index_type(i+(j+(k+1)*(nj_-1))*(ni_-1));

  array[2] = VDElem::index_type(offset1+j+(k+i*(nk_-1))*(nj_-1));
  array[3] = VDElem::index_type(offset1+j+(k+(i+1)*(nk_-1))*(nj_-1));

  array[4] = VDElem::index_type(offset2+k+(i+j*(ni_-1))*(nk_-1));
  array[5] = VDElem::index_type(offset2+k+(i+(j+1)*(ni_-1))*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::get_delems(VDElem::array_type &array,
                              VElem::index_type idx) const
{
  const unsigned int i = idx % (ni_-1);
  const unsigned int jk = idx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.resize(6);

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  array[0] = VDElem::index_type(i+(j+k*(nj_-1))*(ni_-1));
  array[1] = VDElem::index_type(i+(j+(k+1)*(nj_-1))*(ni_-1));

  array[2] = VDElem::index_type(offset1+j+(k+i*(nk_-1))*(nj_-1));
  array[3] = VDElem::index_type(offset1+j+(k+(i+1)*(nk_-1))*(nj_-1));

  array[4] = VDElem::index_type(offset2+k+(i+j*(ni_-1))*(nk_-1));
  array[5] = VDElem::index_type(offset2+k+(i+(j+1)*(ni_-1))*(nk_-1));
}

template <class Basis>
void
LatVolMesh<Basis>::get_delems(VDElem::array_type &array,
                              VFace::index_type idx) const
{
  array.resize(1); array[0] = VDElem::index_type(idx);
}


template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p, VNode::index_type idx) const
{
  typename Node::index_type i;
  to_index(i,idx);
  get_center(p,i); 
}

template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p,Mesh::VEdge::index_type idx) const
{
  typename Edge::index_type i;
  to_index(i,idx);
  get_center(p,i);
}

template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p, Mesh::VFace::index_type idx) const
{
  typename Face::index_type i;
  to_index(i,idx);
  get_center(p, i);
}

template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p, Mesh::VCell::index_type idx) const
{
  typename Cell::index_type i;
  to_index(i,idx);
  get_center(p, i);
}

template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p, Mesh::VElem::index_type idx) const
{
  typename Elem::index_type i;
  to_index(i,idx);
  get_center(p, i);
}

template <class Basis>
void
LatVolMesh<Basis>::get_center(Point &p, Mesh::VDElem::index_type idx) const
{
  typename DElem::index_type i;
  to_index(i,idx);
  get_center(p, i);
}


template <class Basis>
void
LatVolMesh<Basis>::get_weights(const Point& p,VNode::array_type& nodes,
                                              vector<double>& weights) const
{
  typename Cell::index_type idx;
  
  if (locate(idx, p))
  {
    get_nodes(nodes,VCell::index_type(idx));
    vector<double> coords(3);
    if (get_coords(coords, p, idx))
    {
      weights.resize(basis_.dofs());
      basis_.get_weights(coords, &(weights[0]));
    }
  }
}

template <class Basis>
void
LatVolMesh<Basis>::get_weights(const Point& p,VElem::array_type& elems,
                                              vector<double>& weights) const
{
  typename Cell::index_type idx;
  if (locate(idx, p))
  {
    elems.resize(1);
    weights.resize(1);
    elems[0] = static_cast<VElem::index_type>(idx);
    weights[0] = 1.0;
  }
  else
  {
    elems.resize(0);
    weights.resize(0);
  }
}

template <class Basis>
bool 
LatVolMesh<Basis>::locate(VNode::index_type &vi, const Point &point) const
{
  typename Node::index_type i;
  bool ret = locate(i,point);
  vi = static_cast<VNode::index_type>(i);
  return (ret);
}

template <class Basis>
bool 
LatVolMesh<Basis>::locate(VElem::index_type &vi, const Point &point) const
{
  typename Elem::index_type i;
  bool ret = locate(i,point);
  vi = static_cast<VElem::index_type>(i);
  return (ret);
}

template <class Basis>
bool 
LatVolMesh<Basis>::get_coords(vector<double> &coords, const Point &point, 
                                                    VElem::index_type i) const
{
  typename Elem::index_type vi;
  to_index(vi,i);
  return(get_coords(coords,point,vi));
}  
  
template <class Basis>
void 
LatVolMesh<Basis>::interpolate(Point &p, const vector<double> &coords, 
                                                    VElem::index_type i) const
{
  typename Elem::index_type vi;
  to_index(vi,i);
  interpolate(p,coords,vi);
}

template <class Basis>
void 
LatVolMesh<Basis>::derivate(vector<Point> &p, const vector<double> &coords, 
                                                    VElem::index_type i) const
{
  typename Elem::index_type vi;
  to_index(vi,i);
  derivate(coords,vi,p);
}

template <class Basis>
bool 
LatVolMesh<Basis>::get_neighbor(VElem::index_type &neighbor, 
                        VElem::index_type from, VDElem::index_type delem) const
{
  const unsigned int xidx = static_cast<const unsigned int>(from);
  const unsigned int xdelem = static_cast<const unsigned int>(delem);

  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);

  const unsigned int offset1 = (ni_ - 1) * (nj_ - 1) * nk_;
  const unsigned int offset2 = offset1 + ni_ * (nj_ - 1) * (nk_ - 1);

  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);
  
  if (xdelem == (i+(j+k*(nj_-1))*(ni_-1))) 
  {
    if (k <= 0) return (false);
    neighbor = static_cast<VElem::index_type>(i+mj*j+mk*(k-1));
    return (true);
  }

  if (xdelem == (i+(j+(k+1)*(nj_-1))*(ni_-1))) 
  {
    if (k >= nk_-2) return (false);
    neighbor = static_cast<VElem::index_type>(i+mj*j+mk*(k+1));
    return (true);
  }

  if (xdelem == (offset1+j+(k+i*(nk_-1))*(nj_-1))) 
  {
    if (i <= 0) return (false);
    neighbor = static_cast<VElem::index_type>((i-1)+mj*j+mk*k);
    return (true);
  }

  if (xdelem == (offset1+j+(k+(i+1)*(nk_-1))*(nj_-1))) 
  {
    if (i >= ni_-2) return (false);
    neighbor = static_cast<VElem::index_type>((i+1)+mj*j+mk*k);
    return (true);
  }

  if (xdelem == (offset2+k+(i+j*(ni_-1))*(nk_-1))) 
  {
    if (j <= 0) return (false);
    neighbor = static_cast<VElem::index_type>(i+mj*(j-1)+mk*k);
    return (true);
  }

  if (xdelem == (offset2+k+(i+(j+1)*(ni_-1))*(nk_-1))) 
  {
    if (j >= nj_-2) return (false);
    neighbor = static_cast<VElem::index_type>(i+mj*(j+1)+mk*k);
    return (true);
  }  

  return (false);
}

template <class Basis>
void 
LatVolMesh<Basis>::get_neighbors(VElem::array_type &array, 
                                                    VElem::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);
  const unsigned int i = xidx % (ni_-1);
  const unsigned int jk = xidx / (ni_-1);
  const unsigned int j = jk % (nj_-1);
  const unsigned int k = jk / (nj_-1);
  
  array.reserve(6);
  array.clear();
  
  const unsigned int mj = ni_-1;
  const unsigned int mk = (ni_-1)*(nj_-1);
  if (i > 0) array.push_back(static_cast<VElem::index_type>((i-1)+j*mj+k*mk));
  if (i < ni_-2) array.push_back(static_cast<VElem::index_type>((i+1)+j*mj+k*mk));
  if (j > 0) array.push_back(static_cast<VElem::index_type>(i+(j-1)*mj+k*mk));
  if (j < nj_-2) array.push_back(static_cast<VElem::index_type>(i+(j+1)*mj+k*mk));
  if (k > 0) array.push_back(static_cast<VElem::index_type>(i+j*mj+(k-1)*mk));
  if (k < nk_-2) array.push_back(static_cast<VElem::index_type>(i+j*mj+(k+1)*mk));
}

template <class Basis>
void 
LatVolMesh<Basis>::get_neighbors(VNode::array_type &array, 
                                                    VNode::index_type idx) const
{
  const unsigned int xidx = static_cast<const unsigned int>(idx);
  const unsigned int i = xidx % ni_;
  const unsigned int jk = xidx / ni_;
  const unsigned int j = jk % nj_;
  const unsigned int k = jk / nj_;
  
  array.reserve(6);
  array.clear();
  
  const unsigned int mj = ni_;
  const unsigned int mk = ni_*nj_;
  if (i > 0) array.push_back(static_cast<VNode::index_type>((i-1)+j*mj+k*mk));
  if (i < ni_-1) array.push_back(static_cast<VNode::index_type>((i+1)+j*mj+k*mk));
  if (j > 0) array.push_back(static_cast<VNode::index_type>(i+(j-1)*mj+k*mk));
  if (j < nj_-1) array.push_back(static_cast<VNode::index_type>(i+(j+1)*mj+k*mk));
  if (k > 0) array.push_back(static_cast<VNode::index_type>(i+j*mj+(k-1)*mk));
  if (k < nk_-1) array.push_back(static_cast<VNode::index_type>(i+j*mj+(k+1)*mk));
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VNode::index_type i) const
{
  return (0.0);
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VEdge::index_type i) const
{
  typename Edge::index_type vi; to_index(vi,i);
  return (get_size(vi));
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VFace::index_type i) const
{
  typename Face::index_type vi; to_index(vi,i);
  return (get_size(vi));
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VCell::index_type i) const
{
  typename Cell::index_type vi; to_index(vi,i);
  return (get_size(vi));
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VElem::index_type i) const
{
  typename Elem::index_type vi; to_index(vi,i);
  return (get_size(vi));
}

template <class Basis>
double
LatVolMesh<Basis>::get_size(VDElem::index_type i) const
{
  typename DElem::index_type vi; to_index(vi,i);
  return (get_size(vi));
}


template <class Basis>
void 
LatVolMesh<Basis>::pwl_approx_edge(vector<vector<double> > &coords, 
                                  VElem::index_type ci, unsigned int which_edge,
                                  unsigned int div_per_unit) const
{
  basis_.approx_edge(which_edge, div_per_unit, coords);
}

template <class Basis>
void 
LatVolMesh<Basis>::pwl_approx_face(vector<vector<vector<double> > > &coords, 
                                  VElem::index_type ci, unsigned int which_face,
                                  unsigned int div_per_unit) const
{
  basis_.approx_face(which_face, div_per_unit, coords);
}

template <class Basis>
void 
LatVolMesh<Basis>::get_random_point(Point &p, VElem::index_type i,
                                                          MusilRNG &rng) const
{
  typename Elem::index_type vi; to_index(vi,i);
  get_random_point(p,vi,rng);
}

template <class Basis>
void 
LatVolMesh<Basis>::get_dimensions(Mesh::dimension_type& dims)
{
  dims.resize(3);
  dims[0] = ni_;
  dims[1] = nj_;
  dims[2] = nk_;
}


} // namespace SCIRun

#endif 


