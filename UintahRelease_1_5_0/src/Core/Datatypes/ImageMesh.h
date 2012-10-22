/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*
 *  ImageMesh.h: Templated Mesh defined on a 2D Regular Grid
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *
 */

#ifndef SCI_project_ImageMesh_h
#define SCI_project_ImageMesh_h 1

#include <Core/Geometry/BBox.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Math/MusilRNG.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/StackVector.h>

namespace SCIRun {

using std::string;

template <class Basis>
class ImageMesh : public Mesh {
public:
  typedef LockingHandle<ImageMesh<Basis> > handle_type;
  typedef Basis        basis_type;
  struct ImageIndex;
  friend struct ImageIndex;

  struct ImageIndex
  {
  public:
    ImageIndex() : i_(0), j_(0), mesh_(0) {}

    ImageIndex(const ImageMesh *m, unsigned i, unsigned j)
      : i_(i), j_(j), mesh_(m) {}

    operator unsigned() const {
      ASSERT(mesh_);
      return i_ + j_*mesh_->ni_;
    }

    std::ostream& str_render(std::ostream& os) const {
      os << "[" << i_ << "," << j_ << "]";
      return os;
    }

    unsigned i_, j_;

    const ImageMesh *mesh_;
  };

  struct IFaceIndex : public ImageIndex
  {
    IFaceIndex() : ImageIndex() {}
    IFaceIndex(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIndex(m, i, j) {}

    operator unsigned() const {
      ASSERT(this->mesh_);
      return this->i_ + this->j_ * (this->mesh_->ni_-1);
    }
  };

  struct INodeIndex : public ImageIndex
  {
    INodeIndex() : ImageIndex() {}
    INodeIndex(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIndex(m, i, j) {}
  };

  struct ImageIter : public ImageIndex
  {
    ImageIter() : ImageIndex() {}
    ImageIter(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIndex(m, i, j) {}

    const ImageIndex &operator *() { return *this; }

    bool operator ==(const ImageIter &a) const
    {
      return this->i_ == a.i_ && this->j_ == a.j_ && this->mesh_ == a.mesh_;
    }

    bool operator !=(const ImageIter &a) const
    {
      return !(*this == a);
    }
  };

  struct INodeIter : public ImageIter
  {
    INodeIter() : ImageIter() {}
    INodeIter(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIter(m, i, j) {}

    const INodeIndex &operator *() const { return (const INodeIndex&)(*this); }

    INodeIter &operator++()
    {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_ + this->mesh_->ni_) {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
      }
      return *this;
    }

  private:

    INodeIter operator++(int)
    {
      INodeIter result(*this);
      operator++();
      return result;
    }
  };


  struct IFaceIter : public ImageIter
  {
    IFaceIter() : ImageIter() {}
    IFaceIter(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIter(m, i, j) {}

    const IFaceIndex &operator *() const { return (const IFaceIndex&)(*this); }

    IFaceIter &operator++()
    {
      this->i_++;
      if (this->i_ >= this->mesh_->min_i_+this->mesh_->ni_-1) {
        this->i_ = this->mesh_->min_i_;
        this->j_++;
      }
      return *this;
    }

  private:

    IFaceIter operator++(int)
    {
      IFaceIter result(*this);
      operator++();
      return result;
    }
  };

  struct ImageSize
  {
  public:
    ImageSize() : i_(0), j_(0) {}
    ImageSize(unsigned i, unsigned j) : i_(i), j_(j) {}

    operator unsigned() const { return i_*j_; }

    std::ostream& str_render(std::ostream& os) const {
      os << i_*j_ << " (" << i_ << " x " << j_ << ")";
      return os;
    }


    unsigned i_, j_;
  };

  struct INodeSize : public ImageSize
  {
    INodeSize() : ImageSize() {}
    INodeSize(unsigned i, unsigned j) : ImageSize(i,j) {}
  };


  struct IFaceSize : public ImageSize
  {
    IFaceSize() : ImageSize() {}
    IFaceSize(unsigned i, unsigned j) : ImageSize(i,j) {}
  };


  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef INodeIndex                       index_type;
    typedef INodeIter                        iterator;
    typedef INodeSize                        size_type;
    typedef StackVector<index_type, 4>       array_type;
  };

  struct Edge {
    typedef EdgeIndex<unsigned int>          index_type;
    typedef EdgeIterator<unsigned int>       iterator;
    typedef EdgeIndex<unsigned int>          size_type;
    typedef vector<index_type>               array_type;
  };

  struct Face {
    typedef IFaceIndex                       index_type;
    typedef IFaceIter                        iterator;
    typedef IFaceSize                        size_type;
    typedef vector<index_type>               array_type;
  };

  struct Cell {
    typedef CellIndex<unsigned int>          index_type;
    typedef CellIterator<unsigned int>       iterator;
    typedef CellIndex<unsigned int>          size_type;
    typedef vector<index_type>               array_type;
  };

  typedef Face Elem;
  typedef Edge DElem;

  friend class INodeIter;
  friend class IFaceIter;
  friend class IFaceIndex;

  friend class ElemData;

  class ElemData
  {
  public:
    ElemData(const ImageMesh<Basis>& msh,
             const typename Elem::index_type ind) :
      mesh_(msh),
      index_(ind)
    {}

    // the following designed to coordinate with ::get_nodes
    inline
    unsigned node0_index() const {
      return (index_.i_ + mesh_.get_ni()*index_.j_);
    }
    inline
    unsigned node1_index() const {
      return (index_.i_+ 1 + mesh_.get_ni()*index_.j_);
    }
    inline
    unsigned node2_index() const {
      return (index_.i_ + 1 + mesh_.get_ni()*(index_.j_ + 1));

    }
    inline
    unsigned node3_index() const {
      return (index_.i_ + mesh_.get_ni()*(index_.j_ + 1));
    }

    // the following designed to coordinate with ::get_edges
    inline
    unsigned edge0_index() const {
      return index_.i_ + index_.j_ * (mesh_.ni_- 1);
    }
    inline
    unsigned edge1_index() const {
      return index_.i_ + (index_.j_ + 1) * (mesh_.ni_ - 1);
    }
    inline
    unsigned edge2_index() const {
      return index_.i_    *(mesh_.nj_ - 1) + index_.j_ +
        ((mesh_.ni_ - 1) * mesh_.nj_);
     }
    inline
    unsigned edge3_index() const {
      return (index_.i_ + 1) * (mesh_.nj_ - 1) + index_.j_ +
        ((mesh_.ni_ - 1) * mesh_.nj_);
    }


    inline
    const Point node0() const {
      Point p(index_.i_, index_.j_, 0.0);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node1() const {
      Point p(index_.i_ + 1, index_.j_, 0.0);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node2() const {
      Point p(index_.i_ + 1, index_.j_ + 1, 0.0);
      return mesh_.transform_.project(p);
    }
    inline
    const Point node3() const {
      Point p(index_.i_, index_.j_ + 1, 0.0);
      return mesh_.transform_.project(p);
    }


  private:
    const ImageMesh<Basis>          &mesh_;
    const typename Elem::index_type  index_;
  };


  ImageMesh()
    : min_i_(0), min_j_(0),
      ni_(1), nj_(1) {}
  ImageMesh(unsigned x, unsigned y, const Point &min, const Point &max);
  ImageMesh(ImageMesh* mh, unsigned int mx, unsigned int my,
            unsigned int x, unsigned int y)
    : min_i_(mx), min_j_(my), ni_(x), nj_(y), transform_(mh->transform_) {}
  ImageMesh(const ImageMesh &copy)
    : min_i_(copy.min_i_), min_j_(copy.min_j_),
      ni_(copy.get_ni()), nj_(copy.get_nj()), transform_(copy.transform_) {}
  virtual ImageMesh *clone() { return new ImageMesh(*this); }
  virtual ~ImageMesh() {}

  Basis& get_basis() { return basis_; }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an edge.
  void pwl_approx_edge(vector<vector<double> > &coords,
                       typename Elem::index_type ci,
                       unsigned which_edge,
                       unsigned div_per_unit) const
  {
    // Needs to match unit_edges in Basis/QuadBilinearLgn.cc
    // compare get_nodes order to the basis order
    int basis_emap[] = {0, 2, 3, 1};
    basis_.approx_edge(basis_emap[which_edge], div_per_unit, coords);
  }

  //! Generate the list of points that make up a sufficiently accurate
  //! piecewise linear approximation of an face.
  void pwl_approx_face(vector<vector<vector<double> > > &coords,
                       typename Elem::index_type,
                       unsigned,
                       unsigned div_per_unit) const
  {
    basis_.approx_face(0, div_per_unit, coords);
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
  void derivate(const vector<double> &coords,
                typename Elem::index_type idx,
                vector<Point> &J) const
  {
    ElemData ed(*this, idx);
    basis_.derivate(coords, ed, J);
  }

  //! get the mesh statistics
  unsigned get_min_i() const { return min_i_; }
  unsigned get_min_j() const { return min_j_; }
  bool get_min(vector<unsigned int>&) const;
  unsigned get_ni() const { return ni_; }
  unsigned get_nj() const { return nj_; }
  bool get_dim(vector<unsigned int>&) const;
  Vector diagonal() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);
  virtual void get_canonical_transform(Transform &t);
  virtual bool synchronize(unsigned int);

  //! set the mesh statistics
  void set_min_i(unsigned i) {min_i_ = i; }
  void set_min_j(unsigned j) {min_j_ = j; }
  void set_min(vector<unsigned int> mins);
  void set_ni(unsigned i) { ni_ = i; }
  void set_nj(unsigned j) { nj_ = j; }
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

  void to_index(typename Node::index_type &index, unsigned int i);
  void to_index(typename Edge::index_type &index, unsigned int i) { index= i; }
  void to_index(typename Face::index_type &index, unsigned int i);
  void to_index(typename Cell::index_type &index, unsigned int i) { index= i; }

  //! get the child elements of the given index
  void get_nodes(typename Node::array_type &, typename Edge::index_type) const;
  void get_nodes(typename Node::array_type &, typename Face::index_type) const;
  void get_nodes(typename Node::array_type &, typename Cell::index_type) const {}
  void get_edges(typename Edge::array_type &, typename Face::index_type) const;
  void get_edges(typename Edge::array_type &, typename Cell::index_type) const {}
  void get_faces(typename Face::array_type &, typename Cell::index_type) const {}
  void get_faces(typename Face::array_type &a,
                 typename Face::index_type f) const
  { a.push_back(f); }

  //! get the parent element(s) of the given index
  void get_elems(typename Elem::array_type &result,
                 typename Node::index_type idx) const;
  void get_elems(typename Elem::array_type &result,
                 typename Edge::index_type idx) const;
  void get_elems(typename Elem::array_type &result,
                 typename Face::index_type idx) const {}


  //! Wrapper to get the derivative elements from this element.
  void get_delems(typename DElem::array_type &result,
                  const typename Elem::index_type &idx) const
  {
    get_edges(result, idx);
  }

  //! return all face_indecies that overlap the BBox in arr.
  void get_faces(typename Face::array_type &arr, const BBox &box);

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const typename Node::index_type &/*i*/) const { return 0.0; }
  double get_size(typename Edge::index_type idx) const
  {
    typename Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(const typename Face::index_type &idx) const
  {
    typename Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length();
  }
  double get_size(typename Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(typename Edge::index_type idx) const { return get_size(idx); };
  double get_area(typename Face::index_type idx) const { return get_size(idx); };
  double get_volume(typename Cell::index_type idx) const { return get_size(idx); };

  int get_valence(const typename Node::index_type &idx) const
  {
    return ((idx.i_ == 0 || idx.i_ == ni_ - 1 ? 1 : 2) +
            (idx.j_ == 0 || idx.j_ == nj_ - 1 ? 1 : 2));
  }
  int get_valence(typename Edge::index_type idx) const;
  int get_valence(const typename Face::index_type &/*idx*/) const { return 0; }
  int get_valence(typename Cell::index_type /*idx*/) const { return 0; }


  bool get_neighbor(typename Face::index_type &neighbor,
                    typename Face::index_type from,
                    typename Edge::index_type idx) const;

  void get_neighbors(typename Face::array_type &array, typename Face::index_type idx) const;

  void get_normal(Vector &, const typename Node::index_type &) const
  { ASSERTFAIL("This mesh type does not have node normals."); }
  void get_normal(Vector &result, vector<double> &coords,
                  typename Elem::index_type eidx, unsigned int)
  {
    ElemData ed(*this, eidx);
    vector<Point> Jv;
    basis_.derivate(coords, ed, Jv);
    result = Cross(Jv[0].asVector(), Jv[1].asVector());
    result.normalize();
  }

  //! get the center point (in object space) of an element
  void get_center(Point &, const typename Node::index_type &) const;
  void get_center(Point &, typename Edge::index_type) const;
  void get_center(Point &, const typename Face::index_type &) const;
  void get_center(Point &, typename Cell::index_type) const {}

  bool locate(typename Node::index_type &, const Point &);
  bool locate(typename Edge::index_type &, const Point &) const { return false; }
  bool locate(typename Face::index_type &, const Point &);
  bool locate(typename Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &p, typename Node::array_type &l, double *w);
  int get_weights(const Point & , typename Edge::array_type & , double * )
  {ASSERTFAIL("ImageMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point &p, typename Face::array_type &l, double *w);
  int get_weights(const Point & , typename Cell::array_type & , double * )
  {ASSERTFAIL("ImageMesh::get_weights for cells isn't supported"); }

  void get_point(Point &p, const typename Node::index_type &i) const
  { get_center(p, i); }

  void get_random_point(Point &, const typename Elem::index_type &, MusilRNG &rng) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans)
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 2; }
  virtual int  topology_geometry() const { return (STRUCTURED | REGULAR); }

  static const TypeDescription* node_type_description();
  static const TypeDescription* edge_type_description();
  static const TypeDescription* face_type_description();
  static const TypeDescription* cell_type_description();
  static const TypeDescription* elem_type_description()
  { return face_type_description(); }
  static const TypeDescription* node_index_type_description();
  static const TypeDescription* face_index_type_description();

  // returns a ImageMesh
  static Persistent *maker() { return new ImageMesh<Basis>(); }

protected:

  //! the min_typename Node::index_type ( incase this is a subLattice )
  unsigned               min_i_;
  unsigned               min_j_;
  //! the typename Node::index_type space extents of a ImageMesh
  //! (min=min_typename Node::index_type, max=min+extents-1)
  unsigned               ni_;
  unsigned               nj_;

  //! the object space extents of a ImageMesh
  Transform              transform_;

  Vector                 normal_;

  //! The basis class
  Basis                  basis_;

};


template <class Basis>
PersistentTypeID
ImageMesh<Basis>::type_id(type_name(-1),"Mesh", maker);


template<class Basis>
ImageMesh<Basis>::ImageMesh(unsigned i, unsigned j,
                            const Point &min, const Point &max) :
  min_i_(0), min_j_(0), ni_(i), nj_(j)
{
  transform_.pre_scale(Vector(1.0 / (i-1.0), 1.0 / (j-1.0), 1.0));
  transform_.pre_scale(max - min);
  transform_.pre_translate(Vector(min));
  transform_.compute_imat();

  normal_ = Vector(0.0, 0.0, 0.0);
  transform_.project_normal(normal_);
  normal_.safe_normalize();
}


template<class Basis>
BBox
ImageMesh<Basis>::get_bounding_box() const
{
  Point p0(0.0,   0.0,   0.0);
  Point p1(ni_-1, 0.0,   0.0);
  Point p2(ni_-1, nj_-1, 0.0);
  Point p3(0.0,   nj_-1, 0.0);

  BBox result;
  result.extend(transform_.project(p0));
  result.extend(transform_.project(p1));
  result.extend(transform_.project(p2));
  result.extend(transform_.project(p3));
  return result;
}


template<class Basis>
Vector
ImageMesh<Basis>::diagonal() const
{
  return get_bounding_box().diagonal();
}


template<class Basis>
void
ImageMesh<Basis>::get_canonical_transform(Transform &t)
{
  t = transform_;
  t.post_scale(Vector(ni_ - 1.0, nj_ - 1.0, 1.0));
}


template<class Basis>
bool
ImageMesh<Basis>::synchronize(unsigned int flag)
{
  if (flag & NORMALS_E)
  {
    normal_ = Vector(0.0, 0.0, 0.0);
    transform_.project_normal(normal_);
    normal_.safe_normalize();
    return true;
  }
  return false;
}


template<class Basis>
void
ImageMesh<Basis>::transform(const Transform &t)
{
  transform_.pre_trans(t);
}


template<class Basis>
bool
ImageMesh<Basis>::get_min(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(min_i_);
  array.push_back(min_j_);

  return true;
}


template<class Basis>
bool
ImageMesh<Basis>::get_dim(vector<unsigned int> &array) const
{
  array.resize(2);
  array.clear();

  array.push_back(ni_);
  array.push_back(nj_);

  return true;
}


template<class Basis>
void
ImageMesh<Basis>::set_min(vector<unsigned int> min)
{
  min_i_ = min[0];
  min_j_ = min[1];
}


template<class Basis>
void
ImageMesh<Basis>::set_dim(vector<unsigned int> dim)
{
  ni_ = dim[0];
  nj_ = dim[1];
}


template<class Basis>
void
ImageMesh<Basis>::get_nodes(typename Node::array_type &array,
                            typename Face::index_type idx) const
{
  array.resize(4);

  array[0].i_ = idx.i_;   array[0].j_ = idx.j_;
  array[1].i_ = idx.i_+1; array[1].j_ = idx.j_;
  array[2].i_ = idx.i_+1; array[2].j_ = idx.j_+1;
  array[3].i_ = idx.i_;   array[3].j_ = idx.j_+1;

  array[0].mesh_ = idx.mesh_;
  array[1].mesh_ = idx.mesh_;
  array[2].mesh_ = idx.mesh_;
  array[3].mesh_ = idx.mesh_;
}


template<class Basis>
void
ImageMesh<Basis>::get_nodes(typename Node::array_type &array,
                            typename Edge::index_type idx) const
{
  array.resize(2);

  const int j_idx = idx - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const int i = j_idx / (nj_ - 1);
    const int j = j_idx % (nj_ - 1);
    array[0] = typename Node::index_type(this, i, j);
    array[1] = typename Node::index_type(this, i, j+1);
  }
  else
  {
    const int i = idx % (ni_ - 1);
    const int j = idx / (ni_ - 1);
    array[0] = typename Node::index_type(this, i, j);
    array[1] = typename Node::index_type(this, i+1, j);
  }
}


template<class Basis>
void
ImageMesh<Basis>::get_edges(typename Edge::array_type &array, typename Face::index_type idx) const
{
  array.resize(4);

  const int j_idx = (ni_-1) * nj_;

  array[0] = idx.i_             + idx.j_    *(ni_-1);
  array[1] = idx.i_             + (idx.j_+1)*(ni_-1);
  array[2] = idx.i_    *(nj_-1) + idx.j_+ j_idx;
  array[3] = (idx.i_+1)*(nj_-1) + idx.j_+ j_idx;
}

template<class Basis>
void
ImageMesh<Basis>::get_elems(typename Face::array_type &array, typename Edge::index_type idx) const
{
  array.reserve(2);
  array.clear();
  
  const unsigned int offset = (ni_-1)*nj_;
  
  if (idx < offset)
  {
    const unsigned int j = idx/(ni_-1); 
    const unsigned int i = idx - j*(ni_-1);
    if (j < (nj_-1)) array.push_back(IFaceIndex(this,i,j));
    if (j > 0) array.push_back(IFaceIndex(this,i,j-1));  
  }
  else
  {
    idx -= offset;
    const unsigned int i = idx/(nj_-1); 
    const unsigned int j = idx - i*(nj_-1);
    if (i < (ni_-1)) array.push_back(IFaceIndex(this,i,j));
    if (i > 0) array.push_back(IFaceIndex(this,i-1));  
  }
}




template<class Basis>
bool
ImageMesh<Basis>::get_neighbor(typename Face::index_type &neighbor,
                               typename Face::index_type from,
                               typename Edge::index_type edge) const
{
  neighbor.mesh_ = this;
  const int j_idx = edge - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const unsigned int i = j_idx / (nj_ - 1);
    if (i == 0 || i == ni_-1)
      return false;
    neighbor.j_ = from.j_;
    if (i == from.i_)
      neighbor.i_ = from.i_ - 1;
    else
      neighbor.i_ = from.i_ + 1;
  }
  else
  {
    const unsigned int j = edge / (ni_ - 1);;
    if (j == 0 || j == nj_-1)
      return false;
    neighbor.i_ = from.i_;
    if (j == from.j_)
      neighbor.j_ = from.j_ - 1;
    else
      neighbor.j_ = from.j_ + 1;
  }
  return true;
}


template<class Basis>
void
ImageMesh<Basis>::get_neighbors(typename Face::array_type &array,
                                typename Face::index_type idx) const
{
  typename Edge::array_type edges;
  get_edges(edges, idx);
  array.clear();
  typename Edge::array_type::iterator iter = edges.begin();
  while(iter != edges.end()) {
    typename Face::index_type nbor;
    if (get_neighbor(nbor, idx, *iter)) {
      array.push_back(nbor);
    }
    ++iter;
  }
}


template<class Basis>
int
ImageMesh<Basis>::get_valence(typename Edge::index_type idx) const
{
  const int j_idx = idx - (ni_-1) * nj_;
  if (j_idx >= 0)
  {
    const unsigned int i = j_idx / (nj_ - 1);
    return (i == 0 || i == ni_ - 1) ? 1 : 2;
  }
  else
  {
    const unsigned int j = idx / (ni_ - 1);
    return (j == 0 || j == nj_ - 1) ? 1 : 2;
  }
}


template<class Basis>
void
ImageMesh<Basis>::get_elems(typename Elem::array_type &result,
                            const typename Node::index_type idx) const
{
  result.reserve(4);
  result.clear();

  const unsigned int i0 = idx.i_ ? idx.i_ - 1 : 0;
  const unsigned int j0 = idx.j_ ? idx.j_ - 1 : 0;

  const unsigned int i1 = idx.i_ < ni_-1 ? idx.i_+1 : ni_-1;
  const unsigned int j1 = idx.j_ < nj_-1 ? idx.j_+1 : nj_-1;

  unsigned int i, j, k;
  for (j = j0; j < j1; j++)
    for (i = i0; i < i1; i++)
      result.push_back(typename Face::index_type(this, i, j));
}


//! return all face_indecies that overlap the BBox in arr.
template<class Basis>
void
ImageMesh<Basis>::get_faces(typename Face::array_type &arr, const BBox &bbox)
{
  arr.clear();
  typename Face::index_type min;
  locate(min, bbox.min());
  typename Face::index_type max;
  locate(max, bbox.max());

  if (max.i_ >= ni_ - 1) max.i_ = ni_ - 2;
  if (max.j_ >= nj_ - 1) max.j_ = nj_ - 2;

  for (unsigned i = min.i_; i <= max.i_; i++) {
    for (unsigned j = min.j_; j <= max.j_; j++) {
      arr.push_back(typename Face::index_type(this, i,j));
    }
  }
}


template<class Basis>
void
ImageMesh<Basis>::get_center(Point &result,
                             const typename Node::index_type &idx) const
{
  Point p(idx.i_, idx.j_, 0.0);
  result = transform_.project(p);
}


template<class Basis>
void
ImageMesh<Basis>::get_center(Point &result,
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


template<class Basis>
void
ImageMesh<Basis>::get_center(Point &result,
                             const typename Face::index_type &idx) const
{
  Point p(idx.i_ + 0.5, idx.j_ + 0.5, 0.0);
  result = transform_.project(p);
}


template<class Basis>
bool
ImageMesh<Basis>::locate(typename Face::index_type &face, const Point &p)
{
  if (basis_.polynomial_order() > 1) return elem_locate(face, *this, p);
  const Point r = transform_.unproject(p);

  const double rx = floor(r.x());
  const double ry = floor(r.y());

  // Clip before int conversion to avoid overflow errors.
  if (rx < 0.0 || ry < 0.0 || rx >= (ni_-1) || ry >= (nj_-1))
  {
    return false;
  }

  face.i_ = (unsigned int)rx;
  face.j_ = (unsigned int)ry;
  face.mesh_ = this;

  return true;
}


template<class Basis>
bool
ImageMesh<Basis>::locate(typename Node::index_type &node, const Point &p)
{
  const Point r = transform_.unproject(p);

  const double rx = floor(r.x() + 0.5);
  const double ry = floor(r.y() + 0.5);

  // Clip before int conversion to avoid overflow errors.
  if (rx < 0.0 || ry < 0.0 || rx >= ni_ || ry >= nj_)
  {
    return false;
  }

  node.i_ = (unsigned int)rx;
  node.j_ = (unsigned int)ry;
  node.mesh_ = this;

  return true;
}


template <class Basis>
int
ImageMesh<Basis>::get_weights(const Point &p, typename Node::array_type &l,
                              double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    get_nodes(l,idx);
    vector<double> coords(2);
    if (get_coords(coords, p, idx))
    {
      basis_.get_weights(coords, w);
      return basis_.dofs();
    }
  }
  return 0;
}


template <class Basis>
int
ImageMesh<Basis>::get_weights(const Point &p, typename Face::array_type &l,
                              double *w)
{
  typename Face::index_type idx;
  if (locate(idx, p))
  {
    l.resize(1);
    l[0] = idx;
    w[0] = 1.0;
    return 1;
  }
  return 0;
}


/* To generate a random point inside of a triangle, we generate random
   barrycentric coordinates (independent random variables between 0 and
   1 that sum to 1) for the point. */

template<class Basis>
void
ImageMesh<Basis>::get_random_point(Point &p,
                                   const typename Elem::index_type &ci,
                                   MusilRNG &rng) const
{
  // Get the positions of the vertices.
  typename Node::array_type ra;
  get_nodes(ra,ci);
  Point p00, p10, p11, p01;
  get_center(p00,ra[0]);
  get_center(p10,ra[1]);
  get_center(p11,ra[2]);
  get_center(p01,ra[3]);
  Vector dx=p10-p00;
  Vector dy=p01-p00;

  // Generate the barycentric coordinates.
  const double u = rng();
  const double v = rng();

  // Compute the position of the random point.
  p = p00 + dx*u + dy*v;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::node_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription(ImageMesh<Basis>::type_name(-1) +
                                string("::INodeIndex"),
                                string(__FILE__),
                                "SCIRun");
  }
  return td;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::face_index_type_description()
{
  static TypeDescription* td = 0;
  if(!td){
    td = scinew TypeDescription(ImageMesh<Basis>::type_name(-1) +
                                string("::IFaceIndex"),
                                string(__FILE__),
                                "SCIRun");
  }
  return td;
}


template<class Basis>
void
Pio(Piostream& stream, typename ImageMesh<Basis>::INodeIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  stream.end_cheap_delim();
}


template<class Basis>
void
Pio(Piostream& stream, typename ImageMesh<Basis>::IFaceIndex& n)
{
  stream.begin_cheap_delim();
  Pio(stream, n.i_);
  Pio(stream, n.j_);
  stream.end_cheap_delim();
}


#define IMAGEMESH_VERSION 2

template<class Basis>
void
ImageMesh<Basis>::io(Piostream& stream)
{
  int version = stream.begin_class(type_name(-1), IMAGEMESH_VERSION);

  Mesh::io(stream);

  // IO data members, in order
  Pio(stream, ni_);
  Pio(stream, nj_);

  if (version >= 2) {
    basis_.io(stream);
  }

  stream.end_class();
}


template<class Basis>
const string
ImageMesh<Basis>::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if (n == 0)
  {
    static const string nm("ImageMesh");
    return nm;
  }
  else
  {
    return find_type_name((Basis *)0);
  }
}


template<class Basis>
void
ImageMesh<Basis>::begin(typename ImageMesh::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_);
}


template<class Basis>
void
ImageMesh<Basis>::end(typename ImageMesh::Node::iterator &itr) const
{
  itr = typename Node::iterator(this, min_i_, min_j_ + nj_);
}


template<class Basis>
void
ImageMesh<Basis>::size(typename ImageMesh::Node::size_type &s) const
{
  s = typename Node::size_type(ni_, nj_);
}


template<class Basis>
void
ImageMesh<Basis>::to_index(typename ImageMesh::Node::index_type &idx, unsigned int a)
{
  const unsigned int i = a % ni_;
  const unsigned int j = a / ni_;
  idx = typename Node::index_type(this, i, j);

}


template<class Basis>
void
ImageMesh<Basis>::begin(typename ImageMesh::Edge::iterator &itr) const
{
  itr = 0;
}


template<class Basis>
void
ImageMesh<Basis>::end(typename ImageMesh::Edge::iterator &itr) const
{
  itr = (ni_-1) * (nj_) + (ni_) * (nj_ -1);
}


template<class Basis>
void
ImageMesh<Basis>::size(typename ImageMesh::Edge::size_type &s) const
{
  s = (ni_-1) * (nj_) + (ni_) * (nj_ -1);
}


template<class Basis>
void
ImageMesh<Basis>::begin(typename ImageMesh::Face::iterator &itr) const
{
  itr = typename Face::iterator(this,  min_i_, min_j_);
}


template<class Basis>
void
ImageMesh<Basis>::end(typename ImageMesh::Face::iterator &itr) const
{
  itr = typename Face::iterator(this, min_i_, min_j_ + nj_ - 1);
}


template<class Basis>
void
ImageMesh<Basis>::size(typename ImageMesh::Face::size_type &s) const
{
  s = typename Face::size_type(ni_-1, nj_-1);
}


template<class Basis>
void
ImageMesh<Basis>::to_index(typename ImageMesh::Face::index_type &idx, unsigned int a)
{
  const unsigned int i = a % (ni_-1);
  const unsigned int j = a / (ni_-1);
  idx = typename Face::index_type(this, i, j);

}


template<class Basis>
void
ImageMesh<Basis>::begin(typename ImageMesh::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(0);
}


template<class Basis>
void
ImageMesh<Basis>::end(typename ImageMesh::Cell::iterator &itr) const
{
  itr = typename Cell::iterator(0);
}


template<class Basis>
void
ImageMesh<Basis>::size(typename ImageMesh::Cell::size_type &s) const
{
  s = typename Cell::size_type(0);
}


template<class Basis>
const TypeDescription*
get_type_description(ImageMesh<Basis> *)
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *sub = SCIRun::get_type_description((Basis*)0);
    TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
    (*subs)[0] = sub;
    td = scinew TypeDescription("ImageMesh", subs,
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::get_type_description() const
{
  return SCIRun::get_type_description((ImageMesh *)0);
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::node_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ImageMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Node",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::edge_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ImageMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Edge",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::face_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ImageMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Face",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}


template<class Basis>
const TypeDescription*
ImageMesh<Basis>::cell_type_description()
{
  static TypeDescription *td = 0;
  if (!td)
  {
    const TypeDescription *me =
      SCIRun::get_type_description((ImageMesh<Basis> *)0);
    td = scinew TypeDescription(me->get_name() + "::Cell",
                                string(__FILE__),
                                "SCIRun",
                                TypeDescription::MESH_E);
  }
  return td;
}

} // namespace SCIRun

#endif // SCI_project_ImageMesh_h
