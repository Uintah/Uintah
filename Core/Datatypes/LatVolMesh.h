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

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/share/share.h>
#include <Core/Math/MusilRNG.h>
#include <string>
#include <iostream>

namespace SCIRun {

using std::string;
using std::cerr;
using std::endl;

class SCICORESHARE LatVolMesh : public Mesh
{
public:

  static const string& get_h_file_path();

  struct UnfinishedIndex
  {
  public:
    UnfinishedIndex() : i_(0) {}
    UnfinishedIndex(unsigned i) : i_(i) {}

    operator unsigned() const { return i_; }

    unsigned i_;
  };

  struct EdgeIndex : public UnfinishedIndex
  {
    EdgeIndex() : UnfinishedIndex() {}
    EdgeIndex(unsigned i) : UnfinishedIndex(i) {}
    friend void Pio(Piostream&, EdgeIndex&);
    friend const TypeDescription* get_type_description(EdgeIndex *);
    friend const string find_type_name(EdgeIndex *);
  };

  struct FaceIndex : public UnfinishedIndex
  {
    FaceIndex() : UnfinishedIndex() {}
    FaceIndex(unsigned i) : UnfinishedIndex(i) {}
    friend void Pio(Piostream&, FaceIndex&);
    friend const TypeDescription* get_type_description(FaceIndex *);
    friend const string find_type_name(FaceIndex *);
  };

  struct UnfinishedIter : public UnfinishedIndex
  {
    UnfinishedIter(const LatVolMesh *m, unsigned i)
      : UnfinishedIndex(i), mesh_(m) {}

    const UnfinishedIndex &operator *() { return *this; }

    bool operator ==(const UnfinishedIter &a) const
    {
      return i_ == a.i_ && mesh_ == a.mesh_;
    }

    bool operator !=(const UnfinishedIter &a) const
    {
      return !(*this == a);
    }

    const LatVolMesh *mesh_;
  };

  struct EdgeIter : public UnfinishedIter
  {
    EdgeIter() : UnfinishedIter(0, 0) {}
    EdgeIter(const LatVolMesh *m, unsigned i)
      : UnfinishedIter(m, i) {}

    const EdgeIndex &operator *() const { return (const EdgeIndex&)(*this); }

    EdgeIter &operator++() { return *this; }

  private:

    EdgeIter operator++(int)
    {
      EdgeIter result(*this);
      operator++();
      return result;
    }
  };

  struct FaceIter : public UnfinishedIter
  {
    FaceIter() : UnfinishedIter(0, 0) {}
    FaceIter(const LatVolMesh *m, unsigned i)
      : UnfinishedIter(m, i) {}

    const FaceIndex &operator *() const { return (const FaceIndex&)(*this); }

    FaceIter &operator++() { return *this; }

  private:

    FaceIter operator++(int)
    {
      FaceIter result(*this);
      operator++();
      return result;
    }
  };

  struct LatIndex
  {
  public:
    LatIndex() : i_(0), j_(0), k_(0) {}
    LatIndex(unsigned i, unsigned j, unsigned k) : i_(i), j_(j), k_(k) {}

    operator unsigned() const { return i_*j_*k_; }

    unsigned i_, j_, k_;
  };

  struct CellIndex : public LatIndex
  {
    CellIndex() : LatIndex() {}
    CellIndex(unsigned i, unsigned j, unsigned k) : LatIndex(i,j,k) {}
    friend void Pio(Piostream&, CellIndex&);
    friend const TypeDescription* get_type_description(CellIndex *);
    friend const string find_type_name(CellIndex *);
  };

  struct NodeIndex : public LatIndex
  {
    NodeIndex() : LatIndex() {}
    NodeIndex(unsigned i, unsigned j, unsigned k) : LatIndex(i,j,k) {}
    static string type_name(int i=-1) { ASSERT(i<1); return "LatVolMesh::NodeIndex"; }
    friend void Pio(Piostream&, NodeIndex&);
    friend const TypeDescription* get_type_description(NodeIndex *);
    friend const string find_type_name(NodeIndex *);
  };

  struct LatIter : public LatIndex
  {
    LatIter() : LatIndex(0, 0, 0), mesh_(0) {}
    LatIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : LatIndex(i, j, k), mesh_(m) {}

    const LatIndex &operator *() { return *this; }

    bool operator ==(const LatIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
    }

    bool operator !=(const LatIter &a) const
    {
      return !(*this == a);
    }

    const LatVolMesh *mesh_;
  };


  struct NodeIter : public LatIter
  {
    NodeIter() : LatIter() {}
    NodeIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : LatIter(m, i, j, k) {}

    const NodeIndex &operator *() const { return (const NodeIndex&)(*this); }

    NodeIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->min_x_+mesh_->nx_)	{
	i_ = mesh_->min_x_;
	j_++;
	if (j_ >=  mesh_->min_y_+mesh_->ny_) {
	  j_ = mesh_->min_y_;
	  k_++;
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
    CellIter(const LatVolMesh *m, unsigned i, unsigned j, unsigned k)
      : LatIter(m, i, j, k) {}

    const CellIndex &operator *() const { return (const CellIndex&)(*this); }

    CellIter &operator++()
    {
      i_++;
      if (i_ >= mesh_->min_x_+mesh_->nx_-1) {
	i_ = mesh_->min_x_;
	j_++;
	if (j_ >= mesh_->min_y_+mesh_->ny_-1) {
	  j_ = mesh_->min_y_;
	  k_++;
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

  //typedef LatIndex        under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex          index_type;
    typedef NodeIter           iterator;
    typedef NodeIndex          size_type;
    typedef vector<index_type> array_type;
  };			
  			
  struct Edge {		
    typedef EdgeIndex          index_type;
    typedef EdgeIter           iterator;
    typedef EdgeIndex          size_type;
    typedef vector<index_type> array_type;
  };			
			
  struct Face {		
    typedef FaceIndex          index_type;
    typedef FaceIter           iterator;
    typedef FaceIndex          size_type;
    typedef vector<index_type> array_type;
  };			
			
  struct Cell {		
    typedef CellIndex          index_type;
    typedef CellIter           iterator;
    typedef CellIndex          size_type;
    typedef vector<index_type> array_type;
  };

  typedef Cell Elem;

  typedef vector<double>      weight_array;

  friend class NodeIter;
  friend class CellIter;
  friend class EdgeIter;
  friend class FaceIter;

  LatVolMesh()
    : min_x_(0), min_y_(0), min_z_(0),
      nx_(1), ny_(1), nz_(1) {}
  LatVolMesh(unsigned x, unsigned y, unsigned z,
	     const Point &min, const Point &max);
  LatVolMesh(LatVolMesh* /* mh */,  // FIXME: Is this constructor broken?
	     unsigned mx, unsigned my, unsigned mz,
	     unsigned x, unsigned y, unsigned z)
    : min_x_(mx), min_y_(my), min_z_(mz),
      nx_(x), ny_(y), nz_(z) {}
  LatVolMesh(const LatVolMesh &copy)
    : min_x_(copy.min_x_), min_y_(copy.min_y_), min_z_(copy.min_z_),
      nx_(copy.get_nx()),ny_(copy.get_ny()),nz_(copy.get_nz()) {}
  virtual LatVolMesh *clone() { return new LatVolMesh(*this); }
  virtual ~LatVolMesh() {}

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

  //! get the mesh statistics
  unsigned get_nx() const { return nx_; }
  unsigned get_ny() const { return ny_; }
  unsigned get_nz() const { return nz_; }
  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);
  double get_volume(const Cell::index_type &) { return 0; }
  double get_area(const Face::index_type &) { return 0; }
  double get_element_size(const Elem::index_type &) {  return 0; }

  //! set the mesh statistics
  void set_min_x(unsigned x) {min_x_ = x; }
  void set_min_y(unsigned y) {min_y_ = y; }
  void set_min_z(unsigned z) {min_z_ = z; }
  void set_nx(unsigned x) { nx_ = x; }
  void set_ny(unsigned y) { ny_ = y; }
  void set_nz(unsigned z) { nz_ = z; }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const {}
  void get_nodes(Node::array_type &, Face::index_type) const {}
  void get_nodes(Node::array_type &, Cell::index_type) const;
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) { return 0; }

  //! return all cell_indecies that overlap the BBox in arr.
  void get_cells(Cell::array_type &arr, const BBox &box);

  //! similar to get_cells() with Face::index_type argument, but
  //  returns the "other" cell if it exists, not all that exist
  bool get_neighbor(Cell::index_type & /*neighbor*/, Cell::index_type /*from*/,
		    Face::index_type /*idx*/) const {
    ASSERTFAIL("LatVolMesh::get_neighbor not implemented.");
  }
  //! get the center point (in object space) of an element
  void get_center(Point &, Node::index_type) const;
  void get_center(Point &, Edge::index_type) const {}
  void get_center(Point &, Face::index_type) const {}
  void get_center(Point &, Cell::index_type) const;

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &);

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("LatVolMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &, Face::array_type &, vector<double> &) {ASSERTFAIL("LatVolMesh::get_weights for faces isn't supported");}
  void get_weights(const Point &p, Cell::array_type &l, vector<double> &w);

  void get_point(Point &p, Node::index_type i) const
  { get_center(p, i); }
  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  void get_random_point(Point &p, const Elem::index_type &ei, 
			int seed=0) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const string get_type_name(int n = -1) const { return type_name(n); }

  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

private:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned min_x_, min_y_, min_z_;
  //! the Node::index_type space extents of a LatVolMesh
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned nx_, ny_, nz_;

  //! the object space extents of a LatVolMesh
  //Point min_, max_;

  //! volume of each cell
  //double cell_volume_;

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

} // namespace SCIRun

#endif // SCI_project_LatVolMesh_h
