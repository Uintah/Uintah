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
 *  ImageMesh.h: Templated Mesh defined on a 2D Regular Grid
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

#ifndef SCI_project_ImageMesh_h
#define SCI_project_ImageMesh_h 1

#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/share/share.h>
#include <string>
#include <iostream>

namespace SCIRun {

using std::string;

class SCICORESHARE ImageMesh : public Mesh
{
public:
  struct ImageIndex;
  friend struct ImageIndex;

  struct ImageIndex
  {
  public:
    ImageIndex() : i_(0), j_(0), mesh_(0) {}

    ImageIndex(unsigned i, unsigned j) 
      : i_(i), j_(j), mesh_(0) {}

    ImageIndex(const ImageMesh *m, unsigned i, unsigned j) 
      : i_(i), j_(j), mesh_(m) {}

    operator unsigned() const { 
      if (mesh_ == 0) 
        return i_*j_; 
      else
        return i_ + j_ * mesh_->nx_;
    }

    unsigned i_, j_;

    const ImageMesh *mesh_;
  };

  struct IFaceIndex : public ImageIndex
  {
    IFaceIndex() : ImageIndex() {}
    IFaceIndex(unsigned i, unsigned j) 
      : ImageIndex(i, j) {}
    friend void Pio(Piostream&, IFaceIndex&);
    friend const TypeDescription* get_type_description(IFaceIndex *);
    friend const string find_type_name(IFaceIndex *);
  };

  struct INodeIndex : public ImageIndex
  {
    INodeIndex() : ImageIndex() {}
    INodeIndex(unsigned i, unsigned j) 
      : ImageIndex(i, j) {}
    friend void Pio(Piostream&, INodeIndex&);
    friend const TypeDescription* get_type_description(INodeIndex *);
    friend const string find_type_name(INodeIndex *);
  };

  struct ImageIter : public ImageIndex
  {
    ImageIter() : ImageIndex(0, 0) {}
    ImageIter(const ImageMesh *m, unsigned i, unsigned j)
      : ImageIndex(m, i, j) {}

    const ImageIndex &operator *() { return *this; }

    bool operator ==(const ImageIter &a) const
    {
      return i_ == a.i_ && j_ == a.j_ && mesh_ == a.mesh_;
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
      i_++;
      if (i_ >= mesh_->min_x_ + mesh_->nx_) {
	i_ = mesh_->min_x_;
	j_++;
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
      i_++;
      if (i_ >= mesh_->min_x_+mesh_->nx_-1) {
	i_ = mesh_->min_x_;
	j_++;
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

  //typedef ImageIndex      under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef INodeIndex                       index_type;
    typedef INodeIter                        iterator;
    typedef INodeIndex                       size_type;
    typedef vector<index_type>               array_type;
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
    typedef IFaceIndex                       size_type;
    typedef vector<index_type>               array_type;
  };			
			
  struct Cell {		
    typedef CellIndex<unsigned int>          index_type;
    typedef CellIterator<unsigned int>       iterator;
    typedef CellIndex<unsigned int>          size_type;
    typedef vector<index_type>               array_type;
  };

  typedef Face Elem;

  friend class INodeIter;
  friend class IFaceIter;

  ImageMesh()
    : min_x_(0), min_y_(0),
      nx_(1), ny_(1) {}
  ImageMesh(unsigned x, unsigned y, const Point &min, const Point &max);
  ImageMesh(ImageMesh* mh, unsigned int mx, unsigned int my,
	    unsigned int x, unsigned int y)
    : min_x_(mx), min_y_(my), nx_(x), ny_(y), transform_(mh->transform_) {}
  ImageMesh(const ImageMesh &copy)
    : min_x_(copy.min_x_), min_y_(copy.min_y_),
      nx_(copy.get_nx()), ny_(copy.get_ny()), transform_(copy.transform_) {}
  virtual ImageMesh *clone() { return new ImageMesh(*this); }
  virtual ~ImageMesh() {}

  //! get the mesh statistics
  unsigned get_nx() const { return nx_; }
  unsigned get_ny() const { return ny_; }
  unsigned get_nz() const { return 1; }
  Vector diagonal() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(Transform &t);

  //! set the mesh statistics
  void set_min_x(unsigned x) {min_x_ = x; }
  void set_min_y(unsigned y) {min_y_ = y; }
  void set_nx(unsigned x) { nx_ = x; }
  void set_ny(unsigned y) { ny_ = y; }

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

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! return all face_indecies that overlap the BBox in arr.
  void get_faces(Face::array_type &arr, const BBox &box);

  //! similar to get_faces() with Face::index_type argument, but
  //  returns the "other" face if it exists, not all that exist
  bool get_neighbor(Face::index_type & /*neighbor*/, Face::index_type /*from*/,
		    Edge::index_type /*idx*/) const {
    ASSERTFAIL("ImageMesh::get_neighbor not implemented.");
  }

  //! get the center point (in object space) of an element
  void get_center(Point &, Node::index_type) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, Face::index_type) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &);
  bool locate(Cell::index_type &, const Point &) const { return false; }

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) {ASSERTFAIL("ImageMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &p, Face::array_type &l, vector<double> &w);
  void get_weights(const Point &, Cell::array_type &, vector<double> &) {ASSERTFAIL("ImageMesh::get_weights for cells isn't supported");}

  void get_point(Point &p, Node::index_type i) const
  { get_center(p, i); }
  void get_normal(Vector &/* result */, Node::index_type /* index */) const
  { ASSERTFAIL("not implemented") }

  void get_random_point(Point &p, const Face::index_type &fi, 
			int seed=0) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

protected:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned min_x_, min_y_;
  //! the Node::index_type space extents of a ImageMesh
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned nx_, ny_;

  //! the object space extents of a ImageMesh
  Transform transform_;

  // returns a ImageMesh
  static Persistent *maker() { return new ImageMesh(); }
};

typedef LockingHandle<ImageMesh> ImageMeshHandle;

const TypeDescription* get_type_description(ImageMesh *);
const TypeDescription* get_type_description(ImageMesh::Node *);
const TypeDescription* get_type_description(ImageMesh::Edge *);
const TypeDescription* get_type_description(ImageMesh::Face *);
const TypeDescription* get_type_description(ImageMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_ImageMesh_h
