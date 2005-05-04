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

#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/StackVector.h>

namespace SCIRun {

using std::string;

class ImageMesh;

struct ImageMeshImageIndex
{
public:
  ImageMeshImageIndex() : i_(0), j_(0), mesh_(0) {}

  ImageMeshImageIndex(const ImageMesh *m, unsigned i, unsigned j) 
    : i_(i), j_(j), mesh_(m) {}

  operator unsigned() const;

  unsigned i_, j_;
  const ImageMesh *mesh_;
};

struct ImageMeshIFaceIndex : public ImageMeshImageIndex
{
  ImageMeshIFaceIndex() : ImageMeshImageIndex() {}
  ImageMeshIFaceIndex(const ImageMesh *m, unsigned i, unsigned j) 
    : ImageMeshImageIndex(m, i, j) {}

  operator unsigned() const;

  friend void Pio(Piostream&, ImageMeshIFaceIndex&);
  friend const TypeDescription* get_type_description(ImageMeshIFaceIndex *);
  friend const string find_type_name(ImageMeshIFaceIndex *);
};

struct ImageMeshINodeIndex : public ImageMeshImageIndex
{
  ImageMeshINodeIndex() : ImageMeshImageIndex() {}
  ImageMeshINodeIndex(const ImageMesh *m, unsigned i, unsigned j) 
    : ImageMeshImageIndex(m, i, j) {}
  friend void Pio(Piostream&, ImageMeshINodeIndex&);
  friend const TypeDescription* get_type_description(ImageMeshINodeIndex *);
  friend const string find_type_name(ImageMeshINodeIndex *);
};

struct ImageIter : public ImageMeshImageIndex
{
  ImageIter() : ImageMeshImageIndex() {}
  ImageIter(const ImageMesh *m, unsigned i, unsigned j)
    : ImageMeshImageIndex(m, i, j) {}

  const ImageMeshImageIndex &operator *() { return *this; }

  bool operator ==(const ImageIter &a) const
  {
    return i_ == a.i_ && j_ == a.j_ && mesh_ == a.mesh_;
  }

  bool operator !=(const ImageIter &a) const
  {
    return !(*this == a);
  }
};

struct ImageMeshINodeIter : public ImageIter
{
  ImageMeshINodeIter() : ImageIter() {}
  ImageMeshINodeIter(const ImageMesh *m, unsigned i, unsigned j)
    : ImageIter(m, i, j) {}

  const ImageMeshINodeIndex &operator *() const { return (const ImageMeshINodeIndex&)(*this); }

  ImageMeshINodeIter &operator++();

private:

  ImageMeshINodeIter operator++(int)
  {
    ImageMeshINodeIter result(*this);
    operator++();
    return result;
  }
};


struct ImageMeshIFaceIter : public ImageIter
{
  ImageMeshIFaceIter() : ImageIter() {}
  ImageMeshIFaceIter(const ImageMesh *m, unsigned i, unsigned j)
    : ImageIter(m, i, j) {}

  const ImageMeshIFaceIndex &operator *() const { return (const ImageMeshIFaceIndex&)(*this); }

  ImageMeshIFaceIter &operator++();

private:

  ImageMeshIFaceIter operator++(int)
  {
    ImageMeshIFaceIter result(*this);
    operator++();
    return result;
  }
};

struct ImageMeshImageSize
{ 
public:
  ImageMeshImageSize() : i_(0), j_(0) {}
  ImageMeshImageSize(unsigned i, unsigned j) : i_(i), j_(j) {}

  operator unsigned() const { return i_*j_; }

  unsigned i_, j_;
};

struct ImageMeshINodeSize : public ImageMeshImageSize
{
  ImageMeshINodeSize() : ImageMeshImageSize() {}
  ImageMeshINodeSize(unsigned i, unsigned j) : ImageMeshImageSize(i,j) {}
};


struct ImageMeshIFaceSize : public ImageMeshImageSize
{
  ImageMeshIFaceSize() : ImageMeshImageSize() {}
  ImageMeshIFaceSize(unsigned i, unsigned j) : ImageMeshImageSize(i,j) {}
};


//! Index and Iterator types required for Mesh Concept.
struct ImageMeshNode {
  typedef ImageMeshINodeIndex                       index_type;
  typedef ImageMeshINodeIter                        iterator;
  typedef ImageMeshINodeSize                        size_type;
  typedef StackVector<index_type, 4>       array_type;
};			
			
struct ImageMeshEdge {		
  typedef EdgeIndex<unsigned int>          index_type;
  typedef EdgeIterator<unsigned int>       iterator;
  typedef EdgeIndex<unsigned int>          size_type;
  typedef vector<index_type>               array_type;
};			
			
struct ImageMeshFace {		
  typedef ImageMeshIFaceIndex                       index_type;
  typedef ImageMeshIFaceIter                        iterator;
  typedef ImageMeshIFaceSize                        size_type;
  typedef vector<index_type>               array_type;
};			
			
struct ImageMeshCell {		
  typedef CellIndex<unsigned int>          index_type;
  typedef CellIterator<unsigned int>       iterator;
  typedef CellIndex<unsigned int>          size_type;
  typedef vector<index_type>               array_type;
};


class ImageMesh : public Mesh
{
public:

  friend struct ImageMeshImageIndex;
  friend struct ImageMeshINodeIter;
  friend struct ImageMeshIFaceIter;
  friend struct ImageMeshIFaceIndex;

  // Backwards compatability with interp fields
  typedef ImageMeshINodeIndex INodeIndex;
  typedef ImageMeshIFaceIndex IFaceIndex;

  typedef ImageMeshNode Node;
  typedef ImageMeshEdge Edge;
  typedef ImageMeshFace Face;
  typedef ImageMeshCell Cell;
  typedef Face Elem;

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
  void to_index(Face::index_type &index, unsigned int i);
  void to_index(Cell::index_type &index, unsigned int i) { index = i; }

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const;
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

  //! Get the size of an elemnt (length, area, volume)
  double get_size(const Node::index_type &/*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(const Face::index_type &idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(const Node::index_type &idx) const
  {
    return ((idx.i_ == 0 || idx.i_ == ni_ - 1 ? 1 : 2) +
	    (idx.j_ == 0 || idx.j_ == nj_ - 1 ? 1 : 2));
  }   
  int get_valence(Edge::index_type idx) const;
  int get_valence(const Face::index_type &/*idx*/) const { return 0; }
  int get_valence(Cell::index_type /*idx*/) const { return 0; }


  bool get_neighbor(Face::index_type &neighbor,
		    Face::index_type from,
		    Edge::index_type idx) const;
  
  void get_neighbors(Face::array_type &array, Face::index_type idx) const;
    
  virtual bool has_normals() const { return true; }
  void get_normal(Vector &n, const Node::index_type &) const
  { n = normal_; }

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, Edge::index_type) const;
  void get_center(Point &, const Face::index_type &) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &, const Point &);
  bool locate(Edge::index_type &, const Point &) const { return false; }
  bool locate(Face::index_type &, const Point &);
  bool locate(Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &p, Node::array_type &l, double *w);
  int get_weights(const Point & , Edge::array_type & , double * )
  {ASSERTFAIL("ImageMesh::get_weights for edges isn't supported"); }
  int get_weights(const Point &p, Face::array_type &l, double *w);
  int get_weights(const Point & , Cell::array_type & , double * )
  {ASSERTFAIL("ImageMesh::get_weights for cells isn't supported"); }

  void get_point(Point &p, const Node::index_type &i) const
  { get_center(p, i); }

  void get_random_point(Point &, const Face::index_type &, int seed=0) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Unsafe due to non-constness of unproject.
  Transform &get_transform() { return transform_; }
  Transform &set_transform(const Transform &trans) 
  { transform_ = trans; return transform_; }

  virtual int dimensionality() const { return 2; }

protected:

  //! the min_Node::index_type ( incase this is a subLattice )
  unsigned min_i_, min_j_;
  //! the Node::index_type space extents of a ImageMesh
  //! (min=min_Node::index_type, max=min+extents-1)
  unsigned ni_, nj_;

  //! the object space extents of a ImageMesh
  Transform transform_;

  Vector normal_;

  // returns a ImageMesh
  static Persistent *maker() { return new ImageMesh(); }
};

typedef LockingHandle<ImageMesh> ImageMeshHandle;

const TypeDescription* get_type_description(ImageMesh *);
const TypeDescription* get_type_description(ImageMesh::Node *);
const TypeDescription* get_type_description(ImageMesh::Edge *);
const TypeDescription* get_type_description(ImageMesh::Face *);
const TypeDescription* get_type_description(ImageMesh::Cell *);
std::ostream& operator<<(std::ostream& os, const ImageMeshImageIndex& n);
std::ostream& operator<<(std::ostream& os, const ImageMeshImageSize& s);

} // namespace SCIRun

#endif // SCI_project_ImageMesh_h
