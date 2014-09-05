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
 *  StructCurveMesh.h: Templated Mesh defined on a 1D Structured Grid
 *
 *  Written by:
 *   Allen R. Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   November 2002
 *
 *  Copyright (C) 2002 SCI Group
 *
 */

/*
  A sturctured curve is a dataset with regular topology but with irregular geometry.
  The line defined may have any shape but can not be overlapping or self-intersecting.
  
  The topology of structured curve is represented using a 1D vector with
  the points being stored in an index based array. The ordering of the curve is
  implicity defined based based upon its indexing.

  For more information on datatypes see Schroeder, Martin, and Lorensen,
  "The Visualization Toolkit", Prentice Hall, 1998.
 */

#ifndef SCI_project_StructCurveMesh_h
#define SCI_project_StructCurveMesh_h 1

#include <Core/Datatypes/ScanlineMesh.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::string;

class StructCurveMesh : public ScanlineMesh
{
public:
  StructCurveMesh() {}
  StructCurveMesh(unsigned int n);
  StructCurveMesh(const StructCurveMesh &copy);
  virtual StructCurveMesh *clone() { return new StructCurveMesh(*this); }
  virtual ~StructCurveMesh() {}

  //! get the mesh statistics
  double get_cord_length() const;
  virtual BBox get_bounding_box() const;
  virtual void transform(const Transform &t);

  void set_dim(vector<unsigned int> dims) {
    ScanlineMesh::set_dim(dims);
    points_.resize(dims[0]);
  }

  bool get_dim(vector<unsigned int>&) const;

  //! get the child elements of the given index
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const {}
  void get_nodes(Node::array_type &, Cell::index_type) const {}
  void get_edges(Edge::array_type &, Face::index_type) const {}
  void get_edges(Edge::array_type &, Cell::index_type) const {}
  //void get_faces(Face::array_type &, Cell::index_type) const {}

  //! get the parent element(s) of the given index
  void get_edges(Edge::array_type &a, Node::index_type idx) const
  { a.push_back(Edge::index_type(idx));}
  bool get_faces(Face::array_type &, Node::index_type) const { return 0; }
  bool get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  bool get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  //! return all edge_indecies that overlap the BBox in arr.
  void get_edges(Edge::array_type &/*arr*/, const BBox &/*box*/) const
  { ASSERTFAIL("StructCurveMesh::get_edges for BBox is not implemented."); }

  bool get_neighbor(Edge::index_type &neighbor, Edge::index_type edge,
		    Node::index_type node) const
  {ASSERTFAIL("StructCurveMesh::get_neighbor for edges needs to be implemented"); }
  void get_neighbors(vector<Node::index_type> &array,
                     Node::index_type idx) const
  {ASSERTFAIL("StructCurveMesh::get_neighbor for nodes needs to be implemented"); }

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type /*idx*/) const { return 0.0; }
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }  
  double get_size(Face::index_type /*idx*/) const { return 0.0; }
  double get_size(Cell::index_type /*idx*/) const { return 0.0; }
  double get_length(Edge::index_type idx) const { return get_size(idx); }
  double get_area(Face::index_type idx) const { return get_size(idx); }
  double get_volume(Cell::index_type idx) const { return get_size(idx); }

  int get_valence(Node::index_type idx) const 
  { return (idx == (unsigned int) 0 ||
	    idx == (unsigned int) (points_.size()-1)) ? 1 : 2; }
  int get_valence(Edge::index_type /*idx*/) const { return 0; }
  int get_valence(Face::index_type /*idx*/) const { return 0; }
  int get_valence(Cell::index_type /*idx*/) const { return 0; }

  //! get the center point (in object space) of an element
  void get_center(Point &, const Node::index_type &) const;
  void get_center(Point &, const Edge::index_type &) const;
  void get_center(Point &, const Face::index_type &) const {}
  void get_center(Point &, const Cell::index_type &) const {}

  bool locate(Node::index_type &, const Point &) const;
  bool locate(Edge::index_type &, const Point &) const;
  bool locate(Face::index_type &, const Point &) const { return false; }
  bool locate(Cell::index_type &, const Point &) const { return false; }

  int get_weights(const Point &, Node::array_type &, double *w);
  int get_weights(const Point &, Edge::array_type &, double *w);
  int get_weights(const Point &, Face::array_type &, double */*w*/)
  {ASSERTFAIL("StructCurveMesh::get_weights for faces isn't supported"); }
  int get_weights(const Point &, Cell::array_type &, double * )
  {ASSERTFAIL("StructCurveMesh::get_weights for cells isn't supported"); }

  void get_point(Point &p, Node::index_type i) const { get_center(p,i); }
  void set_point(const Point &p, Node::index_type i) { points_[i] = p; }

  void get_random_point(Point &/*p*/, const Elem::index_type &/*ei*/, int /*seed=0*/) const
  { ASSERTFAIL("not implemented") }

  virtual bool is_editable() const { return false; }
    
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

private:

  //! the points
  Array1<Point> points_;

  // returns a StructCurveMesh
  static Persistent *maker() { return new StructCurveMesh(); }
}; // end class StructCurveMesh

typedef LockingHandle<StructCurveMesh> StructCurveMeshHandle;

const TypeDescription* get_type_description(StructCurveMesh *);
const TypeDescription* get_type_description(StructCurveMesh::Node *);
const TypeDescription* get_type_description(StructCurveMesh::Edge *);
const TypeDescription* get_type_description(StructCurveMesh::Face *);
const TypeDescription* get_type_description(StructCurveMesh::Cell *);

} // namespace SCIRun

#endif // SCI_project_StructCurveMesh_h
