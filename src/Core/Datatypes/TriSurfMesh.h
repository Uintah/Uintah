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
 *  TriSurfMesh.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_TriSurfMesh_h
#define SCI_project_TriSurfMesh_h 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/MusilRNG.h>

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <set>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using std::vector;

class SCICORESHARE TriSurfMesh : public Mesh
{
public:

  typedef int                           under_type;

  //! Index and Iterator types required for Mesh Concept.
  struct Node {
    typedef NodeIndex<under_type>       index_type;
    typedef NodeIterator<under_type>    iterator;
    typedef NodeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Edge {
    typedef EdgeIndex<under_type>       index_type;
    typedef EdgeIterator<under_type>    iterator;
    typedef EdgeIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Face {
    typedef FaceIndex<under_type>       index_type;
    typedef FaceIterator<under_type>    iterator;
    typedef FaceIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  struct Cell {
    typedef CellIndex<under_type>       index_type;
    typedef CellIterator<under_type>    iterator;
    typedef CellIndex<under_type>       size_type;
    typedef vector<index_type>          array_type;
  };

  typedef Face Elem;

  TriSurfMesh();
  TriSurfMesh(const TriSurfMesh &copy);
  virtual TriSurfMesh *clone() { return new TriSurfMesh(*this); }
  virtual ~TriSurfMesh();

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

  //! Get the child elements of the given index.
  void get_nodes(Node::array_type &, Edge::index_type) const;
  void get_nodes(Node::array_type &, Face::index_type) const;
  void get_nodes(Node::array_type &, Cell::index_type) const;
  void get_edges(Edge::array_type &, Face::index_type) const;
  void get_edges(Edge::array_type &, Cell::index_type) const;
  void get_faces(Face::array_type &, Cell::index_type) const;

  //! get the parent element(s) of the given index
  unsigned get_edges(Edge::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Node::index_type) const { return 0; }
  unsigned get_faces(Face::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Node::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Edge::index_type) const { return 0; }
  unsigned get_cells(Cell::array_type &, Face::index_type) const { return 0; }

  bool get_neighbor(Face::index_type &neighbor, Face::index_type face,
		    Edge::index_type edge) const;
  void get_neighbors(Node::array_type &array, Node::index_type idx) const;

  //! Get the size of an elemnt (length, area, volume)
  double get_size(Node::index_type idx) const { return 0.0; };
  double get_size(Edge::index_type idx) const 
  {
    Node::array_type arr;
    get_nodes(arr, idx);
    Point p0, p1;
    get_center(p0, arr[0]);
    get_center(p1, arr[1]);
    return (p1.asVector() - p0.asVector()).length();
  }
  double get_size(Face::index_type idx) const
  {
    Node::array_type ra;
    get_nodes(ra,idx);
    Point p0,p1,p2;
    get_point(p0,ra[0]);
    get_point(p1,ra[1]);
    get_point(p2,ra[2]);
    return (Cross(p0-p1,p2-p0)).length()*0.5;
  }
  double get_size(Cell::index_type idx) const { return 0.0; };
  double get_length(Edge::index_type idx) const { return get_size(idx); };
  double get_area(Face::index_type idx) const { return get_size(idx); };
  double get_volume(Cell::index_type idx) const { return get_size(idx); };

  int get_valence(Node::index_type idx) const
  {
    Node::array_type nodes;
    get_neighbors(nodes, idx);
    return nodes.size();
  }
  int get_valence(Edge::index_type idx) const { return 0; }
  int get_valence(Face::index_type idx) const { return 0; }
  int get_valence(Cell::index_type idx) const { return 0; }


  void get_center(Point &p, Node::index_type i) const { get_point(p, i); }
  void get_center(Point &p, Edge::index_type i) const;
  void get_center(Point &p, Face::index_type i) const;
  void get_center(Point &, Cell::index_type) const {}

  bool locate(Node::index_type &loc, const Point &p) const;
  bool locate(Edge::index_type &loc, const Point &p) const;
  bool locate(Face::index_type &loc, const Point &p) const;
  bool locate(Cell::index_type &loc, const Point &p) const;

  void get_weights(const Point &p, Node::array_type &l, vector<double> &w);
  void get_weights(const Point &, Edge::array_type &, vector<double> &) 
    {ASSERTFAIL("TriSurfMesh::get_weights for edges isn't supported");}
  void get_weights(const Point &p, Face::array_type &l, vector<double> &w);
  void get_weights(const Point &, Cell::array_type &, vector<double> &) 
    {ASSERTFAIL("TriSurfMesh::get_weights for cells isn't supported");}

  void get_point(Point &result, Node::index_type index) const
    { result = points_[index]; }
  void get_normal(Vector &result, Node::index_type index) const
    { result = normals_[index]; }
  void set_point(const Point &point, Node::index_type index)
    { points_[index] = point; }


  void get_random_point(Point &, Face::index_type, int seed=0) const;

  //! the double return val is the area of the triangle.
  double get_gradient_basis(Face::index_type fi, Vector& g0, Vector& g1,
			    Vector& g2);

  //! function to test if at least one of face's nodes are in supplied range
  inline bool test_nodes_range(Face::index_type fi, int sn, int en){
    if (faces_[fi*3]>=sn && faces_[fi*3]<en
	|| faces_[fi*3+1]>=sn && faces_[fi*3+1]<en
	|| faces_[fi*3+2]>=sn && faces_[fi*3+2]<en)
      return true;
    else
      return false;
  }

  virtual bool		synchronize(unsigned int);

  virtual bool has_normals() const { return true; }

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int n = -1);
  virtual const TypeDescription *get_type_description() const;

  // Extra functionality needed by this specific geometry.

  Node::index_type add_find_point(const Point &p, double err = 1.0e-3);
  void add_triangle(Node::index_type, Node::index_type, Node::index_type);
  void add_triangle(const Point &p0, const Point &p1, const Point &p2);
  Elem::index_type add_elem(Node::array_type a);
  virtual bool is_editable() const { return true; }
  virtual int dimensionality() const { return 2; }

  Node::index_type add_point(const Point &p);

  //! Subdivision Methods
  bool			insert_node(const Point &p);
  void			insert_node(Face::index_type face, const Point &p);
  void			bisect_element(const Face::index_type);


  const Point &point(Node::index_type i) { return points_[i]; }

private:
  void			compute_normals();
  void			compute_node_neighbors();  
  void			compute_edges();
  void			compute_edge_neighbors(double err = 1.0e-8);

  bool inside3_p(int, const Point &p) const;

  int next(int i) { return ((i%3)==2) ? (i-2) : (i+1); }
  int prev(int i) { return ((i%3)==0) ? (i+2) : (i-1); }

  vector<Point>		points_;
  vector<int>		edges_;
  vector<int>		faces_;
  vector<int>		edge_neighbors_;
  vector<Vector>	normals_;   //! normalized per node normal.
  vector<set<int> >	node_neighbors_;
  unsigned int		synchronized_;

};

typedef LockingHandle<TriSurfMesh> TriSurfMeshHandle;

const TypeDescription* get_type_description(TriSurfMesh *);
const TypeDescription* get_type_description(TriSurfMesh::Node *);
const TypeDescription* get_type_description(TriSurfMesh::Edge *);
const TypeDescription* get_type_description(TriSurfMesh::Face *);
const TypeDescription* get_type_description(TriSurfMesh::Cell *);

} // namespace SCIRun


#endif // SCI_project_TriSurfMesh_h
