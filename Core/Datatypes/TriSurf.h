/*
 *  TriSurf.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_TriSurf_h
#define SCI_project_TriSurf_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>

#include <vector>

namespace SCIRun {

using std::vector;

class SCICORESHARE TriSurf : public Datatype
{
private:

  struct IntIter
  {
    int val;

    IntIter(int i) : val(i) {}

    operator int const &() const { return val; }
    int operator *() { return val; };
    int operator ++() { return val++; };
    int operator ++(int) { int tmp = val; val++; return tmp; }
  };

public:

  typedef int          node_index;
  typedef IntIter      node_iterator;

  typedef int          edge_index;
  typedef IntIter      edge_iterator;

  typedef int          face_index;
  typedef IntIter      face_iterator;

  //typedef int          cell_index;
  //typedef IntIter      cell_iterator;

  typedef vector<node_index> node_array;
  typedef vector<edge_index> edge_array;
  //typedef vector<face_index> face_array;

  TriSurf();
  TriSurf(const TriSurf &copy);
  virtual ~TriSurf();

  virtual BBox get_bounding_box() const;

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  //cell_iterator cell_begin() const;
  //cell_iterator cell_end() const;

  void get_nodes_from_edge(node_array &array, edge_index idx) const;
  void get_nodes_from_face(node_array &array, face_index idx) const;
  //void get_nodes_from_cell(node_array &array, cell_index idx) const;
  void get_edges_from_face(edge_array &array, face_index idx) const;
  //void get_edges_from_cell(edge_array &array, cell_index idx) const;
  //void get_faces_from_cell(face_array &array, cell_index idx) const;

  void get_neighbor_from_edge(face_index &neighbor, edge_index idx) const;

  void locate_node(node_index &node, const Point &p);
  //void locate_edge(edge_index &edge, const Point &p);
  //void locate_face(face_index &face, const Point &p);
  //void locate_cell(cell_index &cell, const Point &p);

  void unlocate(Point &result, const Point &p);

  void get_point(Point &result, node_index index) const;


  virtual void io(Piostream&);
  static PersistentTypeID type_id;


  // Extra functionality needed by this specific geometry.
  node_index add_find_point(const Point &p, double err = 1.0e-3);
  void add_triangle(node_index a, node_index b, node_index c,
		    bool cw_p = true);
  void add_triangle(const Point &p0, const Point &p1, const Point &p2,
		    bool cw_p = true);

private:

  bool inside4_p(int, const Point &p);


  vector<Point> points_;
  vector<int>   faces_;
  vector<int>   neighbors_;

};

} // namespace SCIRun


#endif // SCI_project_TriSurf_h
