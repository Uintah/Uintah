/*
 *  TetVolMesh.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_TetVolMesh_h
#define SCI_project_TetVolMesh_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/LockingHandle.h>


namespace SCIRun {


class SCICORESHARE TetVolMesh : public Datatype
{
public:
  typedef int index_type;
  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex<index_type>       node_index;
  typedef NodeIterator<index_type>    node_iterator;

  typedef EdgeIndex<index_type>       edge_index;
  typedef EdgeIterator<index_type>    edge_iterator;

  typedef FaceIndex<index_type>       face_index;
  typedef FaceIterator<index_type>    face_iterator;

  typedef CellIndex<index_type>       cell_index;
  typedef CellIterator<index_type>    cell_iterator;

  typedef vector<node_index> node_array;
  typedef vector<edge_index> edge_array;
  typedef vector<face_index> face_array;

  TetVolMesh();
  TetVolMesh(const TetVolMesh &copy);
  //TetVolMesh(const MeshRG &lattice);
  virtual ~TetVolMesh();

  virtual BBox get_bounding_box() const;

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

  void get_nodes(node_array &array, edge_index idx) const;
  void get_nodes(node_array &array, face_index idx) const;
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &array, face_index idx) const;
  void get_edges(edge_array &array, cell_index idx) const;
  void get_faces(face_array &array, cell_index idx) const;
  void get_neighbor(cell_index &neighbor, face_index idx) const;
  void get_center(Point &result, node_index idx) const;
  void get_center(Point &result, edge_index idx) const;
  void get_center(Point &result, face_index idx) const;
  void get_center(Point &result, cell_index idx) const;

  void locate(node_index &node, const Point &p);
  //void locate_edge(edge_index &edge, const Point &p);
  //void locate_face(face_index &face, const Point &p);
  void locate(cell_index &cell, const Point &p);

  void unlocate(Point &result, const Point &p);

  void get_point(Point &result, node_index index) const;
  

  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  static  const string type_name(int);
  virtual const string get_type_name(int n) const { return type_name(n); }

private:

  bool inside4_p(int, const Point &p);


  Array1<Point>        points_;
  Array1<index_type>   cells_;
  Array1<index_type>   neighbors_;

};

// Handle type for TetVolMesh mesh.
typedef LockingHandle<TetVolMesh> TetVolMeshHandle;

} // namespace SCIRun


#endif // SCI_project_TetVolMesh_h
