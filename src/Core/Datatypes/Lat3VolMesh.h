/*
 *  Lat3VolMesh.h: Templated Mesh defined on a 3D Regular Grid
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

#ifndef SCI_project_Lat3VolMesh_h
#define SCI_project_Lat3VolMesh_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/FieldIterator.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Octree.h>


namespace SCIRun {

class SCICORESHARE Lat3VolMesh : public Datatype
{
public:

  typedef unsigned index_type;

  //! Index and Iterator types required for Mesh Concept.
  typedef NodeIndex<index_type>       node_index;
  typedef NodeIterator<index_type>    node_iterator;
 
  typedef EdgeIndex<index_type>       edge_index;
  typedef EdgeIterator<index_type>    edge_iterator;
 
  typedef FaceIndex<index_type>       face_index;
  typedef FaceIterator<index_type>    face_iterator;
 
  typedef CellIndex<index_type>       cell_index;
  typedef CellIterator<index_type>    cell_iterator;

  //! Storage types for the arguments passed to the 
  //  get_*() functions.  For rg meshes, these all have
  //  known maximum sizes, so we use them.
  typedef node_index      node_array[8];
  typedef edge_index      edge_array[12];
  typedef face_index      face_array[6];
  typedef cell_index      cell_array[8];

  Lat3VolMesh(int x, int y, int z, Point &min, Point &max);
  Lat3VolMesh(const Lat3VolMesh &);
  virtual ~Lat3VolMesh();

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

  //! get the mesh statistics
  int get_nx() { return nx_; }
  int get_ny() { return ny_; }
  int get_nz() { return nz_; }
  Point get_min() { return min_; }
  Point get_max() { return max_; }
  virtual BBox get_bounding_box() const;

  //! get the child elements of the given index
  void get_nodes(node_array &array, edge_index idx) const;
  void get_nodes(node_array &array, face_index idx) const;
  void get_nodes(node_array &array, cell_index idx) const;
  void get_edges(edge_array &array, face_index idx) const;
  void get_edges(edge_array &array, cell_index idx) const;
  void get_faces(face_array &array, cell_index idx) const;

  //! get the parent element(s) of the given index
  int get_edges(edge_array &array, node_index idx) const;
  int get_faces(face_array &array, node_index idx) const;
  int get_faces(face_array &array, edge_index idx) const;
  int get_cells(cell_array &array, node_index idx) const;
  int get_cells(cell_array &array, edge_index idx) const;
  int get_cells(cell_array &array, face_index idx) const;

  //! similar to get_cells() with face_index argument, but
  //  returns the "other" cell if it exists, not all that exist
  void get_neighbor(cell_index &neighbor, face_index idx) const;

  //! get the center point (in object space) of an element
  void get_center(Point &result, node_index idx) const;
  void get_center(Point &result, edge_index idx) const;
  void get_center(Point &result, face_index idx) const;
  void get_center(Point &result, cell_index idx) const;

  void locate_node(node_index &node, const Point &p) const;
  void locate_edge(edge_index &edge, const Point &p, double[2]) const;
  void locate_face(face_index &face, const Point &p, double[4]) const;
  void locate_cell(cell_index &cell, const Point &p, double[8]) const;


  void unlocate(Point &result, const Point &p) const;

  void get_point(Point &result, const node_index &index) const;

  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:

  //! the object space extents of a Lat3VolMesh
  Point min_, max_;

  //! the node_index extents of a Lat3VolMesh (min=0, max=n)
  int nx_, ny_, nz_;

  // returns a Lat3VolMesh
  static Persistent *maker();
};



} // namespace SCIRun



#if 0
//old iterator and index stuff
struct IPoint
{
  IPoint() {}
  IPoint(int i, int j, int k) : i_(i), j_(j), k_(k) {}

  int i_, j_, k_;
};


struct NCIter : public IPoint
{
  NCIter(const Lat3VolMesh *m, int i, int j, int k) : IPoint(i, j, k), mesh_(m) {}

  const IPoint &operator *() { return *this; }

  bool operator ==(const NCIter &a)
  {
    return i_ == a.i_ && j_ == a.j_ && k_ == a.k_ && mesh_ == a.mesh_;
  }

  bool operator !=(const NCIter &a)
  {
    return !(*this == a);
  }

  const Lat3VolMesh *mesh_;
};


struct NodeIter : public NCIter
{
  NodeIter(const Lat3VolMesh *m, int i, int j, int k) : NCIter(m, i, j, k) {}

  NodeIter &operator++()
  {
    i_++;
    if (i_ > mesh_->nx_)
    {
      i_ = 0;
      j_++;
      if (j_ > mesh_->ny_)
      {
	j_ = 0;
	k_++;
      }
    }
    return *this;
  }

  NodeIter operator++(int)
  {
    NodeIter result(*this);
    operator++();
    return result;
  }
};


struct CellIter : public NCIter
{
  CellIter(const Lat3VolMesh *m, int i, int j, int k) : NCIter(m, i, j, k) {}

  CellIter &operator++()
  {
    i_++;
    if (i_ >= mesh_->nx_)
    {
      i_ = 0;
      j_++;
      if (j_ >= mesh_->ny_)
      {
	j_ = 0;
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
#endif



#endif // SCI_project_Lat3VolMesh_h





