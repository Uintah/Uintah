/*
 *  MeshTet.h: Templated Meshs defined on a 3D Regular Grid
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

#ifndef SCI_project_MeshTet_h
#define SCI_project_MeshTet_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Geometry/BBox.h>
#include <Core/Containers/Array1.h>


namespace SCIRun {



class SCICORESHARE MeshTet : public Datatype
{
private:

  struct IntIter
  {
    int val;

    IntIter(int i) : val(i) {}

    operator int const &() const { return val; }
    int operator *() { return val; };
    int operator ++() { return val++; };
  };

public:

  typedef int          node_index;
  typedef IntIter      node_iterator;

  typedef void *       edge_index;
  typedef void *       edge_iterator;

  typedef void *       face_index;
  typedef void *       face_iterator;

  typedef int          cell_index;
  typedef IntIter      cell_iterator;


  MeshTet();
  MeshTet(const MeshTet &copy);
  virtual ~MeshTet();

  virtual BBox get_bounding_box() const;

  node_iterator node_begin() const;
  node_iterator node_end() const;
  edge_iterator edge_begin() const;
  edge_iterator edge_end() const;
  face_iterator face_begin() const;
  face_iterator face_end() const;
  cell_iterator cell_begin() const;
  cell_iterator cell_end() const;

  void locate_node(node_index &node, const Point &p);
  void locate_edge(edge_index &edge, const Point &p);
  void locate_face(face_index &face, const Point &p);
  void locate_cell(cell_index &cell, const Point &p);


  void unlocate(Point &result, const Point &p);

  void get_point(Point &result, node_index index) const;


  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:

  bool inside4_p(int, const Point &p);


  Array1<Point> points_;
  Array1<int>   tets_;
  Array1<int>   neighbors_;

};

} // namespace SCIRun


#endif // SCI_project_MeshTet_h
