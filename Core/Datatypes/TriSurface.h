
/*
 *  TriSurface.h: Triangulated Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Datatypes_TriSurface_h
#define SCI_Datatypes_TriSurface_h 1

#include <Core/Datatypes/Surface.h>
#include <Core/Containers/Array1.h>
#include <Core/Geometry/Point.h>

namespace SCIRun {

class SurfTree;

struct TSElement {
  int i1; 
  int i2; 
  int i3;
  inline TSElement(int i1 = -1, int i2 = -1, int i3 = -1)
    : i1(i1), i2(i2), i3(i3) {}

  inline TSElement(const TSElement& e):i1(e.i1), i2(e.i2), i3(e.i3) {}
};

void Pio (Piostream& stream, TSElement &data);

class SCICORESHARE TriSurface : public Surface
{
private:
  friend class SurfTree;

  int empty_index;

public:
  Array1<Point> points_;
  Array1<TSElement> faces_;

  enum NormalsType {
    PointType,	// one normal per point of the surface
    VertexType,	// one normal for each vertex of each element
    ElementType, 	// one normal for each element
    NrmlsNone
  };
  NormalsType normType;

  Array1<Vector> normals;


  TriSurface();
  TriSurface(const TriSurface& copy);
  virtual ~TriSurface();
  virtual Surface *clone();

  // Persistent representation.
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  // Virtual surface interface.
  virtual bool inside(const Point& p);
  virtual void construct_grid();
  virtual void construct_grid(int, int, int, const Point &, double);
  virtual void construct_hash(int, int, const Point &, double);
  virtual GeomObj* get_geom(const ColorMapHandle&);

  void buildNormals(NormalsType);

  SurfTree *toSurfTree();

  const Point &point(int i) { return points_[i]; }
  void point(int i, const Point &p) { points_[i] = p; }
  int point_count() { return points_.size(); }

  // these two were implemented for isosurfacing btwn two surfaces
  // (MorphMesher3d module/class)
  int cautious_add_triangle(const Point &p1, const Point &p2, const Point &p3,
			    int cw=0);
  int get_closest_vertex_id(const Point &p1, const Point &p2,
			    const Point &p3);


  int intersect(const Point& origin, const Vector& dir, double &d, int &v, int face);

  int add_triangle(int i1, int i2, int i3, int cw=0);


protected:

  // pass in allocated surfaces for conn and d_conn. NOTE: contents will be
  // overwritten
  void separate(int idx, TriSurface* conn, TriSurface* d_conn, int updateConnIndices=1, int updateDConnIndices=1);

  // NOTE: if elements have been added or removed from the surface
  // remove_empty_index() MUST be called before passing a TriSurface
  // to another module!  
  void remove_empty_index();
  int find_or_add(const Point &p);
  void remove_triangle(int i);
  double distance(const Point &p, Array1<int> &res, Point *pp=0);
  double distance(const Point &p, int i, int *type, Point *pp=0);
};

} // End namespace SCIRun


#endif /* SCI_Datatytpes_TriSurface_h */
