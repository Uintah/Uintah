
/*
 *  Surface.h: The Surface Data type
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_project_Surface_h
#define SCI_project_Surface_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/ColorMap.h>
#include <Core/Containers/Handle.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/HashTable.h>
#include <Core/Containers/String.h>
#include <Core/Geometry/Point.h>
#include <Core/Thread/CrowdMonitor.h>

#include <map.h>

namespace SCIRun {

class Grid;
class GeomObj;
class  Surface;
struct Node;
class TriSurface;
class PointsSurface;
class SurfTree;

typedef LockingHandle<Surface> SurfaceHandle;
//typedef Handle<Node> NodeHandle;

class SCICORESHARE Surface : public Datatype {
public:
  CrowdMonitor monitor;
  clString name;

protected:
  enum Representation {
    TriSurf,
    PointsSurf,
    Unused,
    STree,
    RepOther
  };
  Surface(Representation, int closed);

  Representation rep;

  int hash_x;
  int hash_y;
  Point hash_min;
  double resolution;
  int closed;
  Grid *grid;

  //HashTable<int, int> *pntHash;
  typedef map<int, int> MapIntInt;
  MapIntInt* pntHash;

  // Boundary conditions...
  enum BoundaryType {
    DirichletExpression,
    DirichletData,
    BdryNone
  };
  clString boundary_expr;
  BoundaryType boundary_type;

public:

  Surface(const Surface& copy);
  virtual ~Surface();
  virtual Surface *clone() = 0;

  SurfTree* getSurfTree();
  TriSurface* getTriSurface();
  PointsSurface* getPointsSurface();

  void set_bc(const clString& expr);

  virtual int inside(const Point& p)=0;
  virtual void construct_grid(int, int, int, const Point &, double)=0;
  virtual void construct_grid()=0;
  virtual void destroy_grid();
  virtual void destroy_hash();
  //virtual void get_surfnodes(Array1<NodeHandle>&)=0;
  //virtual void set_surfnodes(const Array1<NodeHandle>&)=0;
  virtual GeomObj* get_obj(const ColorMapHandle&)=0;

  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
};

} // End namespace SCIRun

#endif /* SCI_project_Surface_h */
