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

#ifndef SCI_FieldConverters_Surface_h
#define SCI_FieldConverters_Surface_h 1

#include <FieldConverters/share/share.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Containers/Handle.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/HashTable.h>
#include <Core/Geometry/Point.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Geometry/Grid.h>
#include <map>

namespace FieldConverters {

using namespace SCIRun;

class Surface;
class TriSurfFieldace;
class PointsSurface;
class SurfTree;

typedef LockingHandle<Surface> SurfaceHandle;
//typedef Handle<Node> NodeHandle;

class FieldConvertersSHARE Surface : public Datatype {
public:
  CrowdMonitor monitor;
  string name;

protected:
  enum Representation {
    TriSurfField,
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
  string boundary_expr;
  BoundaryType boundary_type;

public:

  Surface(const Surface& copy);
  virtual ~Surface();
  virtual Surface *clone() = 0;

  SurfTree* getSurfTree();
  TriSurfFieldace* getTriSurfFieldace();
  PointsSurface* getPointsSurface();

  void set_bc(const string& expr);

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

} // End namespace FieldConverters

#endif /* SCI_FieldConverters_Surface_h */
