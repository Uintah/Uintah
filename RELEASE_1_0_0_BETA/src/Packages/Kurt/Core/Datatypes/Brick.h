#ifndef BRICK_H
#define BRICK_H


#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Containers/Array3.h>
#include <string.h>
#include <vector>

#include "Polygon.h"
#include "GLVolRenState.h"

namespace Kurt {

using namespace SCIRun;
using std::vector;

/**************************************

CLASS
   Brick
   
   Brick Class for 3D Texturing 

GENERAL INFORMATION

   Brick.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Brick

DESCRIPTION
   Brick class for 3D Texturing.  Stores the texture associated with
   the Brick and the bricks location is space.  For a given view ray,
   min and max ray parameter, and parameter delta, it will create an
   ordered (back to fron) polygon list that can be rendered by a
   volume renderer.

  
WARNING
  
****************************************/

class Brick 
 {
friend class GLVolRenState;
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  Brick(const Point& min, const Point& max,
	int padx, int pady, int padz,int level,
	Array3<unsigned char>* tex);
  Brick();
  // GROUP: Destructors
  //////////
  // Destructor
  ~Brick();

  // GROUP: Access and Info
  //////////
  // Access one of the 8 vertices [0,7]
  const Point& operator[](int i) const { return corner[i]; }
  //////////
  // return a pointer to the texture 
  Array3<unsigned char>* texture(){ return tex; }
  //////////
  // obtain the bounding box of the Brick
  BBox&  bbox()const;

  int level() const { return lev; }
  // GROUP: Computation
  //////////
  // Compute Polygons orthogonal to the ray r.  A ray is composed
  // of an origin and a vector. tmin and tmax are
  // ray parameters in the equation,  point = r.origin + r.direction*tmin
  // or tmax.  Polys will be created from tmin to tmax and separated 
  // by dt.  ts is an array of parameters that correspond to the planes
  // perpendicular to r that run through the vertices of the Brick.
  void ComputePolys(Ray r, double  tmin, double  tmax,
		    double dt, double* ts, vector<Polygon*>& polys) const;
  void ComputePoly(Ray r, double t, Polygon*& p) const;
   unsigned int texName() const { return name;}
   unsigned int* texNameP(){ return &name; }

protected:

  typedef struct {
    double base;
    double step;
  } RayStep;

  Array3<unsigned char>* tex;

  Point corner[8];
  Ray edge[12];
  Ray texEdge[12];
  double aX, aY, aZ;
  int padx, pady, padz;

  int lev;

  unsigned int name;

  void OrderIntersects(Point *p, Point *t, Ray *r, Ray *te,
		       RayStep *dt, int n) const;

};
} // End namespace Kurt

#endif

