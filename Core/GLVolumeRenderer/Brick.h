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

#ifndef BRICK_H
#define BRICK_H


#include <Core/Geometry/Point.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Polygon.h>
#include <Core/Containers/Array3.h>
#include <string.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <sgi_stl_warnings_on.h>

#include <Core/GLVolumeRenderer/GLVolRenState.h>

namespace SCIRun {

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
  virtual ~Brick();

  // GROUP: Access and Info
  //////////
  // Access one of the 8 vertices [0,7]
  const Point& operator[](int i) const { return corner[i]; }
  //////////
  // return a pointer to the texture 
  Array3<unsigned char>* texture(){ return tex; }
  //////////
  // obtain the bounding box of the Brick
  BBox bbox() const;

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

  double ax() const { return aX;}
  double ay() const { return aY;}
  double az() const { return aZ;}

  int padX() const { return padx; }
  int padY() const { return pady; }
  int padZ() const { return padz; }

  Point get_center(){ return corner[0] + 0.5*(corner[7] - corner[0]);}

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

} // End namespace SCIRun
#endif



