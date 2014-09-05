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

#ifndef VOLUME_BRICK_H
#define VOLUME_BRICK_H



#include <string.h>
#include <vector>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/Ray.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geometry/Polygon.h>

#include <Packages/Volume/Core/Datatypes/BrickData.h>

namespace Volume {

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
public:

  // GROUP: Constructors:
  //////////
  // Constructor
  Brick(BrickData *data,
        int padx, int pady, int padz,
        const BBox* bbox, const BBox* tbox = 0);
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
  BrickData* data(){ return data_; }
  //////////
  // obtain the bounding box of the Brick
  BBox bbox() const;

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
  unsigned int texName() const { return name_;}
  unsigned int* texNameP(){ return &name_; }

  double ax() const { return ax_;}
  double ay() const { return ay_;}
  double az() const { return az_;}

  int padx() const { return padx_; }
  int pady() const { return pady_; }
  int padz() const { return padz_; }

  Point get_center(){ return corner[0] + 0.5*(corner[7] - corner[0]);}
  
  inline bool isQuantized() const {return quantized_;}
  inline bool storingAlpha() const {return storingAlpha_;}
protected:

  typedef struct {
    double base;
    double step;
  } RayStep;

  Point corner[8];
  Ray edge[12];
  Ray texEdge[12];
  double ax_, ay_, az_;
  int padx_, pady_, padz_;

  unsigned int name_;
  
  BrickData *data_;

  void OrderIntersects(Point *p, Point *t,
                       Ray *r, Ray *tx,
                       RayStep *dt, int n) const;

  bool quantized_;
  bool storingAlpha_;
};

} // End namespace Volume

#endif
