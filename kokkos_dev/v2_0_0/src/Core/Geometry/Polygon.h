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
 *  Polygon.h: ?
 *
 *  Written by:
 *   Author?
 *   Department of Computer Science
 *   University of Utah
 *   Date?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef Geometry_Polygon_h
#define Geometry_Polygon_h 1

#include <Core/share/share.h>
#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
    
using std::vector;

/**************************************

 CLASS
 Polygon

 Simple Polygon class

 GENERAL INFORMATION

 Polygon.h

 Kurt Zimmerman
 Department of Computer Science
 University of Utah

 Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

 Copyright (C) 2000 SCI Group

 KEYWORDS
 Polygon

 DESCRIPTION
 Simple Polygon class. 

 WARNING
 This class does not check to see if 
 indeed the points are coplanar.  To be added later.
****************************************/


class SCICORESHARE Polygon {
  vector<Point> vertices;
  vector<Point> texcoords;
public:

  // GROUP:  Constructors:
  //////////
  // Constructor
  // parameters: a pointer to a C array of points that correspond
  //             to the vertices and an int specifying the length 
  //             of the array.  Points must be placed in counter
  //             clockwise order.
  Polygon(const Point *p, const Point *t, int nPoints);

  //////////
  // Constructor
  // takes a std:vector of Points that are a counter clockwise
  // ordering of the vertices.
  Polygon(const vector<Point>& v, const vector<Point>& t);
  //////////
  // Copy Constructor
  Polygon(const Polygon&);
  
  // GROUP: Destructors
  //////////
  // Destructor
  ~Polygon(){};

  // GROUP: Boolean operators
  //////////
  // bool
  bool operator==(const Polygon&) const;
  //////////
  // bool
  bool operator!=(const Polygon&) const;

  Polygon& operator=(const Polygon&);

  // GROUP: info/access
  //////////
  // index method: return the ith vertex
  const Point& operator[](int i) const; 
  //////////
  // index method: return the ith vertex
  const Point& getVertex(int i) const; 
  //////////
  // index method: return the ith texture coordinate;
  const Point& getTexCoord(int i) const; 
  //////////
  // size method: How long is the vector?
  int size() const { return int(vertices.size()); }

  //////////
  // string method:
  string get_string() const;

private:
  //////////
  // Hidden Constructor
  Polygon(){};
};

SCICORESHARE std::ostream& operator<<(std::ostream& os, const Polygon& p);

} // End namespace SCIRun

#endif //ifndef Geometry_Polygon_h






