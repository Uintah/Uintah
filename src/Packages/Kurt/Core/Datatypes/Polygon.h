
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
#include <iosfwd>
#include <vector>

namespace Kurt {

using namespace SCIRun;

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
  clString string() const;

private:
  //////////
  // Hidden Constructor
  Polygon(){};
};

  SCICORESHARE std::ostream& operator<<(std::ostream& os, const Polygon& p);
  //SCICORESHARE std::istream& operator>>(std::istream& os, Polygon& p);

} // End namespace Kurt

#endif //ifndef Geometry_Polygon_h
