
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

#include <SCICore/share/share.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Geometry/Point.h>
#include <iosfwd>
#include <vector>


namespace SCICore {
    namespace Containers {
	class clString;
    }
    namespace PersistentSpace {
	class Piostream;
    }
    namespace Tester {
	class RigorousTest;
    }
    
namespace Geometry {
using std::vector;
using SCICore::Containers::clString;
using SCICore::PersistentSpace::Piostream;
using SCICore::Tester::RigorousTest;

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
public:

  // GROUP:  Constructors:
  //////////
  // Constructor
  // parameters: a pointer to a C array of points that correspond
  //             to the vertices and an int specifying the length 
  //             of the array.  Points must be placed in counter
  //             clockwise order.
  Polygon(const Point *p, int nPoints);

  //////////
  // Constructor
  // takes a std:vector of Points that are a counter clockwise
  // ordering of the vertices.
  Polygon(const vector<Point>& v);
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
  // size method: How long is the vector?
  int size() const { return vertices.size(); }
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

} // End namespace Geometry
} // End namespace SCICore

#endif //ifndef Geometry_Polygon_h
