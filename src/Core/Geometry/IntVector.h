
/*
 *  IntVector.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef Geometry_IntVector_h
#define Geometry_IntVector_h

#include <iosfwd>

namespace SCICore {
  namespace Geometry {
    class IntVector {
    public:
	inline IntVector() {
	}
	inline ~IntVector() {
	}
	inline IntVector(const IntVector& copy)
	    : d_x(copy.d_x), d_y(copy.d_y), d_z(copy.d_z) {
	}
	inline IntVector& operator=(const IntVector& copy) {
	    d_x=copy.d_x; d_y=copy.d_y; d_z=copy.d_z;
	    return *this;	
	}

	inline IntVector(int x, int y, int z)
	    : d_x(x), d_y(y), d_z(z)
	{
	}

	inline IntVector operator*(const IntVector& v) const {
	    return IntVector(d_x*v.d_x, d_y*v.d_y, d_z*v.d_z);
	}
	inline IntVector operator/(const IntVector& v) const {
	    return IntVector(d_x/v.d_x, d_y/v.d_y, d_z/v.d_z);
	}
	inline IntVector operator+(const IntVector& v) const {
	    return IntVector(d_x+v.d_x, d_y+v.d_y, d_z+v.d_z);
	}
	inline IntVector operator-(const IntVector& v) const {
	    return IntVector(d_x-v.d_x, d_y-v.d_y, d_z-v.d_z);
	}

	inline int x() const {
	    return d_x;
	}
	inline int y() const {
	    return d_y;
	}
	inline int z() const {
	    return d_z;
	}
	inline void x(int x) {
	    d_x=x;
	}
	inline void y(int y) {
	    d_y=y;
	}
	inline void z(int z) {
	    d_z=z;
	}
    private:
	int d_x, d_y, d_z;
    };

  } // End namespace Geometry
} // End namespace SCICore

std::ostream& operator<<(std::ostream&, const SCICore::Geometry::IntVector&);

//
// $Log$
// Revision 1.2  2000/04/13 06:48:38  sparker
// Implemented more of IntVector class
//
// Revision 1.1  2000/04/12 22:55:59  sparker
// Added IntVector (a vector of you-guess-what)
// Added explicit ctors from point to vector and vice-versa
//
//

#endif
