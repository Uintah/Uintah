
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
#include <SCICore/Math/MinMax.h>
#include <SCICore/Geometry/Vector.h>

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
        friend inline Vector operator*(const Vector&, const IntVector&);
        friend inline Vector operator*(const IntVector&, const Vector&);
    private:
	int d_x, d_y, d_z;
    };

    inline Vector operator*(const Vector& a, const IntVector& b) {
       return Vector(a.x()*b.x(), a.y()*b.y(), a.z()*b.z());
    }
    inline Vector operator*(const IntVector& a, const Vector& b) {
       return Vector(a.x()*b.x(), a.y()*b.y(), a.z()*b.z());
    }
    inline Vector operator/(const Vector& a, const IntVector& b) {
       return Vector(a.x()/b.x(), a.y()/b.y(), a.z()/b.z());
    }
    inline IntVector Min(const IntVector& a, const IntVector& b) {
       using SCICore::Math::Min;
       return IntVector(Min(a.x(), b.x()), Min(a.y(), b.y()), Min(a.z(), b.z()));
    }
    inline IntVector Max(const IntVector& a, const IntVector& b) {
       using SCICore::Math::Max;
       return IntVector(Max(a.x(), b.x()), Max(a.y(), b.y()), Max(a.z(), b.z()));
    }
  } // End namespace Geometry
} // End namespace SCICore

std::ostream& operator<<(std::ostream&, const SCICore::Geometry::IntVector&);

//
// $Log$
// Revision 1.4  2000/05/10 22:15:39  sparker
// Added min/max function for IntVector
//
// Revision 1.3  2000/04/27 23:18:14  sparker
// Added multiplication operators with Vector
//
// Revision 1.2  2000/04/13 06:48:38  sparker
// Implemented more of IntVector class
//
// Revision 1.1  2000/04/12 22:55:59  sparker
// Added IntVector (a vector of you-guess-what)
// Added explicit ctors from point to vector and vice-versa
//
//

#endif
