
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
	inline IntVector(const IntVector& copy) {
	  for (int indx = 0; indx < 3; indx ++)
	    d_value[indx] = copy.d_value[indx];
	}
	inline IntVector& operator=(const IntVector& copy) {
	  for (int indx = 0; indx < 3; indx ++)
	    d_value[indx] = copy.d_value[indx];
	  return *this;	
	}

	inline bool operator==(const IntVector& a) {
	   return d_value[0] == a.d_value[0] && d_value[1] == a.d_value[1] && d_value[2] == a.d_value[2];
	}

	inline bool operator!=(const IntVector& a) {
	   return d_value[0] != a.d_value[0] || d_value[1] != a.d_value[1] || d_value[2] != a.d_value[2];
	}

	inline IntVector(int x, int y, int z) {
	  d_value[0] = x;
	  d_value[1] = y;
	  d_value[2] = z;
	}

	inline IntVector operator*(const IntVector& v) const {
	    return IntVector(d_value[0]*v.d_value[0], d_value[1]*v.d_value[1],
			     d_value[2]*v.d_value[2]);
	}
	inline IntVector operator/(const IntVector& v) const {
	    return IntVector(d_value[0]/v.d_value[0], d_value[1]/v.d_value[1],
			     d_value[2]/v.d_value[2]);
	}
	inline IntVector operator+(const IntVector& v) const {
	    return IntVector(d_value[0]+v.d_value[0], d_value[1]+v.d_value[1], 
			     d_value[2]+v.d_value[2]);
	}
	inline IntVector operator-(const IntVector& v) const {
	    return IntVector(d_value[0]-v.d_value[0], d_value[1]-v.d_value[1], 
			     d_value[2]-v.d_value[2]);
	}

	// IntVector i(0)=i.x()
	//           i(1)=i.y()
	//           i(2)=i.z()
	//   --tan
	inline int operator()(int i) const {
	    return d_value[i];
	}

	inline int& operator()(int i) {
	    return d_value[i];
	}

	inline int x() const {
	    return d_value[0];
	}
	inline int y() const {
	    return d_value[1];
	}
	inline int z() const {
	    return d_value[2];
	}
	inline void x(int x) {
	    d_value[0]=x;
	}
	inline void y(int y) {
	    d_value[1]=y;
	}
	inline void z(int z) {
	    d_value[2]=z;
	}
	// get the array pointer
	inline int* get_pointer() {
	  return d_value;
	}
        friend inline Vector operator*(const Vector&, const IntVector&);
        friend inline Vector operator*(const IntVector&, const Vector&);
    private:
	int d_value[3];
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
// Revision 1.8  2000/07/10 20:19:13  tan
// For IntVector i, i(0)=i.x(),i(1)=i.y(),i(2)=i.z()
//
// Revision 1.7  2000/07/07 03:09:43  tan
// Added operator operator()(int i) to index IntVector element by integer.
//
// Revision 1.6  2000/06/20 20:39:53  rawat
// modified implementation of IntVector.h. Storing vector components as array
// of dim 3. Also added get_pointer for passing into fortran subroutines
//
// Revision 1.5  2000/05/20 08:05:31  sparker
// Added == and != operators
//
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
