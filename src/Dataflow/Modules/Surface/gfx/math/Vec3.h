#ifndef GFXMATH_VEC3_INCLUDED // -*- C++ -*-
#define GFXMATH_VEC3_INCLUDED

/************************************************************************

  3D Vector class.

  $Id$

 ************************************************************************/

class Vec3 {
private:
    real elt[3];

protected:
    inline void copy(const Vec3& v);

public:
    //
    // Standard constructors
    //
    Vec3(real x=0, real y=0, real z=0) { elt[0]=x; elt[1]=y; elt[2]=z; }
#ifdef GFXMATH_VEC2_INCLUDED
    Vec3(const Vec2& v, real z) { elt[0]=v[0]; elt[1]=v[1]; elt[2]=z; }
#endif
    Vec3(const Vec3& v) { copy(v); }
    Vec3(const real *v) { elt[0]=v[0]; elt[1]=v[1]; elt[2]=v[2]; }

    //
    // Access methods
    //
#ifdef SAFETY
    real& operator()(int i)       { assert(i>=0 && i<3); return elt[i]; }
    real  operator()(int i) const { assert(i>=0 && i<3); return elt[i]; }
#else
    real& operator()(int i)       { return elt[i]; }
    real  operator()(int i) const { return elt[i]; }
#endif
    real& operator[](int i)       { return elt[i]; }
    real  operator[](int i) const { return elt[i]; }

    real *raw()             { return elt; }
    const real *raw() const { return elt; }

    //
    // Comparison operators
    //
    inline bool operator==(const Vec3& v) const;
    inline bool operator!=(const Vec3& v) const;

    //
    // Assignment and in-place arithmetic methods
    //
    inline void set(real x, real y, real z) { elt[0]=x; elt[1]=y; elt[2]=z; }
    inline Vec3& operator=(const Vec3& v);
    inline Vec3& operator+=(const Vec3& v);
    inline Vec3& operator-=(const Vec3& v);
    inline Vec3& operator*=(real s);
    inline Vec3& operator/=(real s);

    //
    // Binary arithmetic methods
    //
    inline Vec3 operator+(const Vec3& v) const;
    inline Vec3 operator-(const Vec3& v) const;
    inline Vec3 operator-() const;

    inline Vec3 operator*(real s) const;
    inline Vec3 operator/(real s) const;
    inline real operator*(const Vec3& v) const;
    inline Vec3 operator^(const Vec3& v) const;
};



////////////////////////////////////////////////////////////////////////
//
// Method definitions
//

inline void Vec3::copy(const Vec3& v)
{
    elt[0]=v.elt[0]; elt[1]=v.elt[1]; elt[2]=v.elt[2];
}

inline bool Vec3::operator==(const Vec3& v) const
{
    real dx=elt[X]-v[X],  dy=elt[Y]-v[Y],  dz=elt[Z]-v[Z];
    return (dx*dx + dy*dy + dz*dz) < FEQ_EPS2;
}

inline bool Vec3::operator!=(const Vec3& v) const
{
    real dx=elt[X]-v[X],  dy=elt[Y]-v[Y],  dz=elt[Z]-v[Z];
    return (dx*dx + dy*dy + dz*dz) > FEQ_EPS2;
}

inline Vec3& Vec3::operator=(const Vec3& v)
{
    copy(v);
    return *this;
}

inline Vec3& Vec3::operator+=(const Vec3& v)
{
    elt[0] += v[0];   elt[1] += v[1];   elt[2] += v[2];
    return *this;
}

inline Vec3& Vec3::operator-=(const Vec3& v)
{
    elt[0] -= v[0];   elt[1] -= v[1];   elt[2] -= v[2];
    return *this;
}

inline Vec3& Vec3::operator*=(real s)
{
    elt[0] *= s;   elt[1] *= s;   elt[2] *= s;
    return *this;
}

inline Vec3& Vec3::operator/=(real s)
{
    elt[0] /= s;   elt[1] /= s;   elt[2] /= s;
    return *this;
}


inline Vec3 Vec3::operator+(const Vec3& v) const
{
    return Vec3(elt[0]+v[0], elt[1]+v[1], elt[2]+v[2]);
}

inline Vec3 Vec3::operator-(const Vec3& v) const
{
    return Vec3(elt[0]-v[0], elt[1]-v[1], elt[2]-v[2]);
}

inline Vec3 Vec3::operator-() const
{
    return Vec3(-elt[0], -elt[1], -elt[2]);
}

inline Vec3 Vec3::operator*(real s) const
{
    return Vec3(elt[0]*s, elt[1]*s, elt[2]*s);
}

inline Vec3 Vec3::operator/(real s) const
{
    return Vec3(elt[0]/s, elt[1]/s, elt[2]/s);
}

inline real Vec3::operator*(const Vec3& v) const
{
    return elt[0]*v[0] + elt[1]*v[1] + elt[2]*v[2];
}

inline Vec3 Vec3::operator^(const Vec3& v) const
{
    Vec3 w( elt[1]*v[2] - v[1]*elt[2],
	   -elt[0]*v[2] + v[0]*elt[2],
	    elt[0]*v[1] - v[0]*elt[1] );
    return w;
}

// Make scalar multiplication commutative
inline Vec3 operator*(real s, const Vec3& v) { return v*s; }



////////////////////////////////////////////////////////////////////////
//
// Primitive function definitions
//

inline real norm(const Vec3& v)
{
    return sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2]);
}

inline real norm2(const Vec3& v)
{
    return v[0]*v[0] + v[1]*v[1] + v[2]*v[2];
}

inline real length(const Vec3& v) { return norm(v); }


inline real unitize(Vec3& v)
{
    real l=norm2(v);
    if( l!=1.0 && l!=0.0 )
    {
	l = sqrt(l);
	v /= l;
    }
    return l;
}



////////////////////////////////////////////////////////////////////////
//
// Misc. function definitions
//

inline ostream& operator<<(ostream& out, const Vec3& v)
{
    return out << "[" << v[0] << " " << v[1] << " " << v[2] << "]";
}

inline istream& operator>>(istream& in, Vec3& v)
{
    return in >> "[" >> v[0] >> v[1] >> v[2] >> "]";
}

#ifdef GFXGL_INCLUDED
inline void glV(const Vec3& v) { glVertex(v[X], v[Y], v[Z]); }
inline void glN(const Vec3& v) { glNormal(v[X], v[Y], v[Z]); }
inline void glC(const Vec3& v) { glColor(v[X], v[Y], v[Z]); }
#endif

//
// $Log$
// Revision 1.1  1999/07/27 16:58:04  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:17:15  dav
// added back PSECommon .h files
//
// Revision 1.1.1.1  1999/04/24 23:12:32  dav
// Import sources
//
//


#endif // GFXMATH_VEC3_INCLUDED
