#ifndef GFXMATH_VEC2_INCLUDED // -*- C++ -*-
#define GFXMATH_VEC2_INCLUDED

/************************************************************************

  2D Vector class.

  $Id$

 ************************************************************************/

class Vec2 {
private:
    real elt[2];

protected:
    inline void copy(const Vec2& v);

public:
    //
    // Standard constructors
    //
    Vec2(real x=0, real y=0) { elt[0]=x; elt[1]=y; }
    Vec2(const Vec2& v) { copy(v); }
    Vec2(const real *v) { elt[0]=v[0]; elt[1]=v[1]; }

    //
    // Access methods
    //
#ifdef SAFETY
    real& operator()(int i)       { assert(i>=0 && i<2); return elt[i]; }
    real  operator()(int i) const { assert(i>=0 && i<2); return elt[i]; }
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
    inline bool operator==(const Vec2& v) const;
    inline bool operator!=(const Vec2& v) const;

    //
    // Assignment and in-place arithmetic methods
    //
    inline void set(real x, real y) { elt[0]=x; elt[1]=y; }
    inline Vec2& operator=(const Vec2& v);
    inline Vec2& operator+=(const Vec2& v);
    inline Vec2& operator-=(const Vec2& v);
    inline Vec2& operator*=(real s);
    inline Vec2& operator/=(real s);

    //
    // Binary arithmetic methods
    //
    inline Vec2 operator+(const Vec2& v) const;
    inline Vec2 operator-(const Vec2& v) const;
    inline Vec2 operator-() const;

    inline Vec2 operator*(real s) const;
    inline Vec2 operator/(real s) const;
    inline real operator*(const Vec2& v) const;
};



////////////////////////////////////////////////////////////////////////
//
// Method definitions
//

inline void Vec2::copy(const Vec2& v)
{
    elt[0]=v.elt[0]; elt[1]=v.elt[1];
}

inline bool Vec2::operator==(const Vec2& v) const
{
    real dx=elt[X]-v[X],  dy=elt[Y]-v[Y];
    return (dx*dx + dy*dy) < FEQ_EPS2;
}

inline bool Vec2::operator!=(const Vec2& v) const
{
    real dx=elt[X]-v[X],  dy=elt[Y]-v[Y];
    return (dx*dx + dy*dy) > FEQ_EPS2;
}

inline Vec2& Vec2::operator=(const Vec2& v)
{
    copy(v);
    return *this;
}

inline Vec2& Vec2::operator+=(const Vec2& v)
{
    elt[0] += v[0];   elt[1] += v[1];
    return *this;
}

inline Vec2& Vec2::operator-=(const Vec2& v)
{
    elt[0] -= v[0];   elt[1] -= v[1];
    return *this;
}

inline Vec2& Vec2::operator*=(real s)
{
    elt[0] *= s;   elt[1] *= s;
    return *this;
}

inline Vec2& Vec2::operator/=(real s)
{
    elt[0] /= s;   elt[1] /= s;
    return *this;
}

inline Vec2 Vec2::operator+(const Vec2& v) const
{
    return Vec2(elt[0]+v[0], elt[1]+v[1]);
}

inline Vec2 Vec2::operator-(const Vec2& v) const
{
    return Vec2(elt[0]-v[0], elt[1]-v[1]);
}

inline Vec2 Vec2::operator-() const
{
    return Vec2(-elt[0], -elt[1]);
}

inline Vec2 Vec2::operator*(real s) const
{
    return Vec2(elt[0]*s, elt[1]*s);
}

inline Vec2 Vec2::operator/(real s) const
{
    return Vec2(elt[0]/s, elt[1]/s);
}

inline real Vec2::operator*(const Vec2& v) const
{
    return elt[0]*v[0] + elt[1]*v[1];
}

// Make scalar multiplication commutative
inline Vec2 operator*(real s, const Vec2& v) { return v*s; }



////////////////////////////////////////////////////////////////////////
//
// Primitive function definitions
//

inline real norm(const Vec2& v) { return sqrt(v[0]*v[0] + v[1]*v[1]); }
inline real norm2(const Vec2& v) { return v[0]*v[0] + v[1]*v[1]; }
inline real length(const Vec2& v) { return norm(v); }

inline real unitize(Vec2& v)
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

inline ostream& operator<<(ostream& out, const Vec2& v)
{
    return out << "[" << v[0] << " " << v[1] << "]";
}

inline istream& operator>>(istream& in, Vec2& v)
{
    return in >> "[" >> v[0] >> v[1] >> "]";
}

#ifdef GFXGL_INCLUDED
inline void glV(const Vec2& v) { glVertex(v[X], v[Y]); }
#endif

// GFXMATH_VEC2_INCLUDED
#endif
