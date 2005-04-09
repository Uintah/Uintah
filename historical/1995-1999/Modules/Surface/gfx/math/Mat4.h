#ifndef GFXMATH_MAT4_INCLUDED // -*- C++ -*-
#define GFXMATH_MAT4_INCLUDED

/************************************************************************

  4x4 Matrix class

  $Id$

 ************************************************************************/

#include <gfx/math/Vec3.h>
#include <gfx/math/Vec4.h>

class Mat4
{
private:
    Vec4 row[4];

protected:

    inline void copy(const Mat4& m);
    inline Vec4 col(int i) const
        { return Vec4(row[0][i],row[1][i],row[2][i],row[3][i]); }

public:
    // Standard matrices
    static Mat4 identity;
    static Mat4 zero;
    static Mat4 unit;

    static Mat4 trans(real,real,real);
    static Mat4 scale(real,real,real);
    static Mat4 xrot(real); //
    static Mat4 yrot(real); // Arguments are in radians
    static Mat4 zrot(real); //


    // Standard constructors
    Mat4() { copy(zero); }
    Mat4(const Vec4& r0,const Vec4& r1,const Vec4& r2,const Vec4& r3)
    { row[0]=r0; row[1]=r1; row[2]=r2; row[3]=r3; }
    Mat4(const Mat4& m) { copy(m); }

    // Access methods
    // M(i, j) == row i;col j
    real& operator()(int i, int j)       { return row[i][j]; }
    real  operator()(int i, int j) const { return row[i][j]; }
    const Vec4& operator[](int i) const { return row[i]; }

    // Comparison methods
    inline int operator==(const Mat4&);

    // Assignment methods
    inline Mat4& operator=(const Mat4& m) { copy(m); return *this; }
    inline Mat4& operator+=(const Mat4& m);
    inline Mat4& operator-=(const Mat4& m);

    inline Mat4& operator*=(real s);
    inline Mat4& operator/=(real s);


    // Arithmetic methods
    inline Mat4 operator+(const Mat4& m) const;
    inline Mat4 operator-(const Mat4& m) const;
    inline Mat4 operator-() const;

    inline Mat4 operator*(real s) const;
    inline Mat4 operator/(real s) const;
    Mat4 operator*(const Mat4& m) const;

    inline Vec4 operator*(const Vec4& v) const; // [x y z w]
    inline Vec3 operator*(const Vec3& v) const; // [x y z w]

    // Matrix operations
    real det() const;
    Mat4 transpose() const;
    Mat4 adjoint() const;
    real inverse(Mat4&) const;
    real cramerInverse(Mat4&) const;

    // Input/Output methods
    friend ostream& operator<<(ostream&, const Mat4&);
    friend istream& operator>>(istream&, Mat4&);
};



inline void Mat4::copy(const Mat4& m)
{
    row[0] = m.row[0]; row[1] = m.row[1];
    row[2] = m.row[2]; row[3] = m.row[3];
}

inline int Mat4::operator==(const Mat4& m)
{
    return row[0]==m.row[0] &&
	   row[1]==m.row[1] &&
	   row[2]==m.row[2] &&
	   row[3]==m.row[3] ;
}

inline Mat4& Mat4::operator+=(const Mat4& m)
{
    row[0] += m.row[0]; row[1] += m.row[1];
    row[2] += m.row[2]; row[3] += m.row[3];
    return *this;
}

inline Mat4& Mat4::operator-=(const Mat4& m)
{
    row[0] -= m.row[0]; row[1] -= m.row[1];
    row[2] -= m.row[2]; row[3] -= m.row[3];
    return *this;
}

inline Mat4& Mat4::operator*=(real s)
{
    row[0] *= s; row[1] *= s; row[2] *= s; row[3] *= s;
    return *this;
}

inline Mat4& Mat4::operator/=(real s)
{
    row[0] /= s; row[1] /= s; row[2] /= s; row[3] /= s;
    return *this;
}

inline Mat4 Mat4::operator+(const Mat4& m) const
{
    return Mat4(row[0]+m.row[0],
		row[1]+m.row[1],
		row[2]+m.row[2],
		row[3]+m.row[3]);
}

inline Mat4 Mat4::operator-(const Mat4& m) const
{
    return Mat4(row[0]-m.row[0],
		row[1]-m.row[1],
		row[2]-m.row[2],
		row[3]-m.row[3]);
}

inline Mat4 Mat4::operator-() const
{
    return Mat4(-row[0], -row[1], -row[2], -row[3]);
}

inline Mat4 Mat4::operator*(real s) const
{
    return Mat4(row[0]*s, row[1]*s, row[2]*s, row[3]*s);
}

inline Mat4 Mat4::operator/(real s) const
{
    return Mat4(row[0]/s, row[1]/s, row[2]/s, row[3]/s);
}

inline Vec4 Mat4::operator*(const Vec4& v) const
{
    return Vec4(row[0]*v, row[1]*v, row[2]*v, row[3]*v);
}

//
// Transform a homogeneous 3-vector and reproject into normal 3-space
//
inline Vec3 Mat4::operator*(const Vec3& v) const
{
    Vec4 u=Vec4(v,1);
    real w=row[3]*u;

    if(w==0.0)
	return Vec3(row[0]*u, row[1]*u, row[2]*u);
    else
	return Vec3(row[0]*u/w, row[1]*u/w, row[2]*u/w);
}

inline ostream& operator<<(ostream& out, const Mat4& M)
{
    return out<<M.row[0]<<endl<<M.row[1]<<endl<<M.row[2]<<endl<<M.row[3];
}

inline istream& operator>>(istream& in, Mat4& M)
{
    return in >> M.row[0] >> M.row[1] >> M.row[2] >> M.row[3];
}

extern bool cholesky(Mat4&, Vec4&);
extern bool jacobi(const Mat4& m, Vec4& vals, Vec4 vecs[4]);


// GFXMATH_MAT4_INCLUDED
#endif
