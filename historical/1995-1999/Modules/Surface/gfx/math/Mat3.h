#ifndef GFXMATH_MAT3_INCLUDED // -*- C++ -*-
#define GFXMATH_MAT3_INCLUDED

#include <gfx/math/Vec3.h>

class Mat3
{
private:
    Vec3 row[3];

protected:

    inline void copy(const Mat3& m);
    inline Vec3 col(int i) const {return Vec3(row[0][i],row[1][i],row[2][i]);}

public:
    // Standard matrices
    static Mat3 identity;
    static Mat3 zero;
    static Mat3 unit;
    static Mat3 diag(const Vec3& v);    // diagonal elements from v
    static Mat3 extend(const Vec3& v);  // each row becomes v

    // Standard constructors
    Mat3() { copy(zero); }
    Mat3(const Vec3& r0,const Vec3& r1,const Vec3& r2)
    { row[0]=r0; row[1]=r1; row[2]=r2; }
    Mat3(const Mat3& m) { copy(m); }

    // Access methods
    // M(i, j) == row i;col j
    real& operator()(int i, int j)       { return row[i][j]; }
    real  operator()(int i, int j) const { return row[i][j]; }

    // Assignment methods
    inline Mat3& operator=(const Mat3& m) { copy(m); return *this; }
    inline Mat3& operator+=(const Mat3& m);
    inline Mat3& operator-=(const Mat3& m);

    inline Mat3& operator*=(real s);
    inline Mat3& operator/=(real s);


    // Arithmetic methods
    inline Mat3 operator+(const Mat3& m) const;
    inline Mat3 operator-(const Mat3& m) const;
    inline Mat3 operator-() const;

    inline Mat3 operator*(real s) const;
    inline Mat3 operator/(real s) const;
    Mat3 operator*(const Mat3& m) const;

    inline Vec3 operator*(const Vec3& v) const; // [x y z]

    // Matrix operations
    real det();
    Mat3 transpose();
    Mat3 adjoint();
    real inverse(Mat3&);


    // Input/Output methods
    friend ostream& operator<<(ostream&, const Mat3&);
    friend istream& operator>>(istream&, Mat3&);
};



inline void Mat3::copy(const Mat3& m)
{
    row[0] = m.row[0]; row[1] = m.row[1]; row[2] = m.row[2];
}

inline Mat3& Mat3::operator+=(const Mat3& m)
{
    row[0] += m.row[0]; row[1] += m.row[1]; row[2] += m.row[2];
    return *this;
}

inline Mat3& Mat3::operator-=(const Mat3& m)
{
    row[0] -= m.row[0]; row[1] -= m.row[1]; row[2] -= m.row[2];
    return *this;
}

inline Mat3& Mat3::operator*=(real s)
{
    row[0] *= s; row[1] *= s; row[2] *= s;
    return *this;
}

inline Mat3& Mat3::operator/=(real s)
{
    row[0] /= s; row[1] /= s; row[2] /= s;
    return *this;
}

inline Mat3 Mat3::operator+(const Mat3& m) const
{
    return Mat3(row[0]+m.row[0],
		row[1]+m.row[1],
		row[2]+m.row[2]);
}

inline Mat3 Mat3::operator-(const Mat3& m) const
{
    return Mat3(row[0]-m.row[0],
		row[1]-m.row[1],
		row[2]-m.row[2]);
}

inline Mat3 Mat3::operator-() const
{
    return Mat3(-row[0], -row[1], -row[2]);
}

inline Mat3 Mat3::operator*(real s) const
{
    return Mat3(row[0]*s, row[1]*s, row[2]*s);
}

inline Mat3 Mat3::operator/(real s) const
{
    return Mat3(row[0]/s, row[1]/s, row[2]/s);
}

inline Vec3 Mat3::operator*(const Vec3& v) const
{
    return Vec3(row[0]*v, row[1]*v, row[2]*v);
}

inline ostream& operator<<(ostream& out, const Mat3& M)
{
    return out << M.row[0] << endl << M.row[1] << endl << M.row[2];
}

inline istream& operator>>(istream& in, Mat3& M)
{
    return in >> M.row[0] >> M.row[1] >> M.row[2];
}

extern bool jacobi(const Mat3& m, Vec3& vals, Vec3 vecs[3]);


// GFXMATH_MAT3_INCLUDED
#endif
