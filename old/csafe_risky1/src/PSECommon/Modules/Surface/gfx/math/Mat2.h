#ifndef GFXMATH_MAT2_INCLUDED // -*- C++ -*-
#define GFXMATH_MAT2_INCLUDED

#include <gfx/math/Vec2.h>

class Mat2
{
private:
    Vec2 row[2];

protected:

    inline void copy(const Mat2& m);
    inline Vec2 col(int i) const {return Vec2(row[0][i],row[1][i]);}

public:
    // Standard matrices
    static Mat2 identity;
    static Mat2 zero;
    static Mat2 unit;

    // Standard constructors
    Mat2() { copy(zero); }
    Mat2(const Vec2& r0,const Vec2& r1)
    { row[0]=r0; row[1]=r1; }
    Mat2(const Mat2& m) { copy(m); }

    // Access methods
    // M(i, j) == row i;col j
    real& operator()(int i, int j)       { return row[i][j]; }
    real  operator()(int i, int j) const { return row[i][j]; }

    // Assignment methods
    inline Mat2& operator=(const Mat2& m) { copy(m); return *this; }
    inline Mat2& operator+=(const Mat2& m);
    inline Mat2& operator-=(const Mat2& m);

    inline Mat2& operator*=(real s);
    inline Mat2& operator/=(real s);


    // Arithmetic methods
    inline Mat2 operator+(const Mat2& m) const;
    inline Mat2 operator-(const Mat2& m) const;
    inline Mat2 operator-() const;

    inline Mat2 operator*(real s) const;
    inline Mat2 operator/(real s) const;
    Mat2 operator*(const Mat2& m) const;

    inline Vec2 operator*(const Vec2& v) const; // [x y]

    // Matrix operations
    real det();
    Mat2 transpose();
    real inverse(Mat2&);


    // Input/Output methods
    friend ostream& operator<<(ostream&, const Mat2&);
    friend istream& operator>>(istream&, Mat2&);
};



inline void Mat2::copy(const Mat2& m)
{
    row[0] = m.row[0]; row[1] = m.row[1];
}

inline Mat2& Mat2::operator+=(const Mat2& m)
{
    row[0] += m.row[0]; row[1] += m.row[1];
    return *this;
}

inline Mat2& Mat2::operator-=(const Mat2& m)
{
    row[0] -= m.row[0]; row[1] -= m.row[1];
    return *this;
}

inline Mat2& Mat2::operator*=(real s)
{
    row[0] *= s; row[1] *= s;
    return *this;
}

inline Mat2& Mat2::operator/=(real s)
{
    row[0] /= s; row[1] /= s;
    return *this;
}

inline Mat2 Mat2::operator+(const Mat2& m) const
{
    return Mat2(row[0]+m.row[0],
		row[1]+m.row[1]);
}

inline Mat2 Mat2::operator-(const Mat2& m) const
{
    return Mat2(row[0]-m.row[0],
		row[1]-m.row[1]);
}

inline Mat2 Mat2::operator-() const
{
    return Mat2(-row[0], -row[1]);
}

inline Mat2 Mat2::operator*(real s) const
{
    return Mat2(row[0]*s, row[1]*s);
}

inline Mat2 Mat2::operator/(real s) const
{
    return Mat2(row[0]/s, row[1]/s);
}

inline Vec2 Mat2::operator*(const Vec2& v) const
{
    return Vec2(row[0]*v, row[1]*v);
}

inline ostream& operator<<(ostream& out, const Mat2& M)
{
    return out << M.row[0] << endl  << M.row[1];
}

inline istream& operator>>(istream& in, Mat2& M)
{
    return in >> M.row[0] >> M.row[1];
}


// GFXMATH_MAT2_INCLUDED
#endif
