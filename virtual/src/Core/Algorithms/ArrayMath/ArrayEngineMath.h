/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

#ifndef CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINEMATH_H
#define CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINEMATH_H 1

#ifdef HAVE_TEEM
#include <teem/ten.h>
#endif

#include <math.h>

// Somehow someone had the great idea to specify some of the
// next functions as #define statements on certain platforms
// here we remove these definitions

#ifdef isnan 
#undef isnan
#endif

#ifdef isfinite 
#undef isfinite
#endif

#ifdef isinf 
#undef isinf
#endif

namespace DataArrayMath {

typedef double Scalar;

class Vector {
  public:
    inline Vector();
    inline Vector(const Vector& vec);
    inline Vector(double x, double y, double z);

    inline double operator==(const Vector&) const;
    inline double operator!=(const Vector&) const;

    // operators defined for this class
    inline Vector& operator=(const Vector&);
    inline Vector& operator=(const double&);
    inline double& operator[](int idx);
    inline double operator[](int idx) const;
    inline double& operator()(int idx);
    inline double operator()(int idx) const;
    inline double& operator[](std::string s);
    inline double operator[](std::string s) const;
    inline double& operator()(std::string s);
    inline double operator()(std::string s) const;
    
    inline double x() const;
    inline double y() const;
    inline double z() const;    
    
    inline Vector operator*(const double) const;
    inline Vector operator*(const Vector&) const;
    inline Vector& operator*=(const double);
    inline Vector& operator*=(const Vector&);
    inline Vector operator/(const double) const;
    inline Vector operator/(const Vector&) const;
    inline Vector& operator/=(const double);
    inline Vector& operator/=(const Vector&);
    inline Vector operator+(const double) const;
    inline Vector operator+(const Vector&) const;
    inline Vector& operator+=(const double);
    inline Vector& operator+=(const Vector&);
    inline Vector operator-(const double) const;    
    inline Vector operator-(const Vector&) const;
    inline Vector& operator-=(const double);
    inline Vector& operator-=(const Vector&);
    inline Vector operator^(const double) const;    
    inline Vector operator^(const Vector&) const;
    inline Vector& operator^=(const double);
    inline Vector& operator^=(const Vector&);
    inline Vector operator-() const;
    
    // Operaters that result in scalar values
    inline double length() const;
    inline double norm() const;
    inline double maxnorm() const;
    inline double max() const;
    inline double min() const;

    // Operators that result in vector values
    inline Vector clone() const;
    inline Vector sin() const;    
    inline Vector cos() const;
    inline Vector sinh() const;    
    inline Vector cosh() const;
    inline Vector tan() const;

    inline Vector asin() const;    
    inline Vector acos() const;
    inline Vector asinh() const;    
    inline Vector acosh() const;
    inline Vector atan() const;
    
    inline Vector log() const;
    inline Vector log10() const;
    inline Vector log2() const;
    
    inline Vector abs() const;
    inline Vector cbrt() const;
    inline Vector sqrt() const;
    
    inline Vector pow(double p) const;
    inline Vector exp() const;
        
    inline Vector ceil() const;
    inline Vector floor() const;
    inline Vector round() const;
    inline Vector normalize() const;

    inline Vector findnormal1() const;
    inline Vector findnormal2() const;

    inline double isnan() const;
    inline double isinf() const;
    inline double isfinite() const;

    inline double dot(const Vector&) const;
    inline Vector cross(const Vector&) const;

    inline Vector operator&(const double) const;
    inline Vector& operator&=(const double);

    inline double  boolean() const;
    inline double* getdataptr();
    
  private:
    double d_[3];
};


class Tensor {
  public:
    inline Tensor();
    inline Tensor(const Tensor& vec);
    inline Tensor(double xx, double xy, double xz, double yy, double yz, double zz);
    inline Tensor(double* ptr, int size);
    inline Tensor(Vector e1, Vector e2, double s1, double s2, double s3);
    inline Tensor(Vector e1, Vector e2, double s1, double s2);
    inline Tensor(Vector e1, Vector e2, Vector e3, double s1, double s2, double s3);
    inline Tensor(Vector e1, double s1, double s2);
    inline Tensor(double s1);

    inline Vector eigvec1();
    inline Vector eigvec2();
    inline Vector eigvec3();
    inline Vector eigvals();

    inline double eigval1();
    inline double eigval2();
    inline double eigval3();
    
    inline double trace();
    inline double det();
    inline double B();
    inline double S();
    inline double Q();

    inline double quality();    
    inline double frobenius();
    inline double frobenius2();
    inline double fracanisotropy();

    inline Vector vec1() const;
    inline Vector vec2() const;
    inline Vector vec3() const;

    inline double xx() const;
    inline double xy() const;
    inline double xz() const;
    inline double yy() const;
    inline double yz() const;
    inline double zz() const;

   // operators defined for this class
   
    inline double operator<(const Tensor&);
    inline double operator<=(const Tensor&);
    inline double operator>(const Tensor&);
    inline double operator>=(const Tensor&);
    
    inline double operator<(const double&);
    inline double operator<=(const double&);
    inline double operator>(const double&);
    inline double operator>=(const double&);
  
     
    inline Tensor& operator=(const Tensor&);
    inline double& operator[](int idx);
    inline double operator[](int idx) const;
    inline double& operator()(int idx);
    inline double operator()(int idx) const;

    inline double& operator[](std::string s);
    inline double operator[](std::string s) const;
    inline double& operator()(std::string s);
    inline double operator()(std::string s) const;

    inline Vector operator*(const Vector&) const;
 
    inline Tensor operator*(const double) const;
    inline Tensor operator*(const Tensor&) const;
    inline Tensor& operator*=(const double);
    inline Tensor& operator*=(const Tensor&);

    inline Tensor operator/(const double) const;
    inline Tensor operator/(const Tensor&) const;
    inline Tensor& operator/=(const double);
    inline Tensor& operator/=(const Tensor&);
    
    inline Tensor operator+(const double) const;
    inline Tensor operator+(const Tensor&) const;
    inline Tensor& operator+=(const double);
    inline Tensor& operator+=(const Tensor&);
    inline Tensor operator-(const double) const;    
    inline Tensor operator-(const Tensor&) const;
    inline Tensor& operator-=(const double);
    inline Tensor& operator-=(const Tensor&);
    inline Tensor operator^(const double) const;    
    inline Tensor operator^(const Tensor&) const;
    inline Tensor& operator^=(const double);
    inline Tensor& operator^=(const Tensor&);
    inline Tensor operator-() const;

    inline double norm() const;
    inline double max() const;
    inline double min() const;

    // Operators that result in vector values
    inline Tensor clone() const;
    inline Tensor sin() const;    
    inline Tensor cos() const;
    inline Tensor sinh() const;    
    inline Tensor cosh() const;
    inline Tensor tan() const;

    inline Tensor asin() const;    
    inline Tensor acos() const;
    inline Tensor asinh() const;    
    inline Tensor acosh() const;
    inline Tensor atan() const;
    
    inline Tensor log() const;
    inline Tensor log10() const;
    inline Tensor log2() const;
    
    inline Tensor abs() const;
    inline Tensor cbrt() const;
    inline Tensor sqrt() const;
    
    inline Tensor pow(double) const;
    inline Tensor exp() const;
        
    inline Tensor ceil() const;
    inline Tensor floor() const;
    inline Tensor round() const;
    inline Tensor normalize() const;

    inline Tensor inv();

    inline double isnan() const;
    inline double isinf() const;
    inline double isfinite() const;

    inline Tensor operator&(const double) const;
    inline Tensor& operator&=(const double);

    inline double boolean() const;
    inline double* getdataptr();



  private:
    double d_[6];
    
    bool has_eigs_;
    inline void compute_eigs();
    inline void compute_tensor();

    Vector eigvec1_,eigvec2_,eigvec3_;
    double eigval1_,eigval2_,eigval3_;

};

#ifdef _WIN32
typedef __int64 int64;  
typedef unsigned __int64 uint64;  
#define INT64_VAL(x) x##i64
#define UINT64_VAL(x) x##ui64
#else
typedef long long int64;
typedef unsigned long long uint64;
#define INT64_VAL(x) x##ll
#define UINT64_VAL(x) x##ull
#endif

inline double getnan()
{
  double* dptr;
  uint64 n = UINT64_VAL(0x7fffffffffffffff);
  dptr = reinterpret_cast<double *>(&n);
  return(*dptr);
}

inline double isnan(double d)
{
  const uint64 n1 = UINT64_VAL(0x7ff0000000000000);
  const uint64 n2 = UINT64_VAL(0x000fffffffffffff);
  uint64* n = reinterpret_cast<uint64 *>(&d);
  return((((*n)&n1==n1)&&((*n)&n2))?1.0:0.0);
}

inline double isfinite(double d)
{
  const uint64 n1 = UINT64_VAL(0x7ff0000000000000);
  uint64* n = reinterpret_cast<uint64 *>(&d);
  return(((*n)&n1 != n1)?1.0:0.0);
}

inline double isinf(double d)
{
  const uint64 n1 = UINT64_VAL(0x7ff0000000000000);
  const uint64 n2 = UINT64_VAL(0xfff0000000000000);
  uint64* n = reinterpret_cast<uint64 *>(&d);
  return(((*n)== n1)||((*n == n2))?1.0:0.0);
}

inline double sign(const double d)
{
  return(d<0.0?-1.0:(d>0.0?1.0:0.0));
}

const double nan = getnan();
const double NaN = getnan();
const double pi = 3.14159265358979323846;
const double e = 2.71828182845904523536;

// SCALAR OPERATIONS

inline double inv(double s)
{
  return(1.0/s);
}

inline double boolean(double s)
{
  if (s) return(1.0);
  return(0.0);
} 

inline double abs(double d)
{
  return(d<0?-d:d);
}


inline double norm(double d)
{
  return(d<0?-d:d);
}

inline double round(double d)
{
  return(static_cast<double>(static_cast<int>((d+0.5))));
}

inline double floor(double d)
{
  return(::floor(d));
}

inline double ceil(double d)
{
  return(::ceil(d));
}

inline double exp(double d)
{
  return(::exp(d));
}

inline double pow(double d,double p)
{
  return(::pow(d,p));
}

inline double sqrt(double d)
{
  return(::sqrt(d));
}

inline double log(double d)
{
  return(::log(d));
}

inline double ln(double d)
{
  return(::log(d));
}

inline double log2(double d)
{
  return(log(d)/log(2));
}

inline double log10(double d)
{
  return(::log10(d));
}

inline double cbrt(double d)
{
  return(::pow(d,1.0/3.0));
}

inline double sin(double d)
{
  return(::sin(d));
}

inline double cos(double d)
{
  return(::cos(d));
}

inline double tan(double d)
{
  return(::tan(d));
}

inline double sinh(double d)
{
  return(::sinh(d));
}

inline double cosh(double d)
{
  return(::cosh(d));
}

inline double asin(double d)
{
  return(::asin(d));
}

inline double acos(double d)
{
  return(::acos(d));
}

inline double atan(double d)
{
  return(::atan(d));
}

inline double asinh(double d)
{
#ifndef _WIN32
  return(::asinh(d));
#else
  // asinh and acosh aren't ANSI C, and thus aren't defined on windows
  // according to 'man asinh', this is equivalent to 
  //   sgn(X) * log(abs(X) + sqrt(1+X*X))
  // trinary operator: return -1,0,1 depending on whether less than, equal to, or greater than 0
  return (d==0?0:(d>0?1:-1)) * log(abs(d) + sqrt(1+d*d));
#endif
}

inline double acosh(double d)
{
#ifndef _WIN32
  return(::acosh(d));
#else
  // asinh and acosh aren't ANSI C, and thus aren't defined on windows
  // according to 'man acosh' this is equivalent to
  //   log(X + sqrt(X*X-1))
  return log(d + sqrt(d*d-1));
#endif
}


inline double operator&&(double a, Vector b)
{
  return((a&&b.boolean())?1.0:0.0);
}

inline double operator&&(double a, Tensor b)
{
  return((a&&b.boolean())?1.0:0.0);
}

inline double operator&&(Vector b,double a)
{
  return((a&&b.boolean())?1.0:0.0);
}

inline double operator&&(Tensor b,double a)
{
  return((a&&b.boolean())?1.0:0.0);
}

inline double operator&&(Vector a,Vector b)
{
  return((a.boolean()&&b.boolean())?1.0:0.0);
}

inline double operator&&(Tensor a,Tensor b)
{
  return((a.boolean()&&b.boolean())?1.0:0.0);
}

inline double operator&&(Vector a,Tensor b)
{
  return((a.boolean()&&b.boolean())?1.0:0.0);
}

inline double operator&&(Tensor a,Vector b)
{
  return((a.boolean()&&b.boolean())?1.0:0.0);
}

inline double operator||(double a, Vector b)
{
  return((a||b.boolean())?1.0:0.0);
}

inline double operator||(double a, Tensor b)
{
  return((a||b.boolean())?1.0:0.0);
}

inline double operator||(Vector b,double a)
{
  return((a||b.boolean())?1.0:0.0);
}

inline double operator||(Tensor b,double a)
{
  return((a||b.boolean())?1.0:0.0);
}

inline double operator||(Vector a,Vector b)
{
  return((a.boolean()||b.boolean())?1.0:0.0);
}

inline double operator||(Tensor a,Tensor b)
{
  return((a.boolean()||b.boolean())?1.0:0.0);
}

inline double operator||(Vector a,Tensor b)
{
  return((a.boolean()||b.boolean())?1.0:0.0);
}

inline double operator||(Tensor a,Vector b)
{
  return((a.boolean()||b.boolean())?1.0:0.0);
}

inline double operator!(Vector a)
{
  return((!(a.boolean()))?1.0:0.0);
}

inline double operator!(Tensor a)
{
  return((!(a.boolean()))?1.0:0.0);
}

template<class DATATYPE1, class DATATYPE2> 
inline double AND(DATATYPE1 a,DATATYPE2 b)
{
  return(a&&b);
}

template<class DATATYPE1, class DATATYPE2> 
inline double OR(DATATYPE1 a,DATATYPE2 b)
{
  return(a||b);
}

template<class DATATYPE>
inline double NOT(DATATYPE a)
{
  return(!a);
}

// VECTOR OPERATIONS

inline Vector operator*(const double s, const Vector& vec) 
{
    return(vec*s);
}

inline Vector operator/(const double s, const Vector& vec) 
{
    return(vec/s);
}

inline Vector operator+(const double s, const Vector& vec) 
{
    return(vec+s);
}

inline Vector operator-(const double s, const Vector& vec) 
{
    return(vec-s);
}

inline Vector operator^(const double s, const Vector& vec) 
{
    return(vec^s);
}

inline Vector operator&(const double s, const Vector& vec) 
{
    return(vec&s);
}

inline double length(const Vector& vec)
{
  return(vec.length());
}

inline double norm(const Vector& vec)
{
  return(vec.norm());
}

inline double maxnorm(const Vector& vec)
{
  return(vec.maxnorm());
}

inline double max(const Vector& vec)
{
  return(vec.max());
}

inline double min(const Vector& vec)
{
  return(vec.min());
}

inline double x(const Vector& vec)
{
  return(vec.x());
}

inline double y(const Vector& vec)
{
  return(vec.y());
}

inline double z(const Vector& vec)
{
  return(vec.z());
}


inline Vector sin(const Vector& vec)
{
  return(vec.sin());
}

inline Vector cos(const Vector& vec)
{
  return(vec.cos());
}

inline Vector sinh(const Vector& vec)
{
  return(vec.sinh());
}

inline Vector cosh(const Vector& vec)
{
  return(vec.cosh());
}

inline Vector tan(const Vector& vec)
{
  return(vec.tan());
}

inline Vector asin(const Vector& vec)
{
  return(vec.asin());
}

inline Vector acos(const Vector& vec)
{
  return(vec.acos());
}

inline Vector asinh(const Vector& vec)
{
  return(vec.asinh());
}

inline Vector acosh(const Vector& vec)
{
  return(vec.acosh());
}

inline Vector atan(const Vector& vec)
{
  return(vec.atan());
}

inline Vector log(const Vector& vec)
{
  return(vec.log());
}

inline Vector log10(const Vector& vec)
{
  return(vec.log10());
}

inline Vector log2(const Vector& vec)
{
  return(vec.log2());
}

inline Vector abs(const Vector& vec)
{
  return(vec.abs());
}

inline Vector cbrt(const Vector& vec)
{
  return(vec.cbrt());
}

inline Vector sqrt(const Vector& vec)
{
  return(vec.sqrt());
}

inline Vector pow(const Vector& vec, double d)
{
  return(vec.pow(d));
}

inline Vector exp(const Vector& vec)
{
  return(vec.exp());
}

inline Vector ceil(const Vector& vec)
{
  return(vec.ceil());
}
        
inline Vector floor(const Vector& vec)
{
  return(vec.floor());
}

inline Vector round(const Vector& vec)
{
  return(vec.round());
}
        
inline Vector normalize(const Vector& vec)
{
  return(vec.normalize());
}

inline Vector findnormal1(const Vector& vec)
{
  return(vec.findnormal1());
}

inline Vector findnormal2(const Vector& vec)
{
  return(vec.findnormal2());
}

inline double isnan(const Vector& vec)
{
  return(vec.isnan());
}

inline double isinf(const Vector& vec)
{
  return(vec.isinf());
}

inline double isfinite(const Vector& vec)
{
  return(vec.isfinite());
}

inline double boolean(const Vector& vec)
{
  return(vec.boolean());
}

inline double dot(const Vector vec1,const Vector vec2)
{
  return(vec1.dot(vec2));
}
    
inline Vector cross(const Vector vec1, const Vector vec2)
{
  return(vec1.cross(vec2));
}

// TENSOR OPERATIONS


inline Tensor operator*(const double s, const Tensor& ten) 
{
    return(ten*s);
}

inline Vector operator*(const Vector v, const Tensor& ten) 
{
    // Matrix is symmetric
    return(ten*v);
}

inline Tensor operator/(const double s, const Tensor& ten) 
{
    return(ten/s);
}

inline Tensor operator+(const double s, const Tensor& ten) 
{
    return(ten+s);
}

inline Tensor operator-(const double s, const Tensor& ten) 
{
    return(ten-s);
}

inline Tensor operator^(const double s, const Tensor& ten) 
{
    return(ten^s);
}

inline Tensor operator&(const double s, const Tensor& ten) 
{
    return(ten&s);
}


inline Vector eigvec1(Tensor& ten)
{
  return(ten.eigvec1());
}

inline Vector eigvec2(Tensor& ten)
{
  return(ten.eigvec2());
}

inline Vector eigvec3(Tensor& ten)
{
  return(ten.eigvec3());
}

inline Vector eigvals(Tensor& ten)
{
  return(ten.eigvals());
}

inline double eigval1(Tensor& ten)
{
  return(ten.eigval1());
}

inline double eigval2(Tensor& ten)
{
  return(ten.eigval2());
}
    
inline double eigval3(Tensor& ten)
{
  return(ten.eigval3());
}

inline double trace(Tensor& ten)
{
  return(ten.trace());
}

inline double det(Tensor& ten)
{
  return(ten.det());
}

inline double B(Tensor& ten)
{
  return(ten.B());
}

inline double S(Tensor& ten)
{
  return(ten.S());
}

inline double Q(Tensor& ten)
{
  return(ten.Q());
}

inline double quality(Tensor& ten)
{
  return(ten.quality());
}

inline double frobenius(Tensor& ten)
{
  return(ten.frobenius());
}

inline double frobenius2(Tensor& ten)
{
  return(ten.frobenius2());
}

inline double fracanisotropy(Tensor& ten)
{
  return(ten.fracanisotropy());
}

inline Vector vec1(Tensor& ten)
{
  return(ten.vec1());
}
    
inline Vector vec2(Tensor& ten)
{
  return(ten.vec2());
}

inline Vector vec3(Tensor& ten)
{
  return(ten.vec3());
}

inline double xx(Tensor& ten)
{
  return(ten.xx());
}

inline double xy(Tensor& ten)
{
  return(ten.xy());
}

inline double xz(Tensor& ten)
{
  return(ten.xz());
}

inline double yy(Tensor& ten)
{
  return(ten.yy());
}

inline double yz(Tensor& ten)
{
  return(ten.yz());
}

inline double zz(Tensor& ten)
{
  return(ten.zz());
}

inline double norm(const Tensor ten)
{
  return(ten.norm());
}
    
inline double max(const Tensor ten)
{
  return(ten.max());
}

inline double min(const Tensor ten)
{
  return(ten.min());
}

inline Tensor inv(Tensor ten)
{
  return(ten.inv());
}

inline Tensor sin(const Tensor ten)
{
  return(ten.sin());
}

inline Tensor cos(const Tensor ten)
{
  return(ten.cos());
}

inline Tensor sinh(const Tensor ten)
{
  return(ten.sinh());
}

inline Tensor cosh(const Tensor ten)
{
  return(ten.cosh());
}

inline Tensor tan(const Tensor ten)
{
  return(ten.tan());
}

inline Tensor asin(const Tensor ten)
{
  return(ten.asin());
}

inline Tensor acos(const Tensor ten)
{
  return(ten.acos());
}

inline Tensor asinh(const Tensor ten)
{
  return(ten.asinh());
}

inline Tensor acosh(const Tensor ten)
{
  return(ten.acosh());
}

inline Tensor atan(const Tensor ten)
{
  return(ten.atan());
}

inline Tensor log(const Tensor ten)
{
  return(ten.log());
}

inline Tensor log10(const Tensor ten)
{
  return(ten.log10());
}

inline Tensor log2(const Tensor ten)
{
  return(ten.log2());
}

inline Tensor abs(const Tensor ten)
{
  return(ten.abs());
}

inline Tensor cbrt(const Tensor ten)
{
  return(ten.cbrt());
}

inline Tensor sqrt(const Tensor ten)
{
  return(ten.sqrt());
}

inline Tensor pow(const Tensor ten,double p)
{
  return(ten.pow(p));
}

inline Tensor exp(const Tensor ten)
{
  return(ten.exp());
}

inline Tensor ceil(const Tensor ten)
{
  return(ten.ceil());
}

inline Tensor floor(const Tensor ten)
{
  return(ten.floor());
}

inline Tensor round(const Tensor ten)
{
  return(ten.round());
}

inline Tensor normalize(const Tensor ten)
{
  return(ten.normalize());
}

inline double isnan(const Tensor ten)
{
  return(ten.isnan());
}

inline double isinf(const Tensor ten)
{
  return(ten.isinf());
}

inline double isfinite(const Tensor& ten)
{
  return(ten.isfinite());
}

inline double boolean(const Tensor ten)
{
  return(ten.boolean());
}



// FUNCTIONS DEFINED IN VECTOR CLASS

inline Vector::Vector()
{
  d_[0] = 0.0; d_[1] = 0.0; d_[2] = 0.0;
}

inline Vector::Vector(const Vector& vec)
{
  d_[0] = vec.d_[0]; d_[1] = vec.d_[1]; d_[2] = vec.d_[2];
}
    
inline Vector::Vector(double x, double y, double z)
{
  d_[0] = x; d_[1] = y; d_[2] = z;
}

inline double Vector::operator==(const Vector& vec) const
{
  if ((d_[0]==vec.d_[0])&&(d_[1]==vec.d_[1])&&(d_[2]==vec.d_[2])) return(1.0);
  return(0.0);
}
    
inline double Vector::operator!=(const Vector& vec) const
{
  if ((d_[0]==vec.d_[0])&&(d_[1]==vec.d_[1])&&(d_[2]==vec.d_[2])) return(0.0);
  return(1.0);
}

inline Vector& Vector::operator=(const Vector& vec)
{
    d_[0]=vec.d_[0]; d_[1]=vec.d_[1]; d_[2]=vec.d_[2]; return(*this);
}

inline Vector& Vector::operator=(const double& d)
{
    d_[0]=d; d_[1]=d; d_[2]=d; return(*this);
}

inline double& Vector::operator[](int idx)
{
  return(d_[idx]);
}

inline double Vector::operator[](int idx) const
{
  return(d_[idx]);
}

inline double& Vector::operator()(int idx)
{
  return(d_[idx]);
}

inline double Vector::operator()(int idx) const
{
  return(d_[idx]);
}

inline double& Vector::operator[](std::string s)
{
  if ((s == "x")||(s == "X")||(s == "u")||(s == "U")) return(d_[0]);
  if ((s == "y")||(s == "Y")||(s == "v")||(s == "V")) return(d_[1]);  
  return(d_[2]);
}

inline double Vector::operator[](std::string s) const
{
  if ((s == "x")||(s == "X")||(s == "u")||(s == "U")) return(d_[0]);
  if ((s == "y")||(s == "Y")||(s == "v")||(s == "V")) return(d_[1]);  
  return(d_[2]);
}

inline double& Vector::operator()(std::string s)
{
  if ((s == "x")||(s == "X")||(s == "u")||(s == "U")) return(d_[0]);
  if ((s == "y")||(s == "Y")||(s == "v")||(s == "V")) return(d_[1]);  
  return(d_[2]);
}

inline double Vector::operator()(std::string s) const
{
  if ((s == "x")||(s == "X")||(s == "u")||(s == "U")) return(d_[0]);
  if ((s == "y")||(s == "Y")||(s == "v")||(s == "V")) return(d_[1]);  
  return(d_[2]);
}
    
inline double Vector::x() const
{
  return(d_[0]);
}

inline double Vector::y() const
{
  return(d_[1]);
}

inline double Vector::z() const
{
  return(d_[2]);
}

inline Vector Vector::operator*(const double d) const
{
  return(Vector(d*d_[0],d*d_[1],d*d_[2]));
}

inline Vector Vector::operator*(const Vector& vec) const
{
  return(Vector(vec.d_[0]*d_[0],vec.d_[1]*d_[1],vec.d_[2]*d_[2]));
}

inline Vector& Vector::operator*=(const double d)
{
  d_[0] *= d; d_[1] *= d; d_[2] *= d; return(*this);
}

inline Vector& Vector::operator*=(const Vector& vec)
{
  d_[0] *= vec.d_[0]; d_[1] *= vec.d_[1]; d_[2] *= vec.d_[2]; return(*this);
}

inline Vector Vector::operator/(const double d) const
{
  return(Vector(d_[0]/d,d_[1]/d,d_[2]/d));
}

inline Vector Vector::operator/(const Vector& vec) const
{
  return(Vector(d_[0]/vec.d_[0],d_[1]/vec.d_[1],d_[2]/vec.d_[2]));
}

inline Vector& Vector::operator/=(const double d)
{
  d_[0] /= d; d_[1] /= d; d_[2] /= d; return(*this);
}

inline Vector& Vector::operator/=(const Vector& vec)
{
  d_[0] /= vec.d_[0]; d_[1] /= vec.d_[1]; d_[2] /= vec.d_[2]; return(*this);
}

inline Vector Vector::operator+(const double d) const
{
  return(Vector(d_[0]+d,d_[1]+d,d_[2]+d));
}

inline Vector Vector::operator+(const Vector& vec) const
{
  return(Vector(d_[0]+vec.d_[0],d_[1]+vec.d_[1],d_[2]+vec.d_[2]));
}

inline Vector& Vector::operator+=(const double d)
{
  d_[0] += d; d_[1] += d; d_[2] += d; return(*this);
}

inline Vector& Vector::operator+=(const Vector& vec)
{
  d_[0] += vec.d_[0]; d_[1] += vec.d_[1]; d_[2] += vec.d_[2]; return(*this);
}

inline Vector Vector::operator-(const double d) const
{
  return(Vector(d_[0]-d,d_[1]-d,d_[2]-d));
}

inline Vector Vector::operator-(const Vector& vec) const
{
  return(Vector(d_[0]-vec.d_[0],d_[1]-vec.d_[1],d_[2]-vec.d_[2]));
}

inline Vector& Vector::operator-=(const double d)
{
  d_[0] -= d; d_[1] -= d; d_[2] -= d; return(*this);
}

inline Vector& Vector::operator-=(const Vector& vec)
{
  d_[0] -= vec.d_[0]; d_[1] -= vec.d_[1]; d_[2] -= vec.d_[2]; return(*this);
}

inline Vector Vector::operator^(const double d) const
{
  return(Vector(DataArrayMath::pow(d_[0],d),
    DataArrayMath::pow(d_[1],d),
    DataArrayMath::pow(d_[2],d)));
}

inline Vector Vector::operator^(const Vector& vec) const
{
  return(Vector(DataArrayMath::pow(d_[0],vec.d_[0]),
    DataArrayMath::pow(d_[1],vec.d_[1]),
    DataArrayMath::pow(d_[2],vec.d_[2])));
}

inline Vector& Vector::operator^=(const double d)
{
  d_[0] = DataArrayMath::pow(d_[0],d); 
  d_[1] = DataArrayMath::pow(d_[1],d); 
  d_[2] = DataArrayMath::pow(d_[2],d); 
  return(*this);
}

inline Vector& Vector::operator^=(const Vector& vec)
{
  d_[0] = DataArrayMath::pow(d_[0],vec.d_[0]); 
  d_[1] = DataArrayMath::pow(d_[1],vec.d_[1]); 
  d_[2] = DataArrayMath::pow(d_[2],vec.d_[2]); 
  return(*this);
}

inline Vector Vector::operator-() const
{
  return(Vector(-d_[0],-d_[1],-d_[2]));
}

inline Vector Vector::clone() const
{
  return(Vector(d_[0],d_[1],d_[2]));
}

inline double Vector::length() const
{
  return(DataArrayMath::sqrt(d_[0]*d_[0] + d_[1]*d_[1] +d_[2]*d_[2]));
}

inline double Vector::norm() const
{
  return(DataArrayMath::sqrt(d_[0]*d_[0] + d_[1]*d_[1] +d_[2]*d_[2]));
}
    
inline double Vector::maxnorm() const
{
  double a = DataArrayMath::abs(d_[0]);
  if (a < DataArrayMath::abs(d_[1])) a = DataArrayMath::abs(d_[1]);
  if (a < DataArrayMath::abs(d_[2])) a = DataArrayMath::abs(d_[2]);
  return(a);
}

inline double Vector::max() const
{
  double a = d_[0];
  if (a < d_[1]) a = d_[1];
  if (a < d_[2]) a = d_[2];
  return(a);
}

inline double Vector::min() const
{
  double a = d_[0];
  if (a > d_[1]) a = d_[1];
  if (a > d_[2]) a = d_[2];
  return(a);
}

inline Vector Vector::sin() const
{
  return(Vector(DataArrayMath::sin(d_[0]),
    DataArrayMath::sin(d_[1]),DataArrayMath::sin(d_[2])));
}

inline Vector Vector::cos() const
{
  return(Vector(DataArrayMath::cos(d_[0]),
    DataArrayMath::cos(d_[1]),DataArrayMath::cos(d_[2])));
}
    
inline Vector Vector::sinh() const    
{
  return(Vector(DataArrayMath::sinh(d_[0]),
    DataArrayMath::sinh(d_[1]),DataArrayMath::sinh(d_[2])));
}

inline Vector Vector::cosh() const   
{
  return(Vector(DataArrayMath::cosh(d_[0]),
    DataArrayMath::cosh(d_[1]),DataArrayMath::cosh(d_[2])));
}

inline Vector Vector::tan() const    
{
  return(Vector(DataArrayMath::tan(d_[0]),
    DataArrayMath::tan(d_[1]),DataArrayMath::tan(d_[2])));
}

inline Vector Vector::asin() const
{
  return(Vector(DataArrayMath::asin(d_[0]),
    DataArrayMath::asin(d_[1]),DataArrayMath::asin(d_[2])));
}

inline Vector Vector::acos() const
{
  return(Vector(DataArrayMath::acos(d_[0]),
    DataArrayMath::acos(d_[1]),DataArrayMath::acos(d_[2])));
}
    
inline Vector Vector::asinh() const    
{
  return(Vector(DataArrayMath::asinh(d_[0]),
    DataArrayMath::asinh(d_[1]),DataArrayMath::asinh(d_[2])));
}

inline Vector Vector::acosh() const   
{
  return(Vector(DataArrayMath::acosh(d_[0]),
    DataArrayMath::acosh(d_[1]),DataArrayMath::acosh(d_[2])));
}

inline Vector Vector::atan() const    
{
  return(Vector(DataArrayMath::atan(d_[0]),
    DataArrayMath::atan(d_[1]),DataArrayMath::atan(d_[2])));
}

inline Vector Vector::log() const    
{
  return(Vector(DataArrayMath::log(d_[0]),
    DataArrayMath::log(d_[1]),DataArrayMath::log(d_[2])));
}

inline Vector Vector::log10() const   
{
  return(Vector(DataArrayMath::log10(d_[0]),
    DataArrayMath::log10(d_[1]),DataArrayMath::log10(d_[2])));
}

inline Vector Vector::log2() const    
{
  return(Vector(DataArrayMath::log2(d_[0]),
    DataArrayMath::log2(d_[1]),DataArrayMath::log2(d_[2])));
}

inline Vector Vector::abs() const    
{
  return(Vector(DataArrayMath::abs(d_[0]),
    DataArrayMath::abs(d_[1]),DataArrayMath::abs(d_[2])));
}

inline Vector Vector::sqrt() const    
{
  return(Vector(DataArrayMath::sqrt(d_[0]),
    DataArrayMath::sqrt(d_[1]),DataArrayMath::sqrt(d_[2])));
}

inline Vector Vector::pow(double p) const
{
  return(Vector(DataArrayMath::pow(d_[0],p),
    DataArrayMath::pow(d_[1],p),DataArrayMath::pow(d_[2],p)));
}

inline Vector Vector::cbrt() const
{
  return(Vector(DataArrayMath::cbrt(d_[0]),
    DataArrayMath::cbrt(d_[1]),DataArrayMath::cbrt(d_[2])));
}

inline Vector Vector::exp() const
{
  return(Vector(DataArrayMath::exp(d_[0]),
    DataArrayMath::exp(d_[1]),DataArrayMath::exp(d_[2])));
}

inline Vector Vector::ceil() const
{
  return(Vector(DataArrayMath::ceil(d_[0]),
    DataArrayMath::ceil(d_[1]),DataArrayMath::ceil(d_[2])));
}

inline Vector Vector::floor() const
{
  return(Vector(DataArrayMath::floor(d_[0]),
    DataArrayMath::floor(d_[1]),DataArrayMath::floor(d_[2])));
}

inline Vector Vector::round() const
{
  return(Vector(DataArrayMath::round(d_[0]),
    DataArrayMath::round(d_[1]),DataArrayMath::round(d_[2])));
}

inline Vector Vector::normalize() const
{
  double n = norm();
  if (n > 0.0) return(Vector(d_[0]/n,d_[1]/n,d_[2]/n));
  return(Vector(0.0,0.0,0.0));
}


inline Vector Vector::findnormal1() const
{
  double a0,a1,a2;
  double b0,b1,b2;
  a0 = DataArrayMath::abs(d_[0]); 
  a1 = DataArrayMath::abs(d_[1]); 
  a2 = DataArrayMath::abs(d_[2]);
  
  if ((a0==0.0)&&(a1==0.0)&&(a2==0.0))
  {
    return(Vector(0.0,0.0,0.0));
  }
  
  // determine the two other normal directions
  if ((a0 >= a1)&&(a0 >= a2))
  {
   b0 = d_[1]+d_[2]; b1 = -d_[0]; b2 = -d_[1];
  }
  else if ((a1 >= a0)&&(a1 >= a2))
  {
   b0 = -d_[1]; b1 = d_[0]+d_[2]; b2 = -d_[1];
  }
  else
  {
   b0 = -d_[2]; b1 = -d_[2]; b2 = d_[0]+d_[1]; 
  }

  Vector vec(b0,b1,b2);
  vec.normalize();
  return(vec);
}
    
inline Vector Vector::findnormal2() const
{
  Vector vec = cross(findnormal1());
  vec.normalize();
  return(vec);
}

inline double Vector::isnan() const
{
  if ((DataArrayMath::isnan(d_[0]))||(DataArrayMath::isnan(d_[1]))||
    (DataArrayMath::isnan(d_[2]))) return(1.0);
  return(0.0);
}
    
inline double Vector::isinf() const
{
  if ((DataArrayMath::isinf(d_[0]))||(DataArrayMath::isinf(d_[1]))||
    (DataArrayMath::isinf(d_[2]))) return(1.0);
  return(0.0);
}

inline double Vector::isfinite() const
{
  if ((DataArrayMath::isfinite(d_[0]))&&
    (DataArrayMath::isfinite(d_[1]))&&
    (DataArrayMath::isfinite(d_[2]))) return(1.0);
  return(0.0);
}

inline double Vector::dot(const Vector& vec) const
{
  return(d_[0]*vec.d_[0] + d_[1]*vec.d_[1] + d_[2]*vec.d_[2] );
}

inline Vector Vector::cross(const Vector& vec) const
{
  return(Vector(d_[1]*vec.d_[2] - d_[2]*vec.d_[1], 
                d_[2]*vec.d_[0] - d_[0]*vec.d_[2],
                d_[0]*vec.d_[1] - d_[1]*vec.d_[0]));
}

inline Vector Vector::operator&(const double d) const
{
  if (d) return(Vector(d_[0],d_[1],d_[2]));
  return(Vector(0,0,0));
}

inline Vector& Vector::operator&=(const double d)
{
  if (!d) { d_[0] = 0.0; d_[1] = 0.0; d_[2] = 0.0; } 
  return(*this);
}

inline double Vector::boolean() const
{
  if ((d_[0])||(d_[1])||(d_[2])) return(1.0);
  return(0.0);
}

inline double* Vector::getdataptr()
{
  return(d_);
}

inline Tensor::Tensor() :
  has_eigs_(false)
{
  d_[0] = 0.0; d_[1] = 0.0; d_[2] = 0.0;
  d_[3] = 0.0; d_[4] = 0.0; d_[5] = 0.0;
}
    
inline Tensor::Tensor(const Tensor& ten)
{
  d_[0] = ten.d_[0]; d_[1] = ten.d_[1]; d_[2] = ten.d_[2];
  d_[3] = ten.d_[3]; d_[4] = ten.d_[4]; d_[5] = ten.d_[5];
  has_eigs_ = ten.has_eigs_;
  eigvec1_ = ten.eigvec1_;
  eigvec2_ = ten.eigvec2_;
  eigvec3_ = ten.eigvec3_;
  eigval1_ = ten.eigval1_;
  eigval2_ = ten.eigval2_;
  eigval3_ = ten.eigval3_;
}

inline Tensor::Tensor(double xx, double xy, double xz, double yy, double yz, double zz)
{
  d_[0] = xx; d_[1] = xy; d_[2] = xz;
  d_[3] = yy; d_[4] = yz; d_[5] = zz;
  has_eigs_ = false;
}

inline Tensor::Tensor(double* ptr, int size)
{
  if (size == 6)
  {
    d_[0] = ptr[0]; d_[1] = ptr[1]; d_[2] = ptr[2];
    d_[3] = ptr[3]; d_[4] = ptr[4]; d_[5] = ptr[5];
    has_eigs_ = false;
  }
  else if (size == 9)
  {
    d_[0] = ptr[0]; d_[1] = ptr[1]; d_[2] = ptr[2];
    d_[3] = ptr[4]; d_[4] = ptr[5]; d_[5] = ptr[8];
    has_eigs_ = false;  
  }
  else
  {
    d_[0] = 0.0; d_[1] = 0.0; d_[2] = 0.0;
    d_[3] = 0.0; d_[4] = 0.0; d_[5] = 0.0;
    has_eigs_ = false;  
  }
}

inline Tensor::Tensor(Vector e1, Vector e2, double s1, double s2, double s3)
{
  eigvec1_ = DataArrayMath::normalize(e1);
  eigvec2_ = DataArrayMath::normalize(e2);
  eigvec3_ = DataArrayMath::cross(e1,e2);
  eigval1_ = s1;
  eigval2_ = s2;
  eigval3_ = s3;
  has_eigs_ = true;
  compute_tensor();
}

inline Tensor::Tensor(Vector e1, Vector e2, double s1, double s2)
{
  eigvec1_ = DataArrayMath::normalize(e1);
  eigvec2_ = DataArrayMath::normalize(e2);
  eigvec3_ = DataArrayMath::cross(e1,e2);
  eigval1_ = s1;
  eigval2_ = s2;
  eigval3_ = s2;
  has_eigs_ = true;
  compute_tensor();
}


inline Tensor::Tensor(Vector e1, Vector e2, Vector e3, double s1, double s2, double s3)
{
  eigvec1_ = DataArrayMath::normalize(e1);
  eigvec2_ = DataArrayMath::normalize(e2);
  eigvec3_ = DataArrayMath::normalize(e3);
  eigval1_ = s1;
  eigval2_ = s2;
  eigval3_ = s3;
  has_eigs_ = true;
  compute_tensor();  
}

inline Tensor::Tensor(double s1)
{
  eigvec1_ = Vector(1.0,0.0,0.0);
  eigvec2_ = Vector(0.0,1.0,0.0);
  eigvec3_ = Vector(0.0,0.0,1.0);
  eigval1_ = s1;
  eigval2_ = s1;
  eigval3_ = s1;
  has_eigs_ = true;
  compute_tensor();
}

inline Tensor::Tensor(Vector e1, double s1, double s2)
{
  e1 = DataArrayMath::normalize(e1);
  eigvec1_ = e1;
  eigvec2_ = DataArrayMath::findnormal1(e1);
  eigvec2_ = DataArrayMath::findnormal2(e1);
  eigval1_ = s1;
  eigval2_ = s2;
  eigval3_ = s2;
  has_eigs_ = true;
  compute_tensor();
}

inline Vector Tensor::eigvec1()
{
  if (!has_eigs_) compute_eigs();
  return(eigvec1_);
}

inline Vector Tensor::eigvec2()
{
  if (!has_eigs_) compute_eigs();
  return(eigvec2_);
}

inline Vector Tensor::eigvec3()
{
  if (!has_eigs_) compute_eigs();
  return(eigvec3_);
}

inline Vector Tensor::eigvals()
{
  if (!has_eigs_) compute_eigs();
  return(Vector(eigval1_,eigval2_,eigval3_));
}

inline double Tensor::eigval1()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_);
}

inline double Tensor::eigval2()
{
 if (!has_eigs_) compute_eigs();
 return(eigval2_);
}

inline double Tensor::eigval3()
{
 if (!has_eigs_) compute_eigs();
 return(eigval3_);
}

inline double Tensor::trace()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_+eigval2_+eigval3_);
}

inline double Tensor::det()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_*eigval2_*eigval3_);
}

inline double Tensor::B()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_*eigval2_+eigval1_*eigval3_+eigval2_*eigval3_);
}

inline double Tensor::S()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_);
}

inline double Tensor::Q()
{
 if (!has_eigs_) compute_eigs();
 return(((eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_)-
  (eigval1_*eigval2_+eigval1_*eigval3_+eigval2_*eigval3_))/9.0);
}

inline double Tensor::quality()
{
 if (!has_eigs_) compute_eigs();
 return(((eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_)-
  (eigval1_*eigval2_+eigval1_*eigval3_+eigval2_*eigval3_))/9.0);
}

inline double Tensor::frobenius()
{
 if (!has_eigs_) compute_eigs();
 return(DataArrayMath::sqrt(eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_));
}

inline double Tensor::frobenius2()
{
 if (!has_eigs_) compute_eigs();
 return(eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_);
}
  
inline double Tensor::fracanisotropy()
{
 if (!has_eigs_) compute_eigs();
 double S = eigval1_*eigval1_+eigval2_*eigval2_+eigval3_*eigval3_;
 double B = eigval1_*eigval2_+eigval1_*eigval3_+eigval2_*eigval3_; 
 return(DataArrayMath::sqrt((S-B)/S));
}

inline Vector Tensor::vec1() const
{
  return(Vector(d_[0],d_[1],d_[2]));
}
    
inline Vector Tensor::vec2() const
{
  return(Vector(d_[1],d_[3],d_[4]));
}
    
inline Vector Tensor::vec3() const
{
  return(Vector(d_[2],d_[4],d_[5]));
}

inline double Tensor::xx() const
{
  return(d_[0]);
}

inline double Tensor::xy() const
{
  return(d_[1]);
}

inline double Tensor::xz() const
{
  return(d_[2]);
}

inline double Tensor::yy() const
{
  return(d_[3]);
}

inline double Tensor::yz() const
{
  return(d_[4]);
}

inline double Tensor::zz() const
{
  return(d_[5]);
}

// operators defined for this class
    
inline Tensor& Tensor::operator=(const Tensor& ten)
{
  has_eigs_ = false;
  d_[0] = ten.d_[0]; d_[1] = ten.d_[1]; d_[2] = ten.d_[2]; 
  d_[3] = ten.d_[3]; d_[4] = ten.d_[4]; d_[5] = ten.d_[5];
  return(*this);
}

inline double& Tensor::operator[](int idx)
{
  return(d_[idx]);
}

inline double Tensor::operator[](int idx) const
{
  return(d_[idx]);
}

inline double& Tensor::operator()(int idx)
{
  return(d_[idx]);
}

inline double Tensor::operator()(int idx) const
{
  return(d_[idx]);
}

inline double& Tensor::operator[](std::string s)
{
  if ((s == "xx")||(s == "XX")||(s == "uu")||(s ==  "UU")) return(d_[0]);
  if ((s == "xy")||(s == "XY")||(s == "uv")||(s ==  "UV")) return(d_[1]);
  if ((s == "xz")||(s == "XZ")||(s == "uw")||(s ==  "UW")) return(d_[2]);
  if ((s == "yy")||(s == "YY")||(s == "vv")||(s ==  "VV")) return(d_[3]);
  if ((s == "yz")||(s == "YZ")||(s == "vw")||(s ==  "VW")) return(d_[4]);
  return(d_[5]);
}

inline double Tensor::operator[](std::string s) const
{
  if ((s == "xx")||(s == "XX")||(s == "uu")||(s ==  "UU")) return(d_[0]);
  if ((s == "xy")||(s == "XY")||(s == "uv")||(s ==  "UV")) return(d_[1]);
  if ((s == "xz")||(s == "XZ")||(s == "uw")||(s ==  "UW")) return(d_[2]);
  if ((s == "yy")||(s == "YY")||(s == "vv")||(s ==  "VV")) return(d_[3]);
  if ((s == "yz")||(s == "YZ")||(s == "vw")||(s ==  "VW")) return(d_[4]);
  return(d_[5]);
}

inline double& Tensor::operator()(std::string s)
{
  if ((s == "xx")||(s == "XX")||(s == "uu")||(s ==  "UU")) return(d_[0]);
  if ((s == "xy")||(s == "XY")||(s == "uv")||(s ==  "UV")) return(d_[1]);
  if ((s == "xz")||(s == "XZ")||(s == "uw")||(s ==  "UW")) return(d_[2]);
  if ((s == "yy")||(s == "YY")||(s == "vv")||(s ==  "VV")) return(d_[3]);
  if ((s == "yz")||(s == "YZ")||(s == "vw")||(s ==  "VW")) return(d_[4]);
  return(d_[5]);
}

inline double Tensor::operator()(std::string s) const
{
  if ((s == "xx")||(s == "XX")||(s == "uu")||(s ==  "UU")) return(d_[0]);
  if ((s == "xy")||(s == "XY")||(s == "uv")||(s ==  "UV")) return(d_[1]);
  if ((s == "xz")||(s == "XZ")||(s == "uw")||(s ==  "UW")) return(d_[2]);
  if ((s == "yy")||(s == "YY")||(s == "vv")||(s ==  "VV")) return(d_[3]);
  if ((s == "yz")||(s == "YZ")||(s == "vw")||(s ==  "VW")) return(d_[4]);
  return(d_[5]);
}

inline Vector Tensor::operator*(const Vector& vec) const
{
  return(Vector(d_[0]*vec[0]+d_[1]*vec[1]+d_[2]*vec[2],
                d_[1]*vec[0]+d_[3]*vec[1]+d_[4]*vec[2],
                d_[2]*vec[0]+d_[4]*vec[1]+d_[5]*vec[2]));
}
 
inline Tensor Tensor::operator*(const double d) const
{
  return(Tensor(d*d_[0],d*d_[1],d*d_[2],d*d_[3],d*d_[4],d*d_[5]));
}

inline Tensor Tensor::operator*(const Tensor& ten) const
{
  return(Tensor(d_[0]*ten.d_[0]+d_[1]*ten.d_[1]+d_[2]*ten.d_[2],
                d_[1]*ten.d_[0]+d_[3]*ten.d_[1]+d_[4]*ten.d_[2],
                d_[2]*ten.d_[0]+d_[4]*ten.d_[1]+d_[5]*ten.d_[2],
                d_[1]*ten.d_[1]+d_[3]*ten.d_[3]+d_[4]*ten.d_[4],
                d_[2]*ten.d_[1]+d_[4]*ten.d_[3]+d_[5]*ten.d_[4],
                d_[2]*ten.d_[2]+d_[4]*ten.d_[4]+d_[5]*ten.d_[5]));              

}

inline Tensor& Tensor::operator*=(const double d)
{
  d_[0] *= d; d_[1] *= d; d_[2] *= d; d_[3] *= d; d_[4] *= d; d_[5] *= d; 
  return(*this);
}

inline Tensor& Tensor::operator*=(const Tensor& ten)
{
  double a0 = d_[0]*ten.d_[0]+d_[1]*ten.d_[1]+d_[2]*ten.d_[2];
  double a1 = d_[1]*ten.d_[0]+d_[3]*ten.d_[1]+d_[4]*ten.d_[2];
  double a2 = d_[2]*ten.d_[0]+d_[4]*ten.d_[1]+d_[5]*ten.d_[2];
  double a3 = d_[1]*ten.d_[1]+d_[3]*ten.d_[3]+d_[4]*ten.d_[4];
  double a4 = d_[2]*ten.d_[1]+d_[4]*ten.d_[3]+d_[5]*ten.d_[4];
  double a5 = d_[2]*ten.d_[2]+d_[4]*ten.d_[4]+d_[5]*ten.d_[5];      
  d_[0] = a0; d_[1] = a1; d_[2] = a2; d_[3] = a3; d_[4] = a4; d_[5] = a5; 
  return (*this);
}

inline Tensor Tensor::operator/(const double d) const
{
  return(Tensor(d_[0]/d,d_[1]/d,d_[2]/d,d_[3]/d,d_[4]/d,d_[5]/d));
}

inline Tensor Tensor::operator/(const Tensor& ten) const
{
  return(Tensor(d_[0]/ten.d_[0],d_[1]/ten.d_[1],d_[2]/ten.d_[2],
  d_[3]/ten.d_[3],d_[4]/ten.d_[4],d_[5]/ten.d_[5]));
}

inline Tensor& Tensor::operator/=(const double d)
{
  d_[0] /= d; d_[1] /= d; d_[2] /= d; 
  d_[3] /= d; d_[4] /= d; d_[5] /= d; 
  return(*this);
}

inline Tensor& Tensor::operator/=(const Tensor& ten)
{
  d_[0] /= ten.d_[0]; d_[1] /= ten.d_[1]; d_[2] /= ten.d_[2];
  d_[3] /= ten.d_[3]; d_[4] /= ten.d_[4]; d_[5] /= ten.d_[5]; 
  return(*this);
}

inline Tensor Tensor::operator+(const double d) const
{
  return(Tensor(d_[0]+d,d_[1]+d,d_[2]+d,d_[3]+d,d_[4]+d,d_[5]+d));
}

inline Tensor Tensor::operator+(const Tensor& ten) const
{
  return(Tensor(d_[0]+ten.d_[0],d_[1]+ten.d_[1],
      d_[2]+ten.d_[2],d_[3]+ten.d_[3],
      d_[4]+ten.d_[4],d_[5]+ten.d_[5]));
}

inline Tensor& Tensor::operator+=(const double d)
{
  d_[0] += d; d_[1] += d; d_[2] += d; d_[3] += d; d_[4] += d; d_[5] += d; 
  return(*this);
}

inline Tensor& Tensor::operator+=(const Tensor& ten)
{
  d_[0] += ten.d_[0]; d_[1] += ten.d_[1]; d_[2] += ten.d_[2]; 
  d_[3] += ten.d_[3]; d_[4] += ten.d_[4]; d_[5] += ten.d_[5]; 
  return(*this);
}

inline Tensor Tensor::operator-(const double d) const
{
  return(Tensor(d_[0]-d,d_[1]-d,d_[2]-d,d_[3]-d,d_[4]-d,d_[5]-d));
}

inline Tensor Tensor::operator-(const Tensor& ten) const
{
  return(Tensor(d_[0]-ten.d_[0],d_[1]-ten.d_[1],d_[2]-ten.d_[2]
    ,d_[3]-ten.d_[3],d_[4]-ten.d_[4],d_[5]-ten.d_[5]));
}

inline Tensor& Tensor::operator-=(const double d)
{
  d_[0] -= d; d_[1] -= d; d_[2] -= d;
  d_[3] -= d; d_[4] -= d; d_[5] -= d; 
  return(*this);
}

inline Tensor& Tensor::operator-=(const Tensor& ten)
{
  d_[0] -= ten.d_[0]; d_[1] -= ten.d_[1]; d_[2] -= ten.d_[2];
  d_[3] -= ten.d_[3]; d_[4] -= ten.d_[4]; d_[5] -= ten.d_[5]; 
  return(*this);
}

inline Tensor Tensor::operator^(const double d) const
{
  return(Tensor(DataArrayMath::pow(d_[0],d),
    DataArrayMath::pow(d_[1],d),
    DataArrayMath::pow(d_[2],d),
    DataArrayMath::pow(d_[3],d),
    DataArrayMath::pow(d_[4],d),
    DataArrayMath::pow(d_[5],d)));
}

inline Tensor Tensor::operator^(const Tensor& ten) const
{
  return(Tensor(DataArrayMath::pow(d_[0],ten.d_[0]),
    DataArrayMath::pow(d_[1],ten.d_[1]),
    DataArrayMath::pow(d_[2],ten.d_[2]),
    DataArrayMath::pow(d_[3],ten.d_[3]),
    DataArrayMath::pow(d_[4],ten.d_[4]),
    DataArrayMath::pow(d_[5],ten.d_[5])));
}

inline Tensor& Tensor::operator^=(const double d)
{
  d_[0] = DataArrayMath::pow(d_[0],d); 
  d_[1] = DataArrayMath::pow(d_[1],d); 
  d_[2] = DataArrayMath::pow(d_[2],d); 
  d_[3] = DataArrayMath::pow(d_[3],d); 
  d_[4] = DataArrayMath::pow(d_[4],d); 
  d_[5] = DataArrayMath::pow(d_[5],d); 
  return(*this);
}

inline Tensor& Tensor::operator^=(const Tensor& ten)
{
  d_[0] = DataArrayMath::pow(d_[0],ten.d_[0]); 
  d_[1] = DataArrayMath::pow(d_[1],ten.d_[1]); 
  d_[2] = DataArrayMath::pow(d_[2],ten.d_[2]); 
  d_[3] = DataArrayMath::pow(d_[3],ten.d_[3]); 
  d_[4] = DataArrayMath::pow(d_[4],ten.d_[4]); 
  d_[5] = DataArrayMath::pow(d_[5],ten.d_[5]); 
  return(*this);
}

inline Tensor Tensor::operator-() const
{
  return(Tensor(-d_[0],-d_[1],-d_[2],-d_[3],-d_[4],-d_[5]));
}

inline Tensor Tensor::clone() const
{
  return(Tensor(d_[0],d_[1],d_[2],d_[3],d_[4],d_[5]));
}


inline double Tensor::norm() const
{
  double a = DataArrayMath::abs(d_[0]);
  if (a < DataArrayMath::abs(d_[1])) a = DataArrayMath::abs(d_[1]);
  if (a < DataArrayMath::abs(d_[2])) a = DataArrayMath::abs(d_[2]);
  if (a < DataArrayMath::abs(d_[3])) a = DataArrayMath::abs(d_[3]);
  if (a < DataArrayMath::abs(d_[4])) a = DataArrayMath::abs(d_[4]);
  if (a < DataArrayMath::abs(d_[5])) a = DataArrayMath::abs(d_[5]);
  return(a);
}

inline double Tensor::max() const
{
  double a = d_[0];
  if (a < d_[1]) a = d_[1];
  if (a < d_[2]) a = d_[2];
  if (a < d_[3]) a = d_[3];
  if (a < d_[4]) a = d_[4];
  if (a < d_[5]) a = d_[5];
  return(a);
}

inline double Tensor::min() const
{
  double a = d_[0];
  if (a > d_[1]) a = d_[1];
  if (a > d_[2]) a = d_[2];
  if (a > d_[3]) a = d_[3];
  if (a > d_[4]) a = d_[4];
  if (a > d_[5]) a = d_[5];
  return(a);
}

inline Tensor Tensor::sin() const
{
  return(Tensor(DataArrayMath::sin(d_[0]),DataArrayMath::sin(d_[1]),
    DataArrayMath::sin(d_[2]),DataArrayMath::sin(d_[3]),
    DataArrayMath::sin(d_[4]),DataArrayMath::sin(d_[5])));
}

inline Tensor Tensor::cos() const
{
  return(Tensor(DataArrayMath::cos(d_[0]),DataArrayMath::cos(d_[1]),
    DataArrayMath::cos(d_[2]),DataArrayMath::cos(d_[3]),
    DataArrayMath::cos(d_[4]),DataArrayMath::cos(d_[5])));
}
    
inline Tensor Tensor::sinh() const    
{
  return(Tensor(DataArrayMath::sinh(d_[0]),DataArrayMath::sinh(d_[1]),
    DataArrayMath::sinh(d_[2]),DataArrayMath::sinh(d_[3]),
    DataArrayMath::sinh(d_[4]),DataArrayMath::sinh(d_[5])));
}

inline Tensor Tensor::cosh() const   
{
  return(Tensor(DataArrayMath::cosh(d_[0]),DataArrayMath::cosh(d_[1]),
    DataArrayMath::cosh(d_[2]),DataArrayMath::cosh(d_[3]),
    DataArrayMath::cosh(d_[4]),DataArrayMath::cosh(d_[5])));
}

inline Tensor Tensor::tan() const    
{
  return(Tensor(DataArrayMath::tan(d_[0]),DataArrayMath::tan(d_[1]),
    DataArrayMath::tan(d_[2]),DataArrayMath::tan(d_[3]),
    DataArrayMath::tan(d_[4]),DataArrayMath::tan(d_[5])));
}

inline Tensor Tensor::asin() const
{
  return(Tensor(DataArrayMath::asin(d_[0]),DataArrayMath::asin(d_[1]),
    DataArrayMath::asin(d_[2]),DataArrayMath::asin(d_[3]),
    DataArrayMath::asin(d_[4]),DataArrayMath::asin(d_[5])));
}

inline Tensor Tensor::acos() const
{
  return(Tensor(DataArrayMath::acos(d_[0]),DataArrayMath::acos(d_[1]),
    DataArrayMath::acos(d_[2]),DataArrayMath::acos(d_[3]),
    DataArrayMath::acos(d_[4]),DataArrayMath::acos(d_[5])));
}
    
inline Tensor Tensor::asinh() const    
{
  return(Tensor(DataArrayMath::asinh(d_[0]),DataArrayMath::asinh(d_[1]),
    DataArrayMath::asinh(d_[2]),DataArrayMath::asinh(d_[3]),
    DataArrayMath::asinh(d_[4]),DataArrayMath::asinh(d_[5])));
}

inline Tensor Tensor::acosh() const   
{
  return(Tensor(DataArrayMath::acosh(d_[0]),DataArrayMath::acosh(d_[1]),
    DataArrayMath::acosh(d_[2]),DataArrayMath::acosh(d_[3]),
    DataArrayMath::acosh(d_[4]),DataArrayMath::acosh(d_[5])));
}

inline Tensor Tensor::atan() const    
{
  return(Tensor(DataArrayMath::atan(d_[0]),DataArrayMath::atan(d_[1]),
    DataArrayMath::atan(d_[2]),DataArrayMath::atan(d_[3]),
    DataArrayMath::atan(d_[4]),DataArrayMath::atan(d_[5])));
}

inline Tensor Tensor::log() const    
{
  return(Tensor(DataArrayMath::log(d_[0]),DataArrayMath::log(d_[1]),
    DataArrayMath::log(d_[2]),DataArrayMath::log(d_[3]),
    DataArrayMath::log(d_[4]),DataArrayMath::log(d_[5])));
}

inline Tensor Tensor::log10() const   
{
  return(Tensor(DataArrayMath::log10(d_[0]),DataArrayMath::log10(d_[1]),
    DataArrayMath::log10(d_[2]),DataArrayMath::log10(d_[3]),
    DataArrayMath::log10(d_[4]),DataArrayMath::log10(d_[5])));
}

inline Tensor Tensor::log2() const    
{
  return(Tensor(DataArrayMath::log2(d_[0]),DataArrayMath::log2(d_[1]),
    DataArrayMath::log2(d_[2]),DataArrayMath::log2(d_[3]),
    DataArrayMath::log2(d_[4]),DataArrayMath::log2(d_[5])));
}

inline Tensor Tensor::abs() const    
{
  return(Tensor(DataArrayMath::abs(d_[0]),DataArrayMath::abs(d_[1]),
    DataArrayMath::abs(d_[2]),DataArrayMath::abs(d_[3]),
    DataArrayMath::abs(d_[4]),DataArrayMath::abs(d_[5])));
}

inline Tensor Tensor::sqrt() const    
{
  return(Tensor(DataArrayMath::sqrt(d_[0]),DataArrayMath::sqrt(d_[1]),
    DataArrayMath::sqrt(d_[2]),DataArrayMath::sqrt(d_[3]),
    DataArrayMath::sqrt(d_[4]),DataArrayMath::sqrt(d_[5])));
}

inline Tensor Tensor::pow(double p) const
{
  return(Tensor(DataArrayMath::pow(d_[0],p),DataArrayMath::pow(d_[1],p),
  DataArrayMath::pow(d_[2],p),DataArrayMath::pow(d_[3],p),
  DataArrayMath::pow(d_[4],p),DataArrayMath::pow(d_[5],p)));
}

inline Tensor Tensor::cbrt() const
{
  return(Tensor(DataArrayMath::cbrt(d_[0]),DataArrayMath::cbrt(d_[1]),
    DataArrayMath::cbrt(d_[2]),DataArrayMath::cbrt(d_[3]),
    DataArrayMath::cbrt(d_[4]),DataArrayMath::cbrt(d_[5])));
}

inline Tensor Tensor::exp() const
{
  return(Tensor(DataArrayMath::exp(d_[0]),DataArrayMath::exp(d_[1]),
    DataArrayMath::exp(d_[2]),DataArrayMath::exp(d_[3]),
    DataArrayMath::exp(d_[4]),DataArrayMath::exp(d_[5])));
}

inline Tensor Tensor::ceil() const
{
  return(Tensor(DataArrayMath::ceil(d_[0]),DataArrayMath::ceil(d_[1]),
    DataArrayMath::ceil(d_[2]),DataArrayMath::ceil(d_[3]),
    DataArrayMath::ceil(d_[4]),DataArrayMath::ceil(d_[5])));
}

inline Tensor Tensor::floor() const
{
  return(Tensor(DataArrayMath::floor(d_[0]),DataArrayMath::floor(d_[1]),
    DataArrayMath::floor(d_[2]),DataArrayMath::floor(d_[3]),
    DataArrayMath::floor(d_[4]),DataArrayMath::floor(d_[5])));
}

inline Tensor Tensor::round() const
{
  return(Tensor(DataArrayMath::round(d_[0]),DataArrayMath::round(d_[1]),
    DataArrayMath::round(d_[2]),DataArrayMath::round(d_[3]),
    DataArrayMath::round(d_[4]),DataArrayMath::round(d_[5])));
}

inline Tensor Tensor::normalize() const
{
  double n = norm();
  if (n > 0.0) return(Tensor(d_[0]/n,d_[1]/n,d_[2]/n,d_[3]/n,d_[4]/n,d_[5]/n));
  return(Tensor(0.0,0.0,0.0,0.0,0.0,0.0));
}

inline Tensor Tensor::inv()
{
 if (!has_eigs_) compute_eigs();
 return(Tensor(eigvec1_,eigvec2_,eigvec3_,1.0/eigval1_,1.0/eigval2_,1.0/eigval3_));
}

inline double Tensor::isnan() const
{
  if ((DataArrayMath::isnan(d_[0]))||(DataArrayMath::isnan(d_[1]))||
    (DataArrayMath::isnan(d_[2]))||(DataArrayMath::isnan(d_[3]))||
    (DataArrayMath::isnan(d_[4]))||(DataArrayMath::isnan(d_[5]))) 
    return(1.0);
  return(0.0);
}
    
inline double Tensor::isinf() const
{
  if ((DataArrayMath::isinf(d_[0]))||(DataArrayMath::isinf(d_[1]))||
    (DataArrayMath::isinf(d_[2]))||(DataArrayMath::isinf(d_[3]))||
    (DataArrayMath::isinf(d_[4]))||(DataArrayMath::isinf(d_[5]))) 
    return(1.0);
  return(0.0);
}

inline double Tensor::isfinite() const
{
  if ((DataArrayMath::isfinite(d_[0]))&&(DataArrayMath::isfinite(d_[1]))
    &&(DataArrayMath::isfinite(d_[2]))&&(DataArrayMath::isfinite(d_[3]))
    &&(DataArrayMath::isfinite(d_[4]))&&(DataArrayMath::isfinite(d_[5]))) 
    return(1.0);
  return(0.0);
}

inline Tensor Tensor::operator&(const double d) const
{
  if (d) return(Tensor(d_[0],d_[1],d_[2],d_[3],d_[4],d_[5]));
  return(Tensor(0.0,0.0,0.0,0.0,0.0,0.0));
}

inline Tensor& Tensor::operator&=(const double d)
{
  has_eigs_ = false;
  if (!d) { d_[0] = 0.0; d_[1] = 0.0; d_[2] = 0.0; d_[3] = 0.0; d_[4] = 0.0; d_[5] = 0.0; } 
  return(*this);
}

inline double Tensor::boolean() const
{
  if((d_[0])||(d_[1])||(d_[2])||(d_[3])||(d_[4])||(d_[5])) return(1.0);
  return(0.0);
}

inline void Tensor::compute_eigs()
{
#ifdef HAVE_TEEM
  double ten[7];
  double eval[3];
  double evec[9];
  
  ten[0] = 1.0;
  ten[1] = d_[0];
  ten[2] = d_[1];
  ten[3] = d_[2];
  ten[4] = d_[3];
  ten[5] = d_[4];
  ten[6] = d_[5];
  
  tenEigensolve_d(eval, evec, ten);
 
  eigvec1_[0] = evec[0];
  eigvec1_[1] = evec[1];
  eigvec1_[2] = evec[2];
  eigvec2_[0] = evec[3];
  eigvec2_[1] = evec[4];
  eigvec2_[2] = evec[5];
  eigvec3_[0] = evec[6];
  eigvec3_[1] = evec[7];
  eigvec3_[2] = evec[8];
   
  eigval1_ = eval[0];
  eigval2_ = eval[1];
  eigval3_ = eval[2];
  
  has_eigs_ = true;
#endif
}

inline void Tensor::compute_tensor()
{
  d_[0] = (eigval1_*eigvec1_[0]*eigvec1_[0]+eigval2_*eigvec2_[0]*eigvec2_[0]+eigval3_*eigvec3_[0]*eigvec3_[0]);
  d_[1] = (eigval1_*eigvec1_[1]*eigvec1_[0]+eigval2_*eigvec2_[1]*eigvec2_[0]+eigval3_*eigvec3_[1]*eigvec3_[0]);
  d_[2] = (eigval1_*eigvec1_[2]*eigvec1_[0]+eigval2_*eigvec2_[2]*eigvec2_[0]+eigval3_*eigvec3_[2]*eigvec3_[0]);
  d_[3] = (eigval1_*eigvec1_[1]*eigvec1_[1]+eigval2_*eigvec2_[1]*eigvec2_[1]+eigval3_*eigvec3_[1]*eigvec3_[1]);
  d_[4] = (eigval1_*eigvec1_[1]*eigvec1_[2]+eigval2_*eigvec2_[1]*eigvec2_[2]+eigval3_*eigvec3_[1]*eigvec3_[2]);
  d_[5] = (eigval1_*eigvec1_[2]*eigvec1_[2]+eigval2_*eigvec2_[2]*eigvec2_[2]+eigval3_*eigvec3_[2]*eigvec3_[2]);  
}

inline double* Tensor::getdataptr()
{
  return(d_);
}



inline double Tensor::operator<(const Tensor& t)
{
  return(static_cast<double>(norm() < t.norm()));
}

inline double Tensor::operator<=(const Tensor& t)
{
  return(static_cast<double>(norm() <= t.norm()));
}

inline double Tensor::operator>(const Tensor& t)
{
  return(static_cast<double>(norm() > t.norm()));
}

inline double Tensor::operator>=(const Tensor& t)
{
  return(static_cast<double>(norm() >= t.norm()));
}

inline double Tensor::operator<(const double& d)
{
  return(static_cast<double>(norm() < d));
}

inline double Tensor::operator<=(const double& d)
{
  return(static_cast<double>(norm() <= d));
}

inline double Tensor::operator>(const double& d)
{
  return(static_cast<double>(norm() > d));
}

inline double Tensor::operator>=(const double& d)
{
  return(static_cast<double>(norm() >= d));
}



} // end namespace

#endif
