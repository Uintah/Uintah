/* builtin.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Institute
 *
 *  Classes for built in datatypes
 */

#ifndef builtin_h
#define builtin_h


namespace SCIRun {

class Scalar {
public:
  virtual ~Scalar() {}
  virtual operator char() = 0;
  virtual operator short() = 0;
  virtual operator int() = 0;
  virtual operator float() = 0;
  virtual operator double() = 0;
};

template<class T>
class ScalarType : public Scalar{
public:
  T val_;
  
  ScalarType() {}
  ScalarType( T v ) : val_(v) {}

  void operator=( const ScalarType &copy ) { val_ = copy.val_;}
  operator char()   { return char(val_); }
  operator short()  { return short(val_); }
  operator int()    { return int(val_); }
  operator float()  { return float(val_); }
  operator double() { return double(val_); }
};

typedef ScalarType<char>   Char;
typedef ScalarType<short>  Short;
typedef ScalarType<int>    Int;
typedef ScalarType<float>  Float;
typedef ScalarType<double> Double;

} // end namespace SCIRun

#endif // builtin_h
