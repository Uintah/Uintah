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

#include <Core/Datatypes/TypeName.h>

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

SCICORESHARE inline void Pio(Piostream& stream, Char& d)  {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Short& d) {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Int& d)   {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Float& d) {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream,Double& d) {Pio(stream,d.val_);}

inline const string find_type_name(Char*)  {return find_type_name((char *)0);}
inline const string find_type_name(Short*) {return find_type_name((short *)0);}
inline const string find_type_name(Int*)   {return find_type_name((int *)0);}
inline const string find_type_name(Float*) {return find_type_name((float *)0);}
inline const string find_type_name(Double*){return find_type_name((double *)0);}


template<class T> bool is_scalar() { return false; }
template<> inline bool is_scalar<char>() { return true; }
template<> inline bool is_scalar<short>() { return true; }
template<> inline bool is_scalar<int>() { return true; }
template<> inline bool is_scalar<float>() { return true; }
template<> inline bool is_scalar<double>() { return true; }

} // end namespace SCIRun

#endif // builtin_h
