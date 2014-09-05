/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

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
#include <Core/Persistent/Persistent.h>
#include <Core/share/share.h>

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
typedef ScalarType<unsigned char>   UChar;
typedef ScalarType<short>  Short;
typedef ScalarType<unsigned short>  UShort;
typedef ScalarType<int>    Int;
typedef ScalarType<unsigned int>    UInt;
typedef ScalarType<long long> LongLong;
typedef ScalarType<float>  Float;
typedef ScalarType<double> Double;

SCICORESHARE inline void Pio(Piostream& stream, Char& d)  {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, UChar& d) {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Short& d) {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, UShort& d){Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Int& d)   {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, UInt& d)  {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Float& d) {Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, Double& d){Pio(stream,d.val_);}
SCICORESHARE inline void Pio(Piostream& stream, LongLong& d){Pio(stream,d.val_);}

inline const string find_type_name(Char*)  {return find_type_name((char *)0);}
inline const string find_type_name(UChar*) {return find_type_name((unsigned char *)0);}
inline const string find_type_name(Short*) {return find_type_name((short *)0);}
inline const string find_type_name(UShort*){return find_type_name((unsigned short *)0);}
inline const string find_type_name(Int*)   {return find_type_name((int *)0);}
inline const string find_type_name(UInt*)  {return find_type_name((unsigned int *)0);}
inline const string find_type_name(Float*) {return find_type_name((float *)0);}
inline const string find_type_name(Double*){return find_type_name((double *)0);}
inline const string find_type_name(LongLong*){return find_type_name((double *)0);}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424
#endif

template<class T> bool is_scalar() { return false; }
template<> inline bool is_scalar<char>() { return true; }
template<> inline bool is_scalar<unsigned char>() { return true; }
template<> inline bool is_scalar<short>() { return true; }
template<> inline bool is_scalar<unsigned short>() { return true; }
template<> inline bool is_scalar<int>() { return true; }
template<> inline bool is_scalar<float>() { return true; }
template<> inline bool is_scalar<double>() { return true; }

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif

} // end namespace SCIRun

#endif // builtin_h
