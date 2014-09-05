/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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



/*
 *  UIvar.h: Super-class with +,-,*,/ Operators for GuiVars
 *
 *  Written by:
 *   McKay Davis
 *   Department of Computer Science
 *   University of Utah
 *   September 2004
 *
 *  Copyright (C) 1994 SCI Group
 */


#ifndef SCIRun_src_Core_GuiInterface_UIvar_h
#define SCIRun_src_Core_GuiInterface_UIvar_h 1

#include <Core/GuiInterface/GuiContext.h>
#include <Core/Util/Assert.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

template <class T>
class UiSingle
{
  GuiContext *	context_;
  T		value_;
public:
  UiSingle() : context_(0) {}
  UiSingle(const T &val) : context_(0), value_(val) {}
  UiSingle(GuiContext* ctx) : context_(ctx) {}
  UiSingle(GuiContext* ctx, const T &val) : context_(ctx), value_(val)
  {
    context_->set(value_);
  }

  virtual ~UiSingle() {}

  inline T get() {
    if (context_) context_->get(value_);
    return value_;
  }
  inline void set(const T value) {
    value_ = value;
    if (context_) context_->set(value_);
  }
  // Returns true if variable exists in TCL scope and is of type T
  inline bool valid() {
    return context_ && context_->get(value_);
  }
  
#define inline
  inline UiSingle<T> & operator= (GuiContext *);

  //make sense for -> to return context?
  inline GuiContext & operator->() ;	// unary member selction
  inline UiSingle<T> & operator()();	// unary function call

  inline UiSingle<T> & operator*();	// unary dereference
  inline UiSingle<T> & operator[](int idx);	// unary dereference

  inline const T & operator&();		// unary address of
  inline bool operator!();		// unary logical not
  inline bool operator~();		// unary bitwise ones compliment

  inline UiSingle<T> operator-();	// unary negation
  inline UiSingle<T> operator+();	// unary plus

  inline UiSingle<T> & operator++();	// unary prefix increment
  inline UiSingle<T> operator++(int); // unary postfix increment

  inline UiSingle<T> & operator--();	// unary prefix decrement
  inline UiSingle<T> operator--(int); // unary postfix decrement
  

  inline T operator+(const T & right);	// binary addition
  inline T operator-(const T & right);  // binary subtraction
  inline T operator*(const T & right);  // binary multiplication
  inline T operator/(const T & right);	// binary division

  inline T operator%(const T & right);  // binary modulus
  inline T operator<<(const T & right); // binary left shift
  inline T operator>>(const T & right); // binary right shift
  std::ostream & operator<<(std::ostream &out);
  std::istream & operator>>(std::istream &is);

  inline T operator|(const T & right);	// binary bitwise OR
  inline T operator&(const T & right);	// binary bitwise AND
  inline T operator^(const T & right);	// binary bitwise XOR
  

  inline UiSingle<T> & operator= (const T &right);
  inline UiSingle<T> & operator+=(const T & right);
  inline UiSingle<T> & operator-=(const T & right);
  inline UiSingle<T> & operator*=(const T & right);
  inline UiSingle<T> & operator/=(const T & right);

  inline UiSingle<T> & operator%=(const T & right);	// modulus assign
  inline UiSingle<T> & operator<<=(const T & right);	// left shift assign
  inline UiSingle<T> & operator>>=(const T & right);	// left shift assign


  inline UiSingle<T> & operator|=(const T & right);	// bitwise OR assign
  inline UiSingle<T> & operator&=(const T & right);	// bitwise AND assign
  inline UiSingle<T> & operator^=(const T & right);	// bitwise XOR assign


  inline bool operator< (const T & right) ;	// logical lessthan
  inline bool operator> (const T & right) ;	// logical greaterthan
  inline bool operator<=(const T & right) ;	// logical less, eq
  inline bool operator>=(const T & right) ;	// logical greater, eq
  inline bool operator==(const T & right) ;	// logical equality
  inline bool operator!=(const T & right) ;	// logical inequality
  inline bool operator&&(const T & right) ;	// logical and
  inline bool operator||(const T & right) ;	// logical or

  //  inline operator int();
  //  inline operator double();
  //inline operator string();
  inline operator T() { get(); return value_; };
#undef inline
};


#if 0

template<>
string UiSingle<string>::operator-(const string&) {
  ASSERT(0);
  return string();
}

template<>
UiSingle<string> & UiSingle<string>::operator-=(const string & right) {
  ASSERT(0);
  return *this;
}

template<>
string UiSingle<string>::operator*(const string & right) {
  ASSERT(0);
  return value_;
}

template<>
UiSingle<string> & UiSingle<string>::operator*=(const string & right) {
  ASSERT(0);
  return *this;
}


template<>
string UiSingle<string>::operator/(const string & right) {
  ASSERT(0);
  return value_;
}

template<>
inline UiSingle<string> & UiSingle<string>::operator/=(const string & right) {
  ASSERT(0);
  return *this;
}

#endif


typedef UiSingle<int> UIint;
typedef UiSingle<double> UIdouble;
//typedef UiSingle<string> UIstring;



}
#endif
