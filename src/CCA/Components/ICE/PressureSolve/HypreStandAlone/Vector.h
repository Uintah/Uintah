/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/*#############################################################################
  # Vector.h - a 1-dimensional vector array
  #============================================================================
  # Vector is a vector array. It's very close to stl_vector, with slight
  # changes in assignment operator asserts and other minor features.
  #
  # See also: Sub.
  #
  # Revision history:
  # -----------------
  # 25-DEC-2004   Added comments.
  # 03-AUG-2005   Adapted from ERS code's Tensor.h.
  ###########################################################################*/

#ifndef _VECTOR_H
#define _VECTOR_H

#include "Error.h"

/*============= Begin class Vector =============*/

template<class VAR>
class Vector : public Error {
 public:
   
   /*------------- Vector iterator -------------*/

   /*============= Begin class Vector::iterator =============*/

   class iterator {                         /* Vector iterator */
   public:
    inline iterator(VAR* index) 
      : index_(index) {}
    inline ~iterator(void) {}
    inline void operator ++ (void)
      { index_++; }
    inline void operator ++ (int)
      { index_++; }
    inline void operator -- (void)
      { index_--; }
    inline const VAR& operator * (void) const 
      { return *index_; }
    inline VAR& operator * (void)
      { return *index_; }
    inline int operator == 
      (const iterator& other) const 
      { return index_ == other.index_; }
    inline int operator !=  
      (const iterator& other) const 
      { return !(*this == other); }

   private:
     VAR* index_;
   };

   /*============= End class Vector::iterator =============*/

   /*------------- Construction, destruction, copy -------------*/

  inline Vector
    (const int& start = 0,                // Index base
     const Counter& len = 0,
     VAR* fillPtr = 0,                     // Ignored if len <= 0
     const std::string& name = "",
     const VAR& fillValue = VAR(0))
    : _name(name), _start(start),
    _len(len), _data(0), _width(3)
    /* Construction from a VectorSize */
    {
      if (_len > 0) {
        if (fillPtr) _data = fillPtr;
        else _data = scinew VAR [_len];               // Allocate data array
      }
      if (!fillPtr) {
        for (Counter i = 0; i < _len; i++)
          _data[i] = fillValue;
      }
    }
  
  inline ~Vector(void)
    /* Vector destructor */
    {
      DELETE_BRACKET(_data);
    }
  
  iterator begin(void) const 
    /* Points to first element */
    { 
      return iterator(_data);
    }

  iterator end(void) const
    /* Points to one-after-the-last element. Never access its value! */
    {
      return iterator(&(_data[_len]));
    }

    Vector(const Vector& other)
      /* Vector copy constructor. */
      : _name(other._name),
      _start(other._start),
      _len(other._len),
      _data(0),
      _width(other._width)
    {
      _data = scinew VAR [_len];
      iterator otherIter = other.begin();
      for (iterator iter = begin(); iter != end(); ++iter, ++otherIter) {
        *iter = *otherIter;
      }
    }
  
   /*------------- Element access and size operations -------------*/

  /* Const */
  inline const std::string&  getName (void)           const  { return _name; }
  inline const int&          getStart(void)           const  { return _start; }
  inline const Counter&      getLen  (void)           const  { return _len; }
  inline const VAR*          getData (void)           const  { return _data; }
  inline const Counter&      getWidth(void)           const  { return _width; }
  
  /* Modifiable */
  inline std::string& getName (void)                  { return _name; }
  inline int&         getStart(void)                  { return _start; }
  inline VAR*         getData (void)                  { return _data; }
  inline void         setWidth(const Counter& width)  { _width = width; }

  inline void resize
    (const int& start = 0,                // Index base
     const Counter& len = 0,
     VAR* fillPtr = 0,
     const std::string& name = "")
    /*
      Resizes the Vector to a scinew size.
      Creates a scinew data pointer if length is changed.
      Then previous data is destroyed.
    */
    {
      if (name != "") {
        _name = name;
      }
      Counter newLen  = len;
      if (newLen != _len) {
        _len	= newLen;
        if (_data) {
          DELETE_BRACKET(_data);
        }
        _data	= scinew VAR [_len];
      }
      if (fillPtr) _data = fillPtr;
    }
  
  std::string summary(void) const
    /* Print a summary of the object's properties. */
    {
      std::ostringstream out;
      out << "Vector ["
          << "start=" << _start 
          << ", len=" << _len 
          << "]";
      return out.str();
    }
  
  void rangeError(const int& i) const
    /* Print an error message for out-of-range access attempt. */
    {
      std::ostringstream msg;
      int high = _len + _start- 1;
      msg << "index out of range: i = " << i;
      msg << ", range = ";
      if (high < _start) {
        msg << "empty";
      } else {
        msg << "[" << _start << ".." << high << "]";
      }
      error(msg);
    }
  
  Counter find(const VAR& value) const
    /* Return the index of value, if exists in the array. If not found, 
       return _len (outside the array). */
    {
      for (Counter i = 0; i < _len; i++) 
        if (value == _data[i]) return i;
      return _len;
    }
  
  inline const VAR& operator [] (const int& i) const
    /* Access by 1D array index i, going from 0 to len-1. */
    {
      if ((i < 0) || (i >= _len)) rangeError(i);
      return _data[i];
    }
  
  inline const VAR& operator()(const int& i) const
    /* For 1D arrays, access through one subscript directly */
    {
      if ((i < 0) || (i >= int(_len))) rangeError(i);
      return _data[i];
    }

  inline VAR& operator[](const int& i)
    /* Access by 1D array index i, going from 0 to len-1 (modifyable). */
    {
      if ((i < 0) || (i >= _len)) rangeError(i);
      return _data[i];
    }

  inline VAR& operator()(const int& i)
    /* For 1D arrays, access through one subscript directly */
    {
      if ((i < 0) || (i >= int(_len))) rangeError(i);
      return _data[i];
    }

   /*------------- Assignment, comparison -------------*/
  
  Vector& operator = 
    (const Vector& other) 
    /* Assignment operator. */
    {
      _start  = other._start;
      if (_len != other._len) {
        _len  = other._len;
        DELETE_BRACKET(_data);
        _data = scinew VAR [_len];
      }
      if (_data != other._data) {
        for (Counter i = 0; i < _len; i++) _data[i] = other._data[i];
      }
      return *this;
    }
  
  Vector& operator = (const VAR& FillValue) 
    /* Assignment of a Vector to a scalar (fill with this constant value). */
    {
      for (Counter i = 0; i < _len; i++) _data[i] = FillValue;
      return *this;
    }
  
  Vector& operator |= (const VAR& value);
  Vector& operator &= (const VAR& value);
  Vector& operator ^= (const VAR& value);

  Vector& fillRandom (const VAR& a, const VAR& b);
  
  int operator == (const Vector& b) const
    /* Equality operator. */
    {
      if ((_start != b._start) ||
          (_len    != b._len   )) {
        return 0;
      }
      for (Counter i = 0; i < _len; i++) {
        if (_data[i] != b._data[i]) {
          return 0;
        }
      }
      return 1;
    }
  
  int operator != (const Vector& b) const
    /* Inequality operator */
    { 
      return !(*this == b); 
    }
  
  int operator < (const Vector& b) const
    /* Inequality operator (true if inequality holds for all entries). */
    {
      for (Counter i = 0; i < _len; i++) {
        if (_data[i] >= b._data[i]) {
          return 0;
        }
      }
      return 1;
    }

  int operator <= (const Vector& b) const
    /* Inequality operator (true if inequality holds for all entries). */
    {
      for (Counter i = 0; i < _len; i++) {
        if (_data[i] > b._data[i]) {
          return 0;
        }
      }
      return 1;
    }

  int operator >= (const Vector& b) const
    /* Inequality operator (true if inequality holds for all entries). */
    {
      for (Counter i = 0; i < _len; i++) {
        if (_data[i] < b._data[i]) {
          return 0;
        }
      }
      return 1;
    }

  int operator > (const Vector& b) const
    /* Inequality operator (true if inequality holds for all entries). */
    {
      for (Counter i = 0; i < _len; i++) {
        if (_data[i] <= b._data[i]) {
          return 0;
        }
      }
      return 1;
    }

   /*------------- Arithmetic operations: Vector/scalar -------------*/
  
  Vector operator + (const VAR& b) const
    /* Vector + scalar */
    {
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] + b;
      std::ostringstream newName;
      newName << _name << " + " << b;
      return Vector(_start, _len, news, newName.str());
    }
  
  Vector operator - (const VAR& b) const
    /* Vector - scalar */
    {
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] - b;
      std::ostringstream newName;
      newName << _name << " - " << b;
      return Vector(_start, _len, news, newName.str());
    }
  
  Vector operator * (const VAR& b) const
    /* Vector * scalar */
    {
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] * b;
      std::ostringstream newName;
      newName << _name << " * " << b;
      return Vector(_start, _len, news, newName.str());
    }
  
  Vector operator / (const VAR& b) const
    /* Vector / scalar */
    {
      if (b == VAR(0)) {
        std::ostringstream msg;
        msg << "Vector / VAR: division by 0";
        error(msg);
      }
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] / b;
      std::ostringstream newName;
      newName << _name << " / " << b;
      return Vector(_start, _len, news, newName.str());
     }
    
   /*---------- Arithmetic operations: Vector/scalar, incremental ----------*/
  
  Vector& operator += (const VAR& b) 
    /* Vector += scalar */
    {
      for (Counter i = 0; i < _len; i++) _data[i] += b;
      return *this;
    }

  Vector& operator -= (const VAR& b) 
    /* Vector -= scalar */
    {
      for (Counter i = 0; i < _len; i++) _data[i] -= b;
      return *this;
    }
  
  Vector& operator *= (const VAR& b) 
    /* Vector *= scalar */
    {
      for (Counter i = 0; i < _len; i++) _data[i] *= b;
      return *this;
    }
  
  Vector& operator /= (const VAR& b)
    /* Vector /= scalar */
    {
      if (b == VAR(0)) {
        std::ostringstream msg;
        msg << "Vector / VAR: division by 0";
        error(msg);
      }
      for (Counter i = 0; i < _len; i++) _data[i] /= b;
      return *this;
    }
    
   /*------------- Arithmetic operations: pointwise Vector/Vector -------------*/
  
  Vector operator + (const Vector& b) const
    /* Vector + Vector */
    {
      assertSize(b._start,b._len);
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] + b._data[i];
      std::ostringstream newName;
      newName << _name << " + " << b._name;
      return Vector(_start, _len, news, newName.str());
    }
  
  Vector operator - (const Vector& b) const
    /* Vector - Vector */
    {
      assertSize(b._start,b._len);
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] - b._data[i];
      std::ostringstream newName;
      newName << _name << " - " << b._name;
      return Vector(_start, _len, news, newName.str());
    }
  
  Vector operator * (const Vector& b) const
    /* Vector * Vector */
    {
      assertSize(b._start,b._len);
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = _data[i] * b._data[i];
      std::ostringstream newName;
      newName << _name << " * " << b._name;
      return Vector(_start, _len, news, newName.str());
    }
  
  
  Vector operator / (const Vector& b) const
    /* Vector / Vector */
    {
      assertSize(b._start,b._len);
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) {
        if (b._data[i] == VAR(0)) {
          std::ostringstream msg;
          msg << "Vector<int> / int: division by 0";
          error(msg);
        }
        news[i] = _data[i] / b._data[i];
      }
      std::ostringstream newName;
      newName << _name << " / " << b._name;
      return Vector(_start, _len, news, newName.str());
    }
    
  VAR innerProduct(const Vector& b) const
    /* Vector inner product. */
    {
      assertSize(b._start,b._len);
      VAR res = 0;
      for (Counter i = 0; i < _len; i++) res += _data[i]*b._data[i];
      return res;
    }
  
   /*------------- Incremental operations: Vector/Vector -------------*/

  Vector& operator += (const Vector& b) 
    /* Vector += Vector */
    {
      assertSize(b.getStart(),b.getSize());
      for (Counter i = 0; i < _len; i++) _data[i] += b._data[i];
      return *this;
    }
  
  Vector& operator -= (const Vector& b) 
    /* Vector -= Vector */
    {
      assertSize(b.getStart(),b.getSize());
      for (Counter i = 0; i < _len; i++) _data[i] -= b._data[i];
      return *this;
    }
  
  Vector& operator *= (const Vector& b) 
    /* pointwise Vector *= Vector */
    {
      assertSize(b.getStart(),b.getSize());
      for (Counter i = 0; i < _len; i++) _data[i] *= b._data[i];
      return *this;
    }
  
  Vector& operator /= (const Vector& b) 
    /* postwise Vector /= Vector */
    {
      assertSize(b.getStart(),b.getSize());
      for (Counter i = 0; i < _len; i++) {
        if (b._data[i] == VAR(0)) {
          std::ostringstream msg;
          msg << "Vector<int> / int: division by 0";
          error(msg);
        }
        _data[i] /= b._data[i];
      }
      return *this;
    }
    
   /*------------- Unary operations -------------*/
  
  Vector operator - (void) const
    /* Negative of a Vector. */
    {
      VAR* news = scinew VAR [_len];
      for (Counter i = 0; i < _len; i++) news[i] = -_data[i];
      return Vector(_start, _len, news, std::string("-" + _name));
    }
   
  VAR norm(const Counter& order) const
    /* Vector Lp norm (p=order); p=0 returns the maximum norm. */
    {
      switch (order) {
      case 0: {                   // Convension: norm(0) is the maximum norm
        VAR res = abs(_data[0]);
        for (Counter i = 1; i < _len; i++) {
          VAR tmp = abs(_data[i]);
          if (tmp > res) res = tmp;
        }
        return res;
        break;
      }
      case 1: {
        VAR res = 0;
        for (Counter i = 0; i < _len; i++) res += abs(_data[i]);
        return res;
        break;
      }
      case 2: {
        VAR res = 0, tmp;
        for (Counter i = 0; i < _len; i++) {
          tmp = abs(_data[i]);
          res += tmp*tmp;
        }
        return sqrt(res);
        break;
      }
      default: {
        std::ostringstream msg;
        msg << "Vector::Norm(): Variable order.";
        error(msg);
        return 0;
        break;
      }
      }
    }
  
  VAR max(void) const
    /* Maximum value among Vector elements. */
    {
      VAR res = _data[0];
      for (Counter i = 1; i < _len; i++)
        if (_data[i] > res) res = _data[i];
      return res;
    }
  
  VAR min(void) const
    /* Minimum value among Vector elements. */
    {
      VAR res = _data[0];
      for (Counter i = 1; i < _len; i++) 
        if (_data[i] < res) res = _data[i];
      return res;
    }
  
  int findMax(void) const
    /* Smallest index at which max() value is attained. */
    {
      int ind = 0;
      for (Counter i = 1; i < _len; i++) 
        if (_data[i] > _data[ind]) ind = i;
      return ind;
    }
  
  int findMin(void) const
    /* Smallest index at which min() value is attained. */
    {
      int ind = 0;
      for (Counter i = 1; i < _len; i++) 
        if (_data[i] < _data[ind]) ind = i;
      return ind;
    }
  
  VAR sum(void) const
    /* Sum of all elements */
    {
      VAR result = VAR(0);
      for (Counter i = 0; i < _len; i++)
        result += _data[i];
      return result;
    }
  
  VAR prod(void) const
    /* Product of all elements */
    {
      VAR result = VAR(1);
      for (Counter i = 0; i < _len; i++)
        result *= _data[i];
      return result;
    }

   /*------------- Printouts -------------*/
#ifdef WIN32
  friend std::ostream& operator << (std::ostream& os, const Vector& a);
  friend std::istream& operator >> (std::istream& os, Vector& a);
#else
  template<class T>
  friend std::ostream& operator << (std::ostream& os, const Vector<T>& a);
  template<class T>
  friend std::istream& operator >> (std::istream& os, Vector<T>& a);
#endif
  
 private:

   /*------------- Data members -------------*/
  
  std::string   _name;                    // Name of this Vector
  int           _start;                   // Index base (for all dimensions)
  Counter       _len;                     // Total number of elements
  VAR*          _data;                    // Pointer to data array
  Counter       _width;                   // Field width for printouts

   /*------------- Useful private member functions -------------*/
  
  void assertSize(const int& start, const Counter& len) const
    /* Check whether len and start are the same as for *this. */
    {
      if ((start     != _start) ||
          (len       != _len  )) {
        std::ostringstream msg;
        msg << "incompatible Vectors";
        error(msg);
      }
    }  

}; // class Vector

/*============= End class Vector =============*/

/*============= Non-member functions =============*/

template <class VAR>
std::ostream& operator << (std::ostream& os, const Vector<VAR>& a)
     /* Write the Vector to the stream os. */
{
  //  os << "(width=" << a.getWidth() << ")";
  os << "[";
  for (Counter i = 0; i < a.getLen(); i++) {
    os << std::setw(a.getWidth()) << a[i];
    if (i < a.getLen()-1) {
      os << ",";
    }
  }
  os << "]";
  return os;
}

template <class VAR>
std::istream& operator >> (std::istream& os, Vector<VAR>& a)
     /*
       Read the Vector from the stream os.
     */
{
  for (Counter i = 0; i < a.getLen(); i++) os >> a[i];
  return os;
}

template <class VAR>
Vector<VAR>
abs(const Vector<VAR>& a)/* Absolute value of a Vector. */     
{
  VAR* news = scinew VAR [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = abs(_data[i]);
  return Vector<VAR>(a.getStart(), a.getSize(), news,
                     std::string("abs(" + a.getName() + ")"));
}

template<class T>
Vector<T>
min (const Vector<T>& a,
     const Vector<T>& b)
     // Pointwise min of two vectors
{
  assert( a.getLen() == b.getLen() );
  T* news = scinew T [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) {
    news[i] = min(a.getData()[i],b.getData()[i]);
  }
  std::ostringstream newName;
  newName << "min(" << a.getName() << " , " << b.getName() << ")";
  return Vector<T>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T>
Vector<T>
max (const Vector<T>& a,
     const Vector<T>& b)
     // Pointwise max of two vectors
{
  assert( a.getLen() == b.getLen() );
  T* news = scinew T [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) {
    news[i] = max(a.getData()[i],b.getData()[i]);
  }
  std::ostringstream newName;
  newName << "min(" << a.getName() << " , " << b.getName() << ")";
  return Vector<T>(a.getStart(), a.getLen(), news, newName.str());
}

//--------------------------------------------------------------------
// Misc arithmetic operations. S is a field of values containing T.
//--------------------------------------------------------------------

//----- (Vector<T>, Vector<S>) operations

template<class T, class S>
Vector<S>
operator + (const Vector<S>& a,   /* Vector<S> = Vector<T> + Vector<S> */
            const Vector<T>& b)
{
  assert( a.getLen() == b.getLen() );
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] + b.getData()[i];
  std::ostringstream newName;
  newName << b.getName() << " + " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator - (const Vector<S>& a,   /* Vector<S> = Vector<T> - Vector<S> */
            const Vector<T>& b)
{
  assert( a.getLen() == b.getLen() );
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] - b.getData()[i];
  std::ostringstream newName;
  newName << b.getName() << " - " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator * (const Vector<S>& a,   /* Vector<S> = Vector<T> * Vector<S> */
            const Vector<T>& b)
{
  assert( a.getLen() == b.getLen() );
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] * b.getData()[i];
  std::ostringstream newName;
  newName << b.getName() << " * " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator / (const Vector<S>& a,   /* Vector<S> = Vector<T> / Vector<S> */
            const Vector<T>& b)
{
  assert( a.getLen() == b.getLen() );

  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) {
    if (b.getData()[i] == S(0)) {
      std::ostringstream msg;
      msg << "Vector / VAR: division by 0";
      a.error(msg);
    }
    news[i] = a.getData()[i] / b.getData()[i];
  }

  std::ostringstream newName;
  newName << b.getName() << " / " << a.getName();

  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

//----- (Vector<T>, Scalar<S>) operations

template<class T, class S>
Vector<S>
operator + (const Vector<T>& a,   /* Vector<S> = Vector<T> + scalar<S> */
            const S& b)
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] + b;
  std::ostringstream newName;
  newName << b << " + " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator - (const Vector<T>& a,   /* Vector<S> = Vector<T> - scalar<S> */
            const S& b)
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] - b;
  std::ostringstream newName;
  newName << b << " - " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator * (const Vector<T>& a,   /* Vector<S> = Vector<T> * scalar<S> */
            const S& b)
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] * b;
  std::ostringstream newName;
  newName << b << " * " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator / (const Vector<T>& a,   /* Vector<S> = Vector<T> / scalar<S> */
            const S& b)
{
  if (b == S(0)) {
    std::ostringstream msg;
    msg << "Vector / VAR: division by 0";
    a.error(msg);
  }
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = a.getData()[i] / b;
  std::ostringstream newName;
  newName << b << " / " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

//----- (Scalar<S> , Vector<T>) operations

template<class T, class S>
Vector<S>
operator + (const S& b,
            const Vector<T>& a)   /* Vector<S> = scalar<S> + Vector<T> */
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = b + a.getData()[i];
  std::ostringstream newName;
  newName << b << " + " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator - (const S& b,
            const Vector<T>& a)   /* Vector<S> = scalar<S> - Vector<T> */
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = b - a.getData()[i];
  std::ostringstream newName;
  newName << b << " - " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}


template<class T, class S>
Vector<S>
operator * (const S& b,
            const Vector<T>& a)   /* Vector<S> = scalar<S> * Vector<T> */
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) news[i] = b * a.getData()[i];
  std::ostringstream newName;
  newName << b << " * " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

template<class T, class S>
Vector<S>
operator / (const S& b,
            const Vector<T>& a)   /* Vector<S> = scalar<S> / Vector<T> */
{
  S* news = scinew S [a.getLen()];
  for (Counter i = 0; i < a.getLen(); i++) {
    if (a.getData()[i] == T(0)) {
      std::ostringstream msg;
      msg << "Vector / VAR: division by 0";
      a.error(msg);
    }
    news[i] = b / a.getData()[i];
  }
  std::ostringstream newName;
  newName << b << " / " << a.getName();
  return Vector<S>(a.getStart(), a.getLen(), news, newName.str());
}

#endif /* _VECTOR_H */
