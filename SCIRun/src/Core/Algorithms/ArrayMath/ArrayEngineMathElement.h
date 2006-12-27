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


#ifndef CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINEMATHELEMENT_H
#define CORE_ALGORITHMS_ARRAYMATH_ARRAYENGINEMATHELEMENT_H 1

#include <Core/Algorithms/ArrayMath/ArrayEngineMath.h>
#include <Core/Algorithms/ArrayMath/ArrayObjectFieldAlgo.h>

namespace DataArrayMath {

class Element {
  public:
    inline Element(SCIRun::Handle<SCIRunAlgo::ArrayObjectFieldElemAlgo> algo);
    inline Element();
  
    inline Element& operator=(const Element&);
    inline Vector center();
    inline Scalar size();
    inline Scalar length();
    inline Scalar area();
    inline Scalar volume();
    inline Scalar dimension();
    inline Vector normal();
  
    inline void next();
    
  private:
    SCIRun::Handle<SCIRunAlgo::ArrayObjectFieldElemAlgo> algo_;
};

inline Element::Element() :
  algo_(0)
{
}

inline Element::Element(SCIRun::Handle<SCIRunAlgo::ArrayObjectFieldElemAlgo> algo) :
  algo_(algo)
{
}

inline Element& Element::operator=(const Element& elem)
{
  algo_ = elem.algo_; return(*this);
}

inline Scalar Element::size()
{
  double s;
  algo_->getsize(s);
  return(s);
}

inline Scalar Element::length()
{
  double s;
  algo_->getlength(s);
  return(s);
}

inline Scalar Element::area()
{
  double s;
  algo_->getarea(s);
  return(s);
}

inline Scalar Element::volume()
{
  double s;
  algo_->getvolume(s);
  return(s);
}

inline Scalar Element::dimension()
{
  double s;
  algo_->getdimension(s);
  return(s);
}

inline Vector Element::center()
{
  Vector v;
  algo_->getcenter(v);
  return(v);
}

inline Vector Element::normal()
{
  Vector v;
  algo_->getnormal(v);
  return(v);
}


inline void Element::next()
{
  algo_->next();
}

inline Scalar volume(Element el)
{
  return(el.volume());
}

inline Scalar area(Element el)
{
  return(el.area());
}

inline Scalar length(Element el)
{
  return(el.length());
}

inline Scalar size(Element el)
{
  return(el.size());
}

inline Scalar dimension(Element el)
{
  return(el.dimension());
}

inline Vector center(Element el)
{
  return(el.center());
}

inline Vector normal(Element el)
{
  return(el.normal());
}

} // end namespace

#endif

