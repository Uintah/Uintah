/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


/********************************************************
 * VectorOperatorFunctors.h
 *
 * Author: Kurt Zimmerman (based on Wayne Witzel's Tensor stuff)
 * Scientific Computing and Imaging,
 * University of Utah
 * Copyright 2000
 */

#include <cmath>
/* Functors used by ScalarFieldOperator and ScalarParticlesOperator */

// Unary Operators
struct NaturalLogOp
{
  NaturalLogOp(){}

  template<class T>
  inline T operator()(const T& s)
  { return (T)log((double)s); }
  
};

struct ExponentialOp
{
  ExponentialOp() {}
  template<class T>
  inline T operator()(const T&  s)
  { return (T)exp((double)s); }
};

struct NoOp
{
  NoOp() {}
  template <class T>
  inline T operator()( const T& s )
  { return s; }
};

// Binary Operators
struct AddOp
{
  AddOp() {}
  template <class T>
  inline T operator()(const T& s1, const T& s2 )
  { return s1 + s2; }

  template <class T, class B>
  inline T operator()(const T& s1, const B& s2 )
  { return s1 + s2; }
};

struct SubOp
{
  SubOp() {}
  template <class T>
  inline T operator()(const T& s1, const T& s2 )
  { return s1 - s2; }
};

struct MultOp
{
  MultOp() {}
  template <class T>
  inline T operator()(const T& s1, const T& s2 )
  { return s1 * s2; }
};

struct DivOp
{
  DivOp() {}
  template <class T>
  inline T operator()(const T& s1, const T& s2 )
  { return s1 / s2; }
};

struct ScaleOp
{
  ScaleOp() {}
  template <class T>
  inline T operator()(const T& s1, const double s2 )
  { return s1 * s2; }
};

struct AverageOp
{
  AverageOp() {}
  template <class T>
  inline T operator()(const T& s1, const T& s2 )
  { return (s1 + s2)/(T)2.0; }
};


