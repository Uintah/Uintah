/********************************************************
 * VectorOperatorFunctors.h
 *
 * Author: Kurt Zimmerman (based on Wayne Witzel's Tensor stuff)
 * Scientific Computing and Imaging,
 * University of Utah
 * Copyright 2000
 */

#include <math.h>
/* Functors used by ScalarFieldOperator and ScalarParticlesOperator */

// Unary Operators
struct NaturalLogOp
{
  NaturalLogOp(){}

  template<class T>
  inline double operator()(const T& s)
  { return log((double)s); }
  
};

struct ExponentialOp
{
  ExponentialOp() {}
  template<class T>
  inline double operator()(const T&  s)
  { return exp((double)s); }
};

struct NoOp
{
  NoOp() {}
  template <class T>
  inline double operator()( const T& s )
  { return double(s); }
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
  inline double operator()(const T& s1, const T& s2 )
  { return s1 - s2; }
};

struct MultOp
{
  MultOp() {}
  template <class T>
  inline double operator()(const T& s1, const T& s2 )
  { return s1 * s2; }
};

struct DivOp
{
  DivOp() {}
  template <class T>
  inline double operator()(const T& s1, const T& s2 )
  { return s1 / s2; }
};

struct ScaleOp
{
  ScaleOp() {}
  template <class T>
  inline double operator()(const T& s1, const double s2 )
  { return s1 * s2; }
};

struct AverageOp
{
  AverageOp() {}
  template <class T>
  inline double operator()(const T& s1, const T& s2 )
  { return (s1 + s2)/2.0; }
};


