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


struct AddOp
{
  AddOp() {}
  template <class T>
  inline double operator()(const T& s1, const T& s2 )
  { return s1 + s2; }
};

struct AverageOp
{
  AverageOp() {}
  template <class T>
  inline double operator()(const T& s1, const T& s2 )
  { return (s1 + s2)/2.0; }
};

struct NoOp
{
  NoOp() {}
  template <class T>
  inline double operator()( const T& s )
  { return double(s); }
};


