/********************************************************
 * VectorOperatorFunctors.h
 *
 * Author: Kurt Zimmerman (based on Wayne Witzel's Tensor stuff)
 * Scientific Computing and Imaging,
 * University of Utah
 * Copyright 2000
 */

#include <Core/Geometry/Vector.h>

/* Functors used by VectorFieldOperator and VectorParticlesOperator */
namespace Uintah {
using SCIRun::Vector;

struct VectorElementExtractionOp
{
  VectorElementExtractionOp(int component)
    : v_component(component) { }
  
  inline double operator()(const Vector& v)
  { return v[v_component]; }
  
  int v_component;
};

struct LengthOp
{
  LengthOp() {}
  inline double operator()(const Vector&  v)
  { return v.length(); }
};

struct VorticityOp
{
  VorticityOp() {}
  inline double operator()(const Vector&  v)
  {
    // This is something I made up.  Put the actuall thing in.
    return v.x() * v.y() + v.y() * v.z() + v.x() * v.z();
  }
};

}
