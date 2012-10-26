/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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


#include "HancockMacKenzieDamage.h"
#include <cmath>

using namespace Uintah;

HancockMacKenzieDamage::HancockMacKenzieDamage(ProblemSpecP& ps)
{
  d_initialData.D0 = 0.0;
  ps->get("D0",d_initialData.D0);
  ps->require("Dc",d_initialData.Dc);
} 
         
HancockMacKenzieDamage::HancockMacKenzieDamage(const HancockMacKenzieDamage* cm)
{
  d_initialData.D0  = cm->d_initialData.D0;
  d_initialData.Dc  = cm->d_initialData.Dc;
} 
         
HancockMacKenzieDamage::~HancockMacKenzieDamage()
{
}

void HancockMacKenzieDamage::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP damage_ps = ps->appendChild("damage_model");
  damage_ps->setAttribute("type","hancock_mackenzie");

  damage_ps->appendElement("D0",d_initialData.D0);
  damage_ps->appendElement("Dc",d_initialData.Dc);
}

         
inline double 
HancockMacKenzieDamage::initialize()
{
  return d_initialData.D0;
}

inline bool
HancockMacKenzieDamage::hasFailed(double damage)
{
  if (damage > d_initialData.Dc) return true;
  return false;
}
    
double 
HancockMacKenzieDamage::computeScalarDamage(const double& plasticStrainRate,
                                            const Matrix3& stress,
                                            const double& ,
                                            const double& delT,
                                            const MPMMaterial* ,
                                            const double& ,
                                            const double& D_old)
{
  // Calculate plastic strain increment
  double epsInc = plasticStrainRate*delT;

  // Compute hydrostatic stress and equivalent stress
  double sig_h = stress.Trace()/3.0;
  Matrix3 I; I.Identity();
  Matrix3 sig_dev = stress - I*sig_h;
  double sig_eq = sqrt((sig_dev.NormSquared())*1.5);

  // Calculate the updated scalar damage parameter
  double D = D_old + (1.0/1.65)*epsInc*exp(1.5*sig_h/sig_eq);
  return D;
}
 
