/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#ifndef _CUSTOMINITIALIZE_H
#define _CUSTOMINITIALIZE_H

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>

namespace Uintah {

  class ICEMaterial;

 // multiple vortices
  struct vortices{
    std::vector<Point> origin;
    std::vector<double> strength;
    std::vector<double> radius;
    ~vortices() {};
  };

// method of manufactured solutions
  struct mms{
    double A;      // mms_1
    double angle;  // mms_3
    ~mms() {};
  };

  struct gaussTemp{
    double spread_x;
    double spread_y;
    double amplitude;
    Point  origin;
    ~gaussTemp() {};
  };

  struct counterflow{
    double strainRate;
    Vector domainLength;
    IntVector refCell;
    ~counterflow() {};
  };

  // powerlaw profile + variance
  // u = U_infinity * pow( h/height )^n
  struct powerLaw{
    Vector U_infinity;            // freestream velocity
    double exponent;

    int verticalDir;              // vertical direction
    double maxHeight;             // max height of velocity profile before it's set to u_infinity
    Point gridMin;
    Point gridMax;

    // variance
    bool addVariance;             // add variance to the inlet velocity profile
    double C_mu;                  // constant
    double u_star;                // roughnes
    ~powerLaw() {};
  };

  // velocity profile according to Moser
  // "Direct Numerical Simulation of turbulent channel Flow up to Re_tau=590
  // Physics of Fluids, pp:943-945, Vol 11, Number 4, 1999
  struct DNS_Moser{
    double    dpdx;              // streamwise pressure gradient
    int       verticalDir;       // vertical direction
    Point     gridFloor;         // domain floor
    Point     gridCeil;          // domain ceiling
    vortices  vortex_vars;

    ~DNS_Moser() {};
  };

  struct customInitialize_basket{
    vortices     vortex_vars;
    mms          mms_vars;
    gaussTemp    gaussTemp_vars;
    counterflow  counterflow_vars;
    powerLaw     powerLaw_vars;
    DNS_Moser    DNS_Moser_vars;
    bool         doesComputePressure;
    std::vector<std::string> whichMethod;
  };

  void vortex( const Patch* patch,
               const Point & origin,
               const double C1,
               const double R,
               CCVariable<double> & rho_CC,
               CCVariable<Vector> & vel_CC,
               CCVariable<double> & press_CC);

  void customInitialization_problemSetup( const ProblemSpecP& cfd_ice_ps,
                                        customInitialize_basket* cib,
                                        GridP& grid);

  void customInitialization(const Patch* patch,
                            CCVariable<double>& rho_CC,
                            CCVariable<double>& temp,
                            CCVariable<Vector>& vel_CC,
                            CCVariable<double>& press_CC,
                            ICEMaterial* ice_matl,
                            const customInitialize_basket* cib);

}// End namespace Uintah
#endif
