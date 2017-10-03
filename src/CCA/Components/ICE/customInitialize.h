/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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

#include <CCA/Components/ICE/ICEMaterial.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>

namespace Uintah {
  
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

  struct customInitialize_basket{
    vortices*     vortex_inputs;
    mms*          mms_inputs;
    gaussTemp*    gaussTemp_inputs;
    counterflow*  counterflow_inputs;
    powerLaw*     powerLaw_inputs;
    bool          doesComputePressure;
    std::string   which;
  };
  
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
