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

#ifndef __MIE_GRUNEISEN_EOS_MODEL_H__
#define __MIE_GRUNEISEN_EOS_MODEL_H__


#include "MPMEquationOfState.h" 
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace Uintah {

  ////////////////////////////////////////////////////////////////////////////
  /*!
    \class MieGruneisenEOS
   
    \brief A Mie-Gruneisen type equation of state model
   
    \author Biswajit Banerjee \n
    C-SAFE and Department of Mechanical Engineering \n
    University of Utah \n

    Reference:
    
    Zocher, Maudlin, Chen, Flower-Maudlin, 2000,
    European Congress on Computational Methods in Applied Science 
    and Engineering, ECOMAS 2000, Barcelona)


    The equation of state is given by
    \f[
    p = \frac{\rho_0 C_0^2 \zeta 
              \left[1 + \left(1-\frac{\Gamma_0}{2}\right)\zeta\right]}
             {\left[1 - (S_{\alpha} - 1) \zeta\right]^2 } + \Gamma_0 C_p T
    \f]
    where 
    \f$ p\f$ = pressure \n
    \f$ C_0 \f$= bulk speed of sound \n
    \f$ \zeta = (\rho/\rho_0 - 1)\f$ \n
    where \f$\rho\f$ = current density \n
    \f$\rho_0\f$ = initial density \n
    \f$ E\f$ = internal energy = \f$C_p T\f$ \n
    where \f$C_p\f$ = specfic heat at constant pressure \n
    \f$T\f$ = temperature \n
    \f$\Gamma_0\f$ = Gruneisen's gamma at reference state \n
    \f$S_{\alpha}\f$ = linear Hugoniot slope coefficient 
  */
  ////////////////////////////////////////////////////////////////////////////

  class MieGruneisenEOS : public MPMEquationOfState {

    // Create datatype for storing model parameters
  public:
    struct CMData {
      double C_0;
      double Gamma_0;
      double S_alpha;
    };   

  private:

    CMData d_const;
         
    // Prevent copying of this class
    // copy constructor
    //MieGruneisenEOS(const MieGruneisenEOS &cm);
    MieGruneisenEOS& operator=(const MieGruneisenEOS &cm);

  public:
    // constructors
    MieGruneisenEOS(ProblemSpecP& ps); 
    MieGruneisenEOS(const MieGruneisenEOS* cm);
         
    // destructor 
    virtual ~MieGruneisenEOS();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    /////////////////////////////////////////////////////////////////////////
    /*! Calculate the pressure using a equation of state */
    /////////////////////////////////////////////////////////////////////////
    virtual double computePressure(const MPMMaterial* matl,
                                   const PlasticityState* state,
                                   const Matrix3& deformGrad,
                                   const Matrix3& rateOfDeformation,
                                   const double& delT);
  
    double eval_dp_dJ(const MPMMaterial* matl,
                      const double& delF,
                      const PlasticityState* state);

    // Compute pressure (option 1)
    double computePressure(const double& rho_orig,
                           const double& rho_cur);

    // Compute pressure (option 2)
    void computePressure(const double& rho_orig,
                         const double& rho_cur,
                         double& pressure,
                         double& dp_drho,
                         double& csquared);

    // Compute bulk modulus
    double computeBulkModulus(const double& rho_orig,
                              const double& rho_cur);

    // Compute strain energy
    double computeStrainEnergy(const double& rho_orig,
                               const double& rho_cur);

    // Compute density given pressure
    double computeDensity(const double& rho_orig,
                          const double& pressure);
  };

} // End namespace Uintah

#endif  // __MIE_GRUNEISEN_EOS_MODEL_H__ 
