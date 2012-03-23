/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef __HYPERELASTIC_EOS_MODEL_H__
#define __HYPERELASTIC_EOS_MODEL_H__


#include "PressureModel.h" 
#include <Core/ProblemSpec/ProblemSpecP.h>

namespace UintahBB {

  ////////////////////////////////////////////////////////////////////////////
  /*! 
    \class Pressure_Hyperelastic
    \brief Hyperelastic relation for pressure from Simo and Hughes, 1998.

    The equation of state is given by
    \f[
    p = Tr(D) K \Delta T
    \f]
    where \n
    \f$p\f$ = pressure\n
    \f$D\f$ = rate of deformation tensor\n
    \f$K\f$ = bulk modulus\n
    \f$\Delta T\f$ = time increment
  */
  ////////////////////////////////////////////////////////////////////////////

  class Pressure_Hyperelastic : public PressureModel {

  private:

    // Prevent copying of this class
    // copy constructor
    //Pressure_Hyperelastic(const Pressure_Hyperelastic &cm);
    Pressure_Hyperelastic& operator=(const Pressure_Hyperelastic &cm);

  public:
    // constructors
    Pressure_Hyperelastic(); // This constructor is used when there is
                             // no equation_of_state tag in the input
                             // file  ** WARNING **
    Pressure_Hyperelastic(ProblemSpecP& ps); 
    Pressure_Hyperelastic(const Pressure_Hyperelastic* cm);
         
    // destructor 
    virtual ~Pressure_Hyperelastic();

    virtual void outputProblemSpec(ProblemSpecP& ps);
         
    //////////
    // Calculate the pressure using a equation of state
    double computePressure(const MPMMaterial* matl,
                           const ModelState* state,
                           const Matrix3& deformGrad,
                           const Matrix3& rateOfDeformation,
                           const double& delT);
  
    double eval_dp_dJ(const MPMMaterial* matl,
                      const double& detF,
                      const ModelState* state);

    // Compute bulk modulus
    double computeBulkModulus(const ModelState* state) {return 0.0;};

    // Compute strain energy
    double computeStrainEnergy(const ModelState* state) {return 0.0;};

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
    double computeInitialBulkModulus();
    double computeBulkModulus(const double& rho_orig,
                              const double& rho_cur);

    // Compute strain energy
    double computeStrainEnergy(const double& rho_orig,
                               const double& rho_cur);

    // Compute density given pressure
    double computeDensity(const double& rho_orig,
                          const double& pressure);

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_v
        where epse_v = tr(epse)
              epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    double computeDpDepse_v(const ModelState* state)
    {
      return 0.0;
    };

    ////////////////////////////////////////////////////////////////////////
    /*! Calculate the derivative of p with respect to epse_s
        where epse_s = sqrt{2}{3} ||ee||
              ee = epse - 1/3 tr(epse) I
              epse = total elastic strain */
    ////////////////////////////////////////////////////////////////////////
    double computeDpDepse_s(const ModelState* state)
    {
      return 0.0;
    };
  };

} // End namespace Uintah

#endif  // __HYPERELASTIC_EOS_MODEL_H__ 
