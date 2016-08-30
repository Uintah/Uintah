/*
 * This project constitutes a work of the United States Government and is not
 * subject to domestic copyright protection under 17 USC § 105.
 *
 * However, because the project utilizes code licensed from contributors and other
 * third parties, it therefore is licensed under the MIT License.
 * http://opensource.org/licenses/mit-license.php.
 *
 * Under that license, permission is granted free of charge, to any
 * person obtaining a copy of this software and associated
 * documentation files (the "Software"), to deal in the Software
 * without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the conditions that any
 * appropriate copyright notices and this permission notice are
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __PORTABLE_MIE_GRUNEISEN_EOS_TEMPERATURE_MODEL_H__
#define __PORTABLE_MIE_GRUNEISEN_EOS_TEMPERATURE_MODEL_H__

#include "PState.h"

namespace PTR {

  ////////////////////////////////////////////////////////////////////////////
  /*!
    \class PortableMieGruneisenEOSTemperature
   
    \brief A Mie-Gruneisen type equation of state model which uses temperature and density to compute the pressure
   
    \author Andrew Tonge \n
    Department of Mechanical Engineering \n
    The Johns Hopkins University \n

    Reference:
    
    Drumheller


    The equation of state is given by
    \f[
    p = \frac{\rho_0 C_0^2 \zeta 
              \left[1 + \left(1-\frac{\Gamma_0}{2}\right)\zeta\right]}
             {\left[1 - (S_{\alpha} - 1) \zeta\right]^2 + \Gamma_0 C_p T}
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

    \ Adapted from the MieGruneisenEOSEnergy. The reference curve for the equation of
    state is the principal Hugoniot. This model accounts for both the "cold energy",
    which is a function only of the density change, and the thermal energy. The thermal
    energy is referenced to the initial temperature, which should be the same temperature
    as the initial temperature for the principal Hugoniot.

  */
  ////////////////////////////////////////////////////////////////////////////

  struct CMData {
    double C_0;
    double Gamma_0;
    double S_1;
    double S_2;
    double S_3;
    double C_v;               // specific heat
    double theta_0;           // reference temperature
    double J_min;             /* unused, may be removed without warning */
    int N_points;             /* unused, may be removed without warning */
  };   

  class PortableMieGruneisenEOSTemperature{

  private:

    CMData d_const;

  public:
    // constructors
    PortableMieGruneisenEOSTemperature(); 
    PortableMieGruneisenEOSTemperature(const CMData cm); 
    PortableMieGruneisenEOSTemperature(const PortableMieGruneisenEOSTemperature* cm);
    /* PortableMieGruneisenEOSTemperature(const PortableMieGruneisenEOSTemperature cm); */
         
    // destructor 
    virtual ~PortableMieGruneisenEOSTemperature();

    CMData getEOSData() const {
      return d_const;
    }

    /////////////////////////////////////////////////////////////////////////
    /*! Calculate the pressure using a equation of state */
    /////////////////////////////////////////////////////////////////////////
	/* double computePressure(const double &rho_0, const PState &state); */

    // Calculate rate of temperature change due to compression/expansion
    double computeIsentropicTemperatureRate(const double T,
                                            const double rho_0,
                                            const double rho_cur,
                                            const double Dtrace) const;
  
    double eval_dp_dJ(const double rho_orig,
                      const double delF,
                      const PState state) const;

    // Compute pressure (option 1)
    double computePressure(const PState state) const;

	double computePressure(const double rho_orig, const double rho_cur) const;

    // Compute pressure (option 2)
    void computePressure(const double rho_orig,
                         const double rho_cur,
                         double *pressure,
                         double *dp_drho,
                         double *csquared) const;

    // Compute bulk modulus
    double computeBulkModulus(const double rho_orig,
                              const double rho_cur) const;

    double computeIsentropicBulkModulus(const double rho_0,
                                        const double rho,
                                        const double theta
                                        )  const;
    
    // Compute strain energy
    double computeStrainEnergy(const double rho_orig,
                               const double rho_cur) const;

  private:
    // Compute p for compressive volumetric deformations
    double pCompression(const double rho_orig,
                        const double eta) const;

    // Compute dp/dJ for compressive volumetric deformations
    double dpdJCompression(const double rho_orig,
                           const double eta) const;

    // Compute p for tensile volumetric deformations
    double pTension(const double rho_orig,
                    const double eta) const;

    // Compute dp/dJ for tensile volumetric deformations
    double dpdJTension(const double rho_orig,
                       const double eta) const;
  };
} // End namespace PTR

#endif  // __MIE_GRUNEISEN_EOS_TEMPERATURE_MODEL_H__
