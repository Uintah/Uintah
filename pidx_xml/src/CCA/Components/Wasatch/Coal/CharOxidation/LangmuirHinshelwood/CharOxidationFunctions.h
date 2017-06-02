#ifndef CharOxidationFunctions_h
#define CharOxidationFunctions_h

#include <CCA/Components/Wasatch/Coal/CharOxidation/CharBase.h>
#include <cmath>

  /**
   *  \ingroup  CharOxidation
   *  \date     December 2015
   *  \author   Josh McConnell
   *  \brief    Defines functions used in CharOxidation.h
 */

namespace CHAR{


// -----------------------------------------------------------------------------
inline double q_fun (double k1, double k2, double po2S){
  return k1*k2*pow(po2S/101325, 0.3)/(k1*pow(po2S/101325, 0.3)+k2);
}
// -----------------------------------------------------------------------------
inline double co2coratio_fun(const double& po2S,const double& tempP){
  return  0.02* pow(po2S, 0.21)*exp(3070/tempP);
}

// -----------------------------------------------------------------------------
double struc_param( const double rho,
                    const double rho_0,
                    const double e0,
                    const double s,
                    const double r );

// -----------------------------------------------------------------------------
double Do2_mix( const double& pressure,
                const double& temp,
                const double& co_ratio );

// -----------------------------------------------------------------------------
double SurfaceO2_error( double& q,           double& co2CoRatio,
                        const double po2S,   const double gaspres,
                        const double tempg,  const double tempP,
                        const double po2Inf, const double c_s,
                        const double dp,     const double k1,
                        const double k2,     const double k_a_s,
                        const CharModel chmodel );
// -----------------------------------------------------------------------------


inline double chi( const double& theta, const double& S_g,
                   const double& rho,   const double& dp )
{
  return 2*pow(theta,5.0/3.0)/(pow(S_g*rho*dp*(1-theta),2))*exp(-1.15); // [2]
}

// -----------------------------------------------------------------------------
inline double Q_number( const double& chi_value,
                        const double& k,
                        const double& temp )
{
  return 0.5*pow(k/(1.975*chi_value*pow(temp/0.032,0.5)),0.5); // [2]
}

// -----------------------------------------------------------------------------
inline double k_fun( const double& temp )
{
  return 62.37*exp(-54.0E3/8.314/temp);
}

// -----------------------------------------------------------------------------
double ka_fun( const double& theta, const double& S_g,
               const double& rho,   const double& dp,
               const double& temp );

// -----------------------------------------------------------------------------
}

#endif /* CharOxidationFunctions_h */
