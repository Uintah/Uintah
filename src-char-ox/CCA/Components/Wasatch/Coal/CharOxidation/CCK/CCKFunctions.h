
#ifndef CCKFunctions_h
#define CCKFunctions_h

#include <cmath>
#include <boost/multi_array.hpp>
#include <CCA/Components/Wasatch/Coal/CoalData.h>
#include <CCA/Components/Wasatch/Coal/CharOxidation/CharData.h>

namespace CCK{

/**
 *  \author   Josh McConnell
 *  \date     June 2015
 *
 */

/******************************************************************************/
//  The ordering of vectors for molecular weights, mass fractions, etc. is
//  assumed to be:
//  [ CO2, CO, O2, H2, H2O ].
//
//  It is extremely important that this is the case since several of the
//  functions below rely on this ordering
/******************************************************************************/

// -----------------------------------------------------------------------------
  std::map<size_t, size_t> speciesSolveMap(const CHAR::Vec& ppInf);
// -----------------------------------------------------------------------------
  double thieleModulus(const double& prtDiam,
                       const double& coreDensity,
                       const double& stoichCoeff,
                       const double& rxnOrder,
                       const double& rxnRate,
                       const double& dEff_i,
                       const double& P_i,
                       const double& prtTemp);

// -----------------------------------------------------------------------------
  double effectivenessFactor( const double& prtDiam,
                              const double& charDens,
                              const double& stoichCoeff,
                              const double& effRxnOrder,
                              const double& rxnRate,
                              const double& dEff_i,
                              const double& P_i,
                              const double& prtTemp );
// -----------------------------------------------------------------------------
  double massTransCoeff( const double& nSh,
                         const double& dEff_i,
                         const double& theta,
                         const double& prtDiam,
                         const double& coreDiam,
                         const double& mTemp,
                         const double& nuOA );
// -----------------------------------------------------------------------------

  inline double binaryDiffCoeff(const double& mw1,
                                const double& mw2,
                                const double& v1,
                                const double& v2,
                                const double& press,
                                const double& temp )
  {
      /*
       * Calculation of binary diffusion coefficients (in m^2/s) using the Fuller-
       * Schettler-Giddings correlation. See pages 68-69 of Multicomponent Mass Transfer
       * (Taylor and Krishna) for further details.
       *
       */

  //  const double cst = 1.013e-2;
  //  const double c13 = 1.0/3.0;

    return 1.013e-2 * pow( temp, 1.75 ) * sqrt( 1/mw1 + 1/mw2 ) /
           ( press * pow( pow( v1, 1.0/3.0 ) + pow( v2, 1.0/3.0 ), 2.0) ) ;
}

// -----------------------------------------------------------------------------

  inline double rxnRateSurf( const double& K1,
                             const double& K2,
                             const double& K3,
                             const double& P1,
                             const double& P2 ){
      /*
       * Calculates a surface reaction rate (eqs. 6.14-15 of [1]) for char
       * gasification reactions.
       *
       * \param K1 & K2: reaction constants (see eqs. 6.14-6.23 of [1])
       * \param P2 & P3: partial pressures
       *
       */
      return (K1*P1)/(1.0 + K2*P1 + K3*P2);
}
// -----------------------------------------------------------------------------

  inline double calc_omega( const double& kr4,
                            const double& kr6,
                            const double& kf7,
                            const double& gamma,
                            const CHAR::Vec&    Ps,
                            const CHAR::Vec&    dEff ){
      /*
       * Calculates omega from eq. 6.20 of [1].
       *
       * \param kri  : reverse rate constant of rxn i
       * \param kfi  : forward rate constant of rxn i
       * \param gamma: kf7/kf5
       * \param Ps   : vector of species partial pressures at char surface
       * \param dEff : vector of effective diffusivities
       */

      // Assumed ordering: [ CO2, CO, O2, H2, H2O ].
      //                   [ 0,   1,  2,  3,  4   ].

      return   kf7
             + gamma*kr4*( Ps[1] + 2*Ps[0]*dEff[0]/dEff[1] + Ps[3]*dEff[4]/dEff[1] )
             + kr6*( Ps[3] + Ps[4]*dEff[4]/dEff[3] );
}
// -----------------------------------------------------------------------------

  inline double calc_K2( const double& kf4,
                         const double& kr4,
                         const double& gamma,
                         const double& omega,
                         const CHAR::Vec&    dEff ){
      /*
       * Calculates K2 from eq. 6.18 of [1].
       *
       * \param kri  : reverse rate constant of rxn i
       * \param kfi  : forward rate constant of rxn i
       * \param gamma: kf7/kf5
       * \param omega: from eq. 6.20 of [1]
       * \param dEff : vector of effective diffusivities
       */

      // Assumed ordering: [ CO2, CO, O2, H2, H2O ].
      //                   [ 0,   1,  2,  3,  4   ].

      return fmax( gamma*( kf4 - 2*kr4*( dEff[0]/dEff[1] ) )/omega,
                   0.0);

}
// -----------------------------------------------------------------------------

  inline double calc_K3( const double& kr4,
                         const double& kf6,
                         const double& kr6,
                         const double& gamma,
                         const double& omega,
                         const CHAR::Vec&    dEff ){
      /*
       * Calculates K3 from eq. 6.18 of [1].
       *
       * \param kri  : reverse rate constant of rxn i
       * \param kfi  : forward rate constant of rxn i
       * \param gamma: kf7/kf5
       * \param omega: from eq. 6.20 of [1]
       * \param dEff : vector of effective diffusivities
       */

      // Assumed ordering: [ CO2, CO, O2, H2, H2O, CH4 ].
      //                   [ 0,   1,  2,  3,  4,   5   ].

      return fmax( ( kf6 - kr6*( dEff[4]/dEff[3] )
                   - gamma*kr4*( dEff[4]/dEff[1] ) )/omega,
                   0.0);

}

// -----------------------------------------------------------------------------

  inline double reactionOrder( const double& K1,
                               const double& K2,
                               const double& P1,
                               const double& P2 ){
      /*
       * Calculates an effective reaction order (eqs. 6.22-23 of [1]) for char
       * gasification reactions.
       *
       * \param K1 & K2: reaction constants (see eqs. 6.14-6.23 of [1])
       * \param P2 & P3: partial pressures
       */
      return 1.0 - (K1*P1)/(1.0 + K1*P1 + K2*P2);
}
// -----------------------------------------------------------------------------
}//namespace CCK

#endif /* CCKFunctions_h */
