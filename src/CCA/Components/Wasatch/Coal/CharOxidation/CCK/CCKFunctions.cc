#include "CCKFunctions.h"
#include "CCKData.h"

#ifndef PI
#define PI 3.1415926535897932384626433832795
#endif

using CHAR::Vec;

namespace CCK{

// -----------------------------------------------------------------------------

  double thieleModulus(const double& prtDiam,
                       const double& coreDensity,
                       const double& stoichCoeff,
                       const double& rxnOrder,
                       const double& rxnRate,
                       const double& dEff_i,
                       const double& p_i,
                       const double& prtTemp){
    /*
     * This function calculates the Thiele modulus
     *
     * \param prtDiam    : particle diameter
     * \param corDensity : appparent density of carbonacious core of the particle
     * \param soichCoeff : stoichiometric coefficient of species
     * \param effRxnOrder: effective reaction order
     * \param rxnRate    : rate of reaction char with species i (s^-1)
     * \param dEff_i     : effective diffusivity of species i in mixture (m^2/s)
     * \param p_i        : partial pressure of species i (Pa)
     * \param prtTemp    : particle temperature (K)
     */

       // molar concentration of species assuming ideal gas law
       const double c_i   = fmax( p_i/( 8.3144621*prtTemp ), 0.0 );

       // molar concentration of char
       const double c_char = coreDensity*1000.0/12.01;


       const double sqrtTerm =  c_char*stoichCoeff
                                *(rxnOrder+1)*(rxnRate) /
                                (2*dEff_i*c_i);

       if( sqrtTerm < 0 ){
         std::ostringstream msg;
         msg << __FILE__ << " : " << __LINE__  <<std::endl
             << "Thiele modulus is imaginary"  <<std::endl
             << "Rxn Rate      " <<rxnRate     <<std::endl
             << "Rxn order     " <<rxnOrder    <<std::endl
             << "pSurf_i       " <<p_i         <<std::endl
             << "char Density  " <<coreDensity <<std::endl
             << std::endl;
         throw std::runtime_error( msg.str() );

     }
       return prtDiam/2*sqrt( sqrtTerm );
  }

// -----------------------------------------------------------------------------

  double effectivenessFactor( const double& prtDiam,
                              const double& coreDensity,
                              const double& stoichCoeff,
                              const double& rxnOrder,
                              const double& rxnRate,
                              const double& dEff_i,
                              const double& p_i,
                              const double& prtTemp ){

    /*
     * This function calculates the effectiveness factor
     *
     * \param prtDiam    : particle diameter
     * \param charDens   : appparent char density
     * \param soichCoeff : stoichiometric coefficient of species (mole i)/(kg char)
     * \param effRxnOrder: effective reaction order
     * \param rxnRate    : rate of reaction char with species i (s^-1)
     * \param dEff_i     : effective diffusivity of species i in mixture (m^2/s)
     * \param p_i        : partial pressure of species i (Pa)
     * \param prtTemp    : particle temperature (K)
     */

    if(p_i > 0.0 ){

    // Thiele modulus
    const double phi =   thieleModulus( prtDiam,     coreDensity,
                                        stoichCoeff, rxnOrder,
                                        rxnRate,     dEff_i,
                                        p_i,         prtTemp );

    // effectiveness factor
    return ( 1.0/tanh(3.0*phi) - 1.0/(3.0*phi) )/phi;
    }
    else return 0.0;
  }
// -----------------------------------------------------------------------------



  std::map<size_t, size_t> speciesSolveMap( const Vec& pInf )
  {
    /* This function returns a map for partial pressures to be solved for.
     */
    std::map<size_t, size_t> sm;

    //CO2
    size_t i = 0;
    if(pInf[0] + pInf[2] > 0.0){
      sm[i] = 0;
      ++i;
    }

    //CO
    if(pInf[0] + pInf[2] + pInf[4] > 0.0){
      sm[i] = 1;
      ++i;
    }

    //O2
    if(pInf[2] > 0.0){
      sm[i] = 2;
      ++i;
    }

    //H2
    if(pInf[3] + pInf[4] > 0.0){
      sm[i] = 3;
      ++i;
    }

    //H2O
    if( pInf[4] > 0.0){
      sm[i] = 4;
      ++i;
    }
    return sm;
  }
// -----------------------------------------------------------------------------
  double CCKData::
  massTransCoeff( const double& nSh,
                  const double& dEff_i,
                  const double& theta,
                  const double& prtDiam,
                  const double& coreDiam,
                  const double& mTemp,
                  const double& nu ) const{
      /*
       * Calculates the mass transfer coefficient from eq. 2.8 of [1]
       * in mole/(Pa-m^2-s)
       *
       * \param nSh  : Sherwood number
       * \param theta  : porosity of ash film
       * \param dEff_i  : Effective diffusivity of species i (m^2/s)
       * \param prtDiam: Particle diameter (m)
       * \param coreDiam: diameter of carbonacious core (m)
       * \param mTemp : mean of gas and particle temperatures (K)
       * \param nuOA : Overall stoichiometric coefficient of species i
       *
       *
       * absolute value of nuOA used because q[i]/mtc[i] needs to be positive if
       * species i is consumed (q[i]>0) and negative if it is produced (q[i]<0).
       * Otherwise, q[i]/mtc[i] would always be greater than or equal to zero.
       */


      const double delta = fmax(0.5*( prtDiam - coreDiam ), deltaMin_ ); // ash film thickness

      return  nSh * dEff_i * pow(theta, 2.5) * prtDiam
              / ( 8.3144621 * mTemp*fabs(nu)
                  * (nSh * delta * coreDiam
                     + pow(theta, 2.5) * pow(coreDiam, 2.0) ) );

    }

  double CCKData::
  co2_co_ratio( const double prtTemp,
                const double csO2  )
  const {

    return 5e-2*exp( (eaFwd_[2]-eaFwd_[1])/(8.3144621*prtTemp) );

    }
// -----------------------------------------------------------------------------
// the mother of all functions (within this file)
// -----------------------------------------------------------------------------
  Vec
  CCKData::char_consumption_rates( const CHAR::Array2D& binDCoeff,
                                   const Vec&     x,
                                   const Vec&     pSurf,
                                   const Vec&     pInf,
                                   const Vec&     kFwd,
                                   const Vec&     kRev,
                                   const double&  pressure,
                                   const double&  prtDiam,
                                   const double&  prtTemp,
                                   const double&  mTemp,
                                   const double&  coreDiam,
                                   const double&  coreDens,
                                   const double&  theta,
                                   const double&  epsilon_,
                                   const double&  tau_f_,
                                   const double&  nSh,
                                   const double&  gamma,
                                   double&        co2_Co_ratio,
                                   Vec&           pError ) const
  {
    /*
     * This function calculates the consumption rates of char by each species considered.
     * The units of the results are in (kg char consumed)/sec .
     */

    Vec Dmix = CHAR::effective_diff_coeffs(binDCoeff, x, 1, 1);
    Vec dEff = CHAR::effective_diff_coeffs(binDCoeff, x, epsilon_, tau_f_);

    // [ CO2, CO, O2, H2, H2O ]
    double omega = calc_omega(kFwd[3],kFwd[5],kFwd[6],gamma,pSurf,dEff);
    double K1a = kFwd[6]*kFwd[3]/omega;
    double K1b = kFwd[6]*kFwd[5]/omega;
    double K2  = calc_K2( kFwd[3],kRev[3],gamma,omega,dEff );
    double K3  = calc_K3( kRev[3],kFwd[5],kRev[5],gamma, omega,dEff );

    const double csO2 = pSurf[2]/(8.3144621*prtTemp);

    // CO2-CO ratio [3].
    co2_Co_ratio = co2_co_ratio( prtTemp, csO2 );

    // moles of CO2 produced per moles of CO2 + CO produced from oxidation
    const double fracCo2 = co2_Co_ratio/( 1.0 + co2_Co_ratio );

    /*
     * The r_i_Surf below are consumption rates in [1/s]
     *  Results will be non-negative
     *
     * Global gasification reactions by CO2 and H2O:
     *
     * C(s) + H2O --> CO + H2
     * C(s) + CO2 --> 2CO
     *
     * Global oxidation reaction:
     *
     * C(s) +0.5(1 + f)*O2 --> f*CO2 + (1 - f)*CO
     */

    // CO2, H2O, and H2 [1]
    double rCo2_surf = rxnRateSurf( K1a, K2, K3, pSurf[0], pSurf[4] );
    double rH2o_surf = rxnRateSurf( K1b, K3, K2, pSurf[4], pSurf[0] );
    double rH2_surf  = kFwd[7]*pSurf[3];

    // O2 [2] (used by [1])
    double rO2_surf  =   ( kFwd[0] * kFwd[1] * pow( pSurf[2], 2 )
                       +   kFwd[0] * kFwd[2] * pSurf[2] )
                       / ( kFwd[0] * pSurf[2] + 0.5*kFwd[2] );

    /*
     * Stoichiometric coefficients for oxidation and gasification in
     * (mole i reacted)/(mole char reacted). These are only for reactants.
     * By default, values of nu are set to 1.0 instead of zero because they
     * are used in the calculation of mass transfer coefficients (and
     * nowhere else).
     */

    Vec nu; nu.assign(5, 1.0);
    nu[0] = 1.0;                  // CO2
    nu[2] = 0.5*( 1.0 + fracCo2 );// O2
    // CO is not a reactant
    nu[3] = 2.0;                  // H2
    nu[4] = 1.0;                  // H2O
    // CH4 is not a reactant

    /* rC calculated below have units
     * (moles char consumed)/(moles char consumed by rxn with i) per second
     */
    Vec rC; rC.clear();
    rC.assign(pSurf.size(), 0.0);
//============================================================================//
/************************* Calculations for oxidation *************************/
//============================================================================//
    if( pSurf[2] > 0.0){

      // effective rxn rate order for char oxidation using eq. 14 from [3]
      double nO2_surf =
                        ( kFwd[0]*kFwd[1]*pow( pSurf[2], 2.0 )
                         + kFwd[2]*( kFwd[1]*pSurf[2] + 0.5*kFwd[2] ) )
                      /
                        ( ( kFwd[0]*pSurf[2] + 0.5*kFwd[2] )
                         *( kFwd[1]*pSurf[2] + kFwd[2] ) );

      // effectiveness factor for oxidation (includes correction factor)
      double effFactorO2
          =  effectivenessFactor( prtDiam,  coreDens,  nu[2],
                                  nO2_surf,  rO2_surf, dEff[2],
                                  pSurf[2], prtTemp );

     rC[2] = effFactorO2 * rO2_surf;
    }
    else rC[2] = 0.0;

//============================================================================//
/******************** Calculations for gasification by CO2 ********************/
//============================================================================//
    if( pSurf[0] > 0.0){

      // effective rxn rate order of CO2 [1]
      double nCo2_surf = reactionOrder( K2, K3, pSurf[0], pSurf[4] );

      // effectiveness factor for gasification by CO2
      double effFactorCo2
          =  effectivenessFactor( prtDiam,   coreDens,  nu[0],
                                  nCo2_surf, rCo2_surf, dEff[0],
                                  pSurf[0], prtTemp );

      rC[0] = effFactorCo2 * rCo2_surf;
    }
    else rC[0] = 0.0;

//============================================================================//
/******************** Calculations for gasification by H2O ********************/
//============================================================================//
    if( pSurf[4]>0.0){
      // effective rxn rate order H2O [1]
      double nH2o_surf = reactionOrder( K3, K2, pSurf[4], pSurf[0] );

      // effectiveness factor for gasification by H2O
      double effFactorH2o
          =  effectivenessFactor( prtDiam,   coreDens,  nu[4],
                                  nH2o_surf, rH2o_surf, dEff[4],
                                  pSurf[4], prtTemp );

      rC[4] = effFactorH2o * rH2o_surf;
    }
    else rC[4] = 0.0;

//============================================================================//
/******************** Calculations for gasification by H2 *********************/
//============================================================================//
    if( pSurf[3]>0.0){
      // effective rxn rate order H2 [1]
      double nH2_surf = 1.0;

      // effectiveness factor for gasification by H2
      double effFactorH2
          =  effectivenessFactor( prtDiam,  coreDens,  nu[3],
                                  nH2_surf,  rH2_surf, dEff[3],
                                  pSurf[3], prtTemp );

      rC[3] = effFactorH2 * rH2_surf;
    }
    else rC[3] = 0.0;
//============================================================================//

    // r is the vector of species rxn rates in [s^-1]
    Vec r; r.assign(pSurf.size(), 0.0);

    r[0] = ( rC[0] - fracCo2*rC[2] );
    r[1] = -( 2.0*rC[0] + (1.0 - fracCo2)*rC[2] + rC[4] );
    r[2] = rC[2];
    r[3] = rC[3] - 2*rC[4]; // eq 6.29 of [1] says this should be rC[3] - nu_H2o/nu_H2*rC4. I'm fairly certain 6.29 is incorrect.
    r[4] = rC[4];
    r[5] = 0.5*rC[3];

    // (moles of char)/particle /( exterior particle surface area)
    const double sc_Char = coreDens * prtDiam/6.0 * 1000.0/12.01;

    // depletion fluxes in (moles i)/m^2-s
    Vec q; q.clear();
    for( size_t i = 0; i < r.size(); ++i ){
      q.push_back( r[i]*sc_Char );
    }

    assert(rC[1] == 0.0);

    // total rate of char consumption
//    const double r_tot = std::accumulate( rC.begin(), rC.end(), 0.0 );


//============================================================================//
//---------------------------                    ------------------------------/
/**************************** Error Calculations ******************************/
//---------------------------                    ------------------------------/
//=============================================================================/


/*=======================   CO2, CO, H2, H2O, CH4    =========================*/
    for( size_t i = 0; i < nu.size(); ++i ){

      if( i == 2 ) continue;

      const double mtc = massTransCoeff( nSh, Dmix[i], theta,
                                         prtDiam, coreDiam, mTemp,
                                         nu[i] );

      const double ppCalc = pInf[i] - q[i]/mtc;

      if( i == 5){ pError[i] = 0;                  }
      else       { pError[i] = (pSurf[i] - ppCalc);}

    }
/*=======================             O2             =========================*/
     const double mtc = massTransCoeff( nSh, Dmix[2], theta,
                                           prtDiam, coreDiam, mTemp,
                                           nu[2] );

     const double xi = (fracCo2 - 1.0)/(fracCo2 + 1.0);

     /*
      * evaluation below based on eq. 6.9 of [1]. As xi --> 0
      * eq. 6.9 --> eq. 6.10. Note: max(xi) = 0;
      */
     const double ppCalc = xi < 0.0?
                  // eq. 6.9
                  pressure * ( 1.0 - ( 1.0 - xi * pInf[2]/(pressure) )
                           * exp(xi * q[2]/(mtc * pressure) ) )/xi :
                  // eq. 6.10
                  pInf[2] - q[2]/mtc;

     pError[2] = (pSurf[2] - ppCalc);

/*============================================================================*/

     /* it is convenient to have the rates in (kg char consumed by i)/s,
      * primarily for enthalpy of rxn calculations.
      */

     for(size_t i = 0; i < rC.size(); ++i){
       rC[i] = -rC[i]*coreDens*pow(prtDiam, 3.0);
     }
    return rC;
  }
} //namespace CCK

/*
    [1] R. Shurtz. Effects of Pressure on the Properties of Coal Char Under
        Gasification Conditions at High Initial Heating Rates. (2011). All Theses
        and Dissertations. Paper 2877. http://scholarsarchive.byu.edu/etd/2877/

    [2] S. Niksa et. al. Coal conversion submodels for design applications at elevated
        pressures. Part I. devolatilization and char oxidation. Progress in Energy and
        Combustion Science 29 (2003) 425�477.
        http://www.sciencedirect.com/science/article/pii/S0360128503000339

    [3] J. Hong et. al.Predicting effectiveness factor for mth-order and Langmuir rate
        equations in spherical coordinates. Chemical Engineering Department, Brigham
        Young University, Provo, UT 84602.

 */


