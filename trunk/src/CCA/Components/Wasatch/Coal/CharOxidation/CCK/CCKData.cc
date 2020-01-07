#include "CCKData.h"
#include <stdexcept>
#include <sstream>
#include <cmath>

using CHAR::Vec;

namespace CCK{

  CCKData::CCKData( const Coal::CoalType coalType )
    : charData_ ( coalType  )
  {
    // note: neD and its standard deviation have units ln(kCal/mol)

    /* Pre-exponential factors for each of the char surface reactions Evolve in time
     * according to the following formula:
     *
     * A_i = A_i0*sqrt(f),
     *
     * where f is the fraction of active sites remaining in the char. The model implented
     * assumes the initial value of f is a lognormal distribution wrt eD. f evolves in
     * time according to the following ODE:
     *
     * df/dt = -f*aD*exp(-eD/RT).
     */

    //Assemble vector of eD values:
    size_t numel = 30;
    eDVec_.resize( numel );
    eDVec_[0]       = 1;
    eDVec_[numel-1] = 121;

    double step = (eDVec_[numel-1] - eDVec_[0])/(numel - 1);

    for(size_t i = 1; i<numel-1; ++i){
      eDVec_[i] = eDVec_[0] + i*step;
    }

    /* The mode-of-burning parameter should be set to 0.2 for oxidation
     * and 0.95 for gasification (without oxidation)
     */
    modeOfBurningParam_ = 0.2;  // for oxidation

    tau_f_ = 12; // tau/f see section 6 of [1] for more details

    ashDensity_ = 2650; // kg/m^3

    const Coal::CoalComposition coalComp = charData_.get_coal_composition();

    c_ = coalComp.get_C();

    o_ = coalComp.get_O();

    fixedCarbon_ = coalComp.get_fixed_c();

    volatiles_   = coalComp.get_vm();

    xA_ = coalComp.get_ash();

    percentC_ = c_*100; // correlations use % basis

    aFwd0_.assign(8,0.0);
    aRev0_.assign(8,0.0);

    eaFwd_.assign(8,0.0);
    eaRev_.assign(8,0.0);

    mean_neD_   = 2.8;       // log-mean of thermal annealing activation energy
    aD_         = exp(18.3); // thermal annealing frequency factor
    stdDev_neD_ = 0.46;      // std deviation of neD

    deltaMin_   = 0.0;  // minimum ash film thickness
    thetaMin_   = 0.21; // minimum ash porosity

    epsilon_    = 0.47; // char porosity

    switch(coalType){
      case Coal::Highvale:
      case Coal::Highvale_Char:
        mean_neD_   = 2.0;
        break;

      case Coal::Eastern_Bituminous:
      case Coal::Eastern_Bituminous_Char:
      case Coal::Utah_Skyline:
      case Coal::Utah_Skyline_Char_10atm:
      case Coal::Utah_Skyline_Char_12_5atm:
      case Coal::Utah_Skyline_Char_15atm:
      case Coal::NorthDakota_Lignite:
      case Coal::Gillette_Subbituminous:
      case Coal::MontanaRosebud_Subbituminous:
      case Coal::Illinois_Bituminous:
      case Coal::Kentucky_Bituminous:
      case Coal::Pittsburgh_Shaddix:
      case Coal::Pittsburgh_Bituminous:
      case Coal::Black_Thunder:
      case Coal::Shenmu:
      case Coal::Guizhou:
      case Coal::Russian_Bituminous:
      case Coal::Utah_Bituminous:
      case Coal::Illinois_No_6:
      case Coal::Illinois_No_6_Char:
        break;

      default:
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "Unsupported coal type" << std::endl
            << std::endl;
        throw std::runtime_error( msg.str() );
    }


    switch(coalType){

      case Coal::Utah_Skyline:
      case Coal::Utah_Skyline_Char_10atm:
      case Coal::Utah_Skyline_Char_12_5atm:
      case Coal::Utah_Skyline_Char_15atm:
        // These data are for char generated from Utah Skyline coal.
        // See Tables 3 & 5 from [1] for more details.
        aFwd0_[6] = 9.97e+8;  // s^-1
        eaFwd_[6] = 166.0e+3; // J/mol
        break;

      case Coal::NorthDakota_Lignite:
      case Coal::Gillette_Subbituminous:
      case Coal::MontanaRosebud_Subbituminous:
      case Coal::Illinois_Bituminous:
      case Coal::Kentucky_Bituminous:
      case Coal::Pittsburgh_Shaddix:
      case Coal::Pittsburgh_Bituminous:
      case Coal::Black_Thunder:
      case Coal::Shenmu:
      case Coal::Guizhou:
      case Coal::Russian_Bituminous:
      case Coal::Utah_Bituminous:
      case Coal::Highvale:
      case Coal::Highvale_Char:
      case Coal::Eastern_Bituminous:
      case Coal::Eastern_Bituminous_Char:
      case Coal::Illinois_No_6:
      case Coal::Illinois_No_6_Char:

        aFwd0_[6] = exp(-0.3672*c_/o_ + 21.33);
        eaFwd_[6] = 146.4e3;
        break;

      default:
        std::ostringstream msg;
        msg << __FILE__ << " : " << __LINE__ << std::endl
            << "Unsupported coal type" << std::endl
            << std::endl;
        throw std::runtime_error( msg.str() );
    }



    /* The kinetic parameters below are from [3]. Afwd0[2] is dependent on
     * pressure, so it is not set here.
     */

    a0_over_a2_ = 1.0;     // aFwd0_[0]/aFwd0_[2]   (m^3/mol)

    a1_over_a2_ = 0.05;    // aFwd0_[1]/aFwd0_[2] (m^3/mol)



    eaFwd_[0] = 25.0e+3;

    eaFwd_[1] = 117.0e+3;

    eaFwd_[2] = 133.8e+3;



    /* Correlations for kinetic parameters below are given in [4], eqs. 6.33-6.45.
     * 7 forward rate constants and 2 reverse rate constants (out of 8) are
     * used (some defined above).
     */

    aFwd0_[3] = (1.84e+3)*exp(-0.073*percentC_)*aFwd0_[6]/101325.0;   // (6.33) (Pa^-1)

    // aFwd0_[4] not set

    aFwd0_[5] = (percentC_ <= 90.6)? 0.5*aFwd0_[6]/101325.0:         // (6.35) (Pa^-1)
        (0.021*percentC_ - 1.86)*aFwd0_[6]/101325.0;         // (6.36) (Pa^-1)


    aFwd0_[7] = 7.9e-5*aFwd0_[6]/101325.0;                    // (6.38) (Pa^-1)



    aRev0_[3] = ( (3.57e-5)*percentC_ - 1.73e-3 )*aFwd0_[6]/101325.0;// (6.34) (Pa^-1)

    aRev0_[5] = ( -(3.68e-8)*percentC_ + 3.2e-6 )*aFwd0_[6]/101325.0;// (6.37) (Pa^-1)


    // activation energies in J/mol
    eaFwd_[3] = eaFwd_[6] + 54.0e3;   // (6.41)

    // eaFwd_[4] not set

    eaFwd_[5] = eaFwd_[6] - 16.0e+3;  // (6.43)


    eaFwd_[7] = eaFwd_[6] -26.0e+3; // (6.45)



    eaRev_[3] = eaFwd_[6] - 53.9e+3;  // (6.42)

    eaRev_[5] = eaFwd_[6] - 156.0e+3; // (6.44)


  }

  //---------------------------------------------------------------------------

  Vec
  CCKData::forward_rate_constants(const double& pressure,
                                  const double& pTemp,
                                  const double& saFactor ) const{

    /* calculation of aFwd0_[3] based on correlation given by eq. (10)  on
     * page 468 [3]. Correlations for 100kPa and "elevated" pressure are
     * provided. Here The limit for elevated pressure will be defined to be
     * 200kPa. For pressures between these two values aFwd0_[3] will be
     * calculated with an exponential interpolation of the two separate
     * correlations
     *
     * saFactor is the product of coefficients accounting for deactivation
     * due to thermal annealing and conversion of char.
     */

    Vec kFwd; // vector of forward rate constants
    kFwd.assign(8,0.0);

    const double eLow = 14.38 - 0.0764*percentC_; //"0.1 MPa only"
    const double eHi  = 12.22 - 0.0535*percentC_; //"elevated pressures"
    const double pLow = 1.0e+5; //
    const double pHi  = 2.0e+5;

    double expon;

    if( pressure < pHi || pressure > pLow){
      expon = (eHi - eLow)*(pressure - pLow )/(pHi - pLow) + eLow;
    }
    else if( pressure > pHi ) expon = eHi;
    else expon = eLow;

    const double aFwd0_2 = pow(expon,10.0);


    const double RT = 8.3144621 * pTemp;

    kFwd[0] = saFactor*(aFwd0_2*a0_over_a2_)*exp( -eaFwd_[0]/RT ); //1

    kFwd[1] = saFactor*(aFwd0_2*a1_over_a2_)*exp( -eaFwd_[1]/RT ); //2

    kFwd[2] = saFactor*aFwd0_2*exp( -eaFwd_[2]/RT );               //3

    kFwd[3] = saFactor*aFwd0_[3]*exp( -eaFwd_[3]/RT );             //4

    // kFwd[4] (rxn 5) is included in the gamma parameter

    kFwd[5] = saFactor*aFwd0_[5]*exp( -eaFwd_[5]/RT );             //6

    kFwd[6] = saFactor*aFwd0_[6]*exp( -eaFwd_[6]/RT );             //7

    kFwd[7] = saFactor*aFwd0_[7]*exp( -eaFwd_[7]/RT );             //8

    return kFwd;
  }

  // -----------------------------------------------------------------------------

  Vec
  CCKData::reverse_rate_constants( const double& pTemp,
                                   const double& saFactor) const{

    /* saFactor is the product of coefficients accounting for deactivation
     * due to thermal annealing and conversion of char.
     */

    Vec kRev; // vector reverse rate constants
    kRev.assign(8,0.0);


    const double RT = 8.3144621 * pTemp;

    kRev[3] = saFactor*aRev0_[3]*exp( -eaRev_[3]/RT ); //4

    kRev[5] = saFactor*aRev0_[5]*exp( -eaRev_[5]/RT ); //6

    return kRev;
  }

  // -----------------------------------------------------------------------------

  const double calc_gamma( const double& pTemp){

    // calculates gamma from eq. 43 in [4]
    double T;
    if(pTemp < 1573 || pTemp > 1073 ) T = pTemp;
    else if(pTemp >= 1573 ) T = 1573;
    else T = 1073;

    return (6.92e+2)*exp( (-5.0e+4)/(8.31446211*T) );
  }

  // -----------------------------------------------------------------------------

}// namespace CCK

/*
    [1] A. Lewis et. al. Pulverized Steam Gasification Rates of Three Bituminous
        Coal Chars in an Entrained-Flow Reactor at Pressurized Conditions. Energy
        and Fuels. 2015, 29, 1479-1493. http://pubs.acs.org/doi/abs/10.1021/ef502608y

    [2] R. Shurtz. Effects of Pressure on the Properties of Coal Char Under
        Gasification Conditions at High Initial Heating Rates. (2011). All Theses
        and Dissertations. Paper 2877. http://scholarsarchive.byu.edu/etd/2877/

    [3] S. Niksa et. al. Coal conversion submodels for design applications at elevated
        pressures. Part I. devolatilization and char oxidation. Progress in Energy and
        Combustion Science 29 (2003) 425�477.
        http://www.sciencedirect.com/science/article/pii/S0360128503000339
 */
