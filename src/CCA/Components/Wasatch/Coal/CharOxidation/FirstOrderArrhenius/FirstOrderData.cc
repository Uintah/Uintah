#include "FirstOrderData.h"
#include <sstream>

namespace FOA{
FirstOrderData::FirstOrderData( const Coal::CoalType coalType )
  : coalComp_( coalType )
{
  // Converts from g/cm^2-s-atm to kg/m^2-s-Pa
  // cf = (10^-3)*(10^4)/101325
  const double cf = 9.8692e-5;
  const double c = coalComp_.get_C();
  const double o = coalComp_.get_O();

  switch( coalType ){
    case Coal::NorthDakota_Lignite:
    case Coal::Gillette_Subbituminous:
    case Coal::MontanaRosebud_Subbituminous:
    case Coal::Illinois_Bituminous:
    case Coal::Kentucky_Bituminous:
    case Coal::Pittsburgh_Shaddix:
    case Coal::Black_Thunder:
    case Coal::Shenmu:
    case Coal::Guizhou:
    case Coal::Russian_Bituminous:
    case Coal::Utah_Bituminous:
    case Coal::Pittsburgh_Bituminous:
    case Coal::Highvale:
    case Coal::Highvale_Char:
    case Coal::Eastern_Bituminous:
    case Coal::Eastern_Bituminous_Char:
    case Coal::Illinois_No_6:
    case Coal::Illinois_No_6_Char:

      /* The correlation for aH2o_ is based on data [1] from a limited range of coal
       * compositions so if c_ is outside the range ah2o_ is set to the value corresponding
       * to the closest edge of the data. */
      if( c > 0.81 || c < 0.75 ){
        aH2o_ = c > 0.81 ? 1.2826e-03 : 5.3253e-04;
      }
      else{
        aH2o_ = exp( 14.65*c - 9.3017 )*cf;
      }

      aCo2_ = exp( -c/o + 3.6505 )*cf; // (kg carbon)/(m^2-s-Pa CO2) [2]
      ea_ = 123e+3; // J/mol
      break;

      // pre-exponential factors given in [1] are in (g carbon)/(cm^2-s-atm).
      // conversion to (kg carbon)/(m^2-s-Pa) is necessary:
      // Conversion factor = (10^-3)*(10^-4)*(101325) = 1.01325e-2;

    case Coal::Utah_Skyline:
    case Coal::Utah_Skyline_Char_10atm:
    case Coal::Utah_Skyline_Char_12_5atm:
    case Coal::Utah_Skyline_Char_15atm:
      // these parameters are for char generated from Utah Skyline coal [1].
      // See [1] for more details.
      aH2o_ = 7.94*cf; // (kg carbon)/(m^2-s-Pa H2O)
      aCo2_ = 1.07*cf; // (kg carbon)/(m^2-s-Pa CO2)
      ea_   = 121.3e+3; // J/mol
      break;

    default:
      std::ostringstream msg2;
      msg2 << __FILE__ << " : " << __LINE__ << std::endl
          << "Unsupported coal type" << std::endl
          << std::endl;
      throw std::runtime_error( msg2.str() );
  }
}

}// namespace FOA

/*
 *   [1] A. Lewis et. al. Pulverized Steam Gasification Rates of Three Bituminous
 *       Coal Chars in an Entrained-Flow Reactor at Pressurized Conditions. Energy
 *       and Fuels. 2015, 29, 1479-1493. http://pubs.acs.org/doi/abs/10.1021/ef502608y
 *
 *   [2] R. Shurtz. Effects of Pressure on the Properties of Coal Char Under
 *       Gasification Conditions at High Initial Heating Rates. (2011). All Theses
 *       and Dissertations. Paper 2877. http://scholarsarchive.byu.edu/etd/2877/
 */
