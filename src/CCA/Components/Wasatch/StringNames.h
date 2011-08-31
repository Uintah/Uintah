#ifndef Wasatch_StringNames_h
#define Wasatch_StringNames_h

#include <string>

namespace Wasatch{

  /**
   *  \ingroup WasatchFields
   *  \ingroup WasatchCore
   *
   *  \class  StringNames
   *  \author James C. Sutherland
   *  \date   June, 2011
   *
   *  \brief Defines names for variables used in Wasatch.
   *
   *  Note: this class is implemented in a singleton.  Access it as follows:
   *  <code>const StringNames& sName = StringNames::self();</code>
   */
  class StringNames
  {
  public:

    /**
     *  Access the StringNames object.
     */
    static const StringNames& self();

    const std::string time;

    const std::string
      xsvolcoord,  ysvolcoord,  zsvolcoord,
      xxvolcoord,  yxvolcoord,  zxvolcoord,
      xyvolcoord,  yyvolcoord,  zyvolcoord,
      xzvolcoord,  yzvolcoord,  zzvolcoord;

    // energy related variables
    const std::string
      temperature,
      e0, rhoE0,
      enthalpy,
      xHeatFlux, yHeatFlux, zHeatFlux;

    // species related variables
    const std::string
      species,
      rhoyi,
      xSpeciesDiffFlux, ySpeciesDiffFlux, zSpeciesDiffFlux,
      mixtureFraction;

    // thermochemistry related variables
    const std::string
      heatCapacity,
      thermalConductivity,
      viscosity;

    // momentum related variables
    const std::string
      xvel, yvel, zvel,
      xmom, ymom, zmom,
      pressure,
      tauxx, tauxy, tauxz,
      tauyx, tauyy, tauyz,
      tauzx, tauzy, tauzz;

  private:
    StringNames();
  };

} // namespace Wasatch

#endif // Wasatch_StringNames_h
