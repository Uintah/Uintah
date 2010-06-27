//-- Wasatch includes --//
#include "StringNames.h"

namespace Wasatch{

  //------------------------------------------------------------------

  StringNames::StringNames() :

    time("time"),
    xcoord("x"),
    ycoord("y"),
    zcoord("z"),

    // energy related variables
    temperature("temperature"),
    e0("total internal energy"),
    rhoE0("rhoE0"),
    enthalpy("enthalpy"),
    xHeatFlux("heatFlux_x"),
    yHeatFlux("heatFlux_y"),
    zHeatFlux("heatFlux_z"),

    // species related variables
    species("species"),
    rhoyi("rhoy"),
    mixtureFraction("mixture fraction"),

    // thermochemistry related variables
    heatCapacity("heat capacity"),
    thermalConductivity("thermal conductivity"),
    viscosity("viscosity"),

    // momentum related variables
    xvel("x-velocity"),
    yvel("y-velocity"),
    zvel("z-velocity"),
    xmom("x-momentum"),
    ymom("y-momentum"),
    zmom("z-momentum"),
    pressure("pressure"),
    tauxx("tau_xx"),
    tauxy("tau_xy"),
    tauxz("tau_xz"),
    tauyx("tau_yx"),
    tauyy("tau_yy"),
    tauyz("tau_yz"),
    tauzx("tau_zx"),
    tauzy("tau_zy"),
    tauzz("tau_zz")
  {}

  //------------------------------------------------------------------

  const StringNames&
  StringNames::self()
  {
    static const StringNames s;
    return s;
  }

  //------------------------------------------------------------------

} // namespace Wasatch
