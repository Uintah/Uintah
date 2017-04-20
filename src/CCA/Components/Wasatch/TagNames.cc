/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
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

//-- Wasatch includes --//
#include "TagNames.h"

namespace WasatchCore{
  
  //------------------------------------------------------------------
  
  TagNames::TagNames() :
  
  time          ( "time"    , Expr::STATE_NONE ),
  dt            ( "dt"      , Expr::STATE_NONE ),  // physical timestep size
  ds            ( "ds"      , Expr::STATE_NONE ),  // dual timestep size
  timestep      ( "timestep", Expr::STATE_NONE ),  // timestep counter
  rkstage       ( "rkstage" , Expr::STATE_NONE ),
  stableTimestep( "StableDT", Expr::STATE_NONE ),
  
  celltype("CellType", Expr::STATE_NONE),
  
  xsvolcoord( "XSVOL", Expr::STATE_NONE ),
  ysvolcoord( "YSVOL", Expr::STATE_NONE ),
  zsvolcoord( "ZSVOL", Expr::STATE_NONE ),
  xxvolcoord( "XXVOL", Expr::STATE_NONE ),
  yxvolcoord( "YXVOL", Expr::STATE_NONE ),
  zxvolcoord( "ZXVOL", Expr::STATE_NONE ),
  xyvolcoord( "XYVOL", Expr::STATE_NONE ),
  yyvolcoord( "YYVOL", Expr::STATE_NONE ),
  zyvolcoord( "ZYVOL", Expr::STATE_NONE ),
  xzvolcoord( "XZVOL", Expr::STATE_NONE ),
  yzvolcoord( "YZVOL", Expr::STATE_NONE ),
  zzvolcoord( "ZZVOL", Expr::STATE_NONE ),
  
  // energy related variables
  temperature        ( "Temperature"      , Expr::STATE_NONE ),
  absorption         ( "AbsCoef"          , Expr::STATE_NONE ),
  radiationsource    ( "RadiationSource"  , Expr::STATE_NONE ),
  radvolq            ( "radiationVolq"    , Expr::STATE_NONE ),
  radvrflux          ( "VRFlux"           , Expr::STATE_NONE ),
  enthalpy           ( "enthalpy"         , Expr::STATE_NONE ),
  xHeatFlux          ( "HeatFlux_X"       , Expr::STATE_NONE ),
  yHeatFlux          ( "HeatFlux_Y"       , Expr::STATE_NONE ),
  zHeatFlux          ( "HeatFlux_Z"       , Expr::STATE_NONE ),
  kineticEnergy      ("KineticEnergy"     , Expr::STATE_NONE ),
  totalKineticEnergy ("TotalKineticEnergy", Expr::STATE_NONE ),
  
  // species related variables
  mixMW( "Mixture_MW", Expr::STATE_NONE ),

  // tar and soot related
  tar                      ( "tar"                         , Expr::STATE_NONE ),
  soot                     ( "soot"                        , Expr::STATE_NONE ),
  sootParticleNumberDensity( "soot_particle_number_density", Expr::STATE_NONE ),
  tarOxidationRate         ( "tar_oxidation_rate"          , Expr::STATE_NONE ),
  sootOxidationRate        ( "soot_oxidation_rate"         , Expr::STATE_NONE ),
  sootFormationRate        ( "soot_formation_rate"         , Expr::STATE_NONE ),
  sootAgglomerationRate    ( "soot_agglomeration_rate"     , Expr::STATE_NONE ),

  // thermochemistry related variables
  soundspeed         ( "sound_speed"         , Expr::STATE_NONE ),
  heatCapacity       ( "heat_capacity"       , Expr::STATE_NONE ),
  cp                 ( "cp"                  , Expr::STATE_NONE ),
  cv                 ( "cv"                  , Expr::STATE_NONE ),
  thermalConductivity( "thermal_conductivity", Expr::STATE_NONE ),
  
  // NSCBC related vars
  dudx( "dudx", Expr::STATE_NONE ),
  dvdy( "dvdy", Expr::STATE_NONE ),
  dwdz( "dwdz", Expr::STATE_NONE ),
  dpdx( "dpdx", Expr::STATE_NONE ),
  dpdy( "dpdy", Expr::STATE_NONE ),
  dpdz( "dpdz", Expr::STATE_NONE ),

  // momentum related variables
  pressure  ( "pressure"  , Expr::STATE_NONE ),
  dilatation( "dilatation", Expr::STATE_NONE ),
  divrhou   ( "divrhou"   , Expr::STATE_NONE ),
  strainxx  (  "strain_xx", Expr::STATE_NONE ),
  strainxy  (  "strain_xy", Expr::STATE_NONE ),
  strainxz  (  "strain_xz", Expr::STATE_NONE ),
  strainyx  (  "strain_yx", Expr::STATE_NONE ),
  strainyy  (  "strain_yy", Expr::STATE_NONE ),
  strainyz  (  "strain_yz", Expr::STATE_NONE ),
  strainzx  (  "strain_zx", Expr::STATE_NONE ),
  strainzy  (  "strain_zy", Expr::STATE_NONE ),
  strainzz  (  "strain_zz", Expr::STATE_NONE ),
  
  // turbulence related
  turbulentviscosity( "TurbulentViscosity",            Expr::STATE_NONE ),
  straintensormag   ( "StrainTensorMagnitude",         Expr::STATE_NONE ),
  vremantensormag   ( "VremanTensorMagnitude",         Expr::STATE_NONE ),
  waletensormag     ( "WaleTensorMagnitude",           Expr::STATE_NONE ),
  dynamicsmagcoef   ( "DynamicSmagorinskyCoefficient", Expr::STATE_NONE ),
  
  // particle related variables. Note on nomenclature: all names that start with p. are particle
  // fields. Variables that start with p_ are spatial fields that are generated by a particle field
  // an example of this are the momentum sources caused by the presence of particles
  pdragx   ("p.drag_x",    Expr::STATE_NONE ),
  pdragy   ( "p.drag_y",   Expr::STATE_NONE ),
  pdragz   ( "p.drag_z",   Expr::STATE_NONE ),
  pbodyx   ( "p.body_x",   Expr::STATE_NONE ),
  pbodyy   ( "p.body_y",   Expr::STATE_NONE ),
  pbodyz   ( "p.body_z",   Expr::STATE_NONE ),
  pmomsrcx ( "p_momsrc_x", Expr::STATE_NONE ),
  pmomsrcy ( "p_momsrc_y", Expr::STATE_NONE ),
  pmomsrcz ( "p_momsrc_z", Expr::STATE_NONE ),
  presponse( "p.tau",      Expr::STATE_NONE ),
  preynolds( "p.re",       Expr::STATE_NONE ),
  pdragcoef( "p.cd",       Expr::STATE_NONE ),
  pHeatTransCoef( "p.heatTransferCoeff", Expr::STATE_NONE ),
  pHeatCapacity(  "p.heatCapacity", Expr::STATE_NONE ),
  
  // predictor related variables
  star           ( "*"),
  rhs            ( "_rhs"),
  convectiveflux ( "_convFlux_"),
  diffusiveflux  ( "_diffFlux_"),
  pressuresrc    ( "pressure_src"  , Expr::STATE_NONE ),
  divu           ( "divu"          , Expr::STATE_NONE ),
  drhodtstar     ( "drhodt*"       , Expr::STATE_NONE ),
  drhodtstarnp1  ( "drhodt*"       , Expr::STATE_NP1  ),
  drhodt         ( "drhodt"        , Expr::STATE_NONE ),
  drhodtnp1      ( "drhodt"        , Expr::STATE_NP1  ),
  unconvergedpts ( "UnconvergedPts", Expr::STATE_NONE ),
  
  // mms varden
  mms_mixfracsrc( "mms_mixture_fraction_src", Expr::STATE_NONE ),
  mms_continuitysrc("mms_continuity_src", Expr::STATE_NONE),
  mms_pressurecontsrc("mms_pressure_continuity_src", Expr::STATE_NONE),
  
  // postprocessing
  continuityresidual( "ContinuityResidual", Expr::STATE_NONE ),

  //compressible flow
  totalinternalenergy("rhoet",Expr::STATE_NONE)
  
  {}
  
  //------------------------------------------------------------------
  template<>
  const Expr::Tag TagNames::make_star(Expr::Tag someTag,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + star, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_star_rhs(Expr::Tag someTag,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someTag.name() + star + rhs, newContext);
  }

  template<>
  const Expr::Tag TagNames::make_star(std::string someName,
                                      Expr::Context newContext) const
  {
    return Expr::Tag(someName + star, newContext);
  }
  
  template<>
  const Expr::Tag TagNames::make_star_rhs(std::string someName,
                                          Expr::Context newContext) const
  {
    return Expr::Tag(someName + star + rhs, newContext);
  }
  

  //------------------------------------------------------------------

  const TagNames&
  TagNames::self()
  {
    static const TagNames s;
    return s;
  }
  
  //------------------------------------------------------------------
  
} // namespace WasatchCore
