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

#ifndef Wasatch_TagNames_h
#define Wasatch_TagNames_h

#include <string>
#include <expression/Expression.h>

namespace WasatchCore{
  
  /**
   *  \ingroup WasatchFields
   *  \ingroup WasatchCore
   *
   *  \class  TagNames
   *  \author James C. Sutherland, Tony Saad
   *  \date   June, 2010
   *
   *  \brief Defines tags for variables used in Wasatch.
   *
   *  Note: this class is implemented in a singleton.  Access it as follows:
   *  <code>const TagNames& tagName = TagNames::self();</code>
   */
  class TagNames
  {
  public:
    
    /**
     *  Access the TagNames object.
     */
    static const TagNames& self();
    
    const Expr::Tag time, dt, ds, timestep, rkstage, stableTimestep;
    
    const Expr::Tag celltype;
    
    const Expr::Tag
    xsvolcoord,  ysvolcoord,  zsvolcoord,
    xxvolcoord,  yxvolcoord,  zxvolcoord,
    xyvolcoord,  yyvolcoord,  zyvolcoord,
    xzvolcoord,  yzvolcoord,  zzvolcoord;
    
    // energy related variables
    const Expr::Tag
    temperature,
    absorption,
    radiationsource, radvolq, radvrflux,
    enthalpy,
    xHeatFlux, yHeatFlux, zHeatFlux,
    kineticEnergy, totalKineticEnergy;
    
    // species related variables
    const Expr::Tag
    mixMW;
    
    // tar and soot related
    const Expr::Tag
    tar, soot,         sootParticleNumberDensity,
    tarOxidationRate,  sootOxidationRate,
    sootFormationRate, sootAgglomerationRate;

    // thermochemistry related variables
    const Expr::Tag
    soundspeed,
    heatCapacity, cp, cv,
    thermalConductivity,    
    dudx, dvdy, dwdz,
    dpdx, dpdy, dpdz;
    
    // momentum related variables
    const Expr::Tag
    xvel, yvel, zvel,
    xmom, ymom, zmom,
    pressure, dilatation, divrhou,
    strainxx, strainxy, strainxz,
    strainyx, strainyy, strainyz,
    strainzx, strainzy, strainzz;
    
    // turbulence related variables
    const Expr::Tag
    turbulentviscosity,
    straintensormag, vremantensormag,
    waletensormag, dynamicsmagcoef;

    // particle related
    const Expr::Tag
    pdragx, pdragy, pdragz,
    pbodyx, pbodyy, pbodyz,
    pmomsrcx, pmomsrcy, pmomsrcz,
    presponse, preynolds, pdragcoef,
    pHeatTransCoef, pHeatCapacity;

    // varden
    const std::string
    star, rhs;
    
    const std::string
    convectiveflux, diffusiveflux;
    
    const Expr::Tag
    pressuresrc,
    divu, drhodtstar, drhodtstarnp1,
    drhodt,drhodtnp1,unconvergedpts;
    
    // mms varden
    const Expr::Tag
    mms_mixfracsrc, mms_continuitysrc,
    mms_pressurecontsrc;
    
    // postprocessing-related tags
    const Expr::Tag
    continuityresidual;
    
    // compressible flow
    const Expr::Tag
    totalinternalenergy;

    template < typename T >
    const Expr::Tag make_star(T someTag,
                              Expr::Context newContext=Expr::STATE_NONE) const;
    template < typename T >
    const Expr::Tag make_star_rhs(T someTag,
                                  Expr::Context newContext=Expr::STATE_NONE) const;
  private:
    TagNames();
  };
  
} // namespace WasatchCore

#endif // Wasatch_TagNames_h
