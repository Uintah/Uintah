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

#ifndef SootOxidationRate_h
#define SootOxidationRate_h

#include <expression/Expression.h>


/**
 *  @class SootOxidationRate
 *  @author Josh McConnell
 *  @date   January, 2015
 *  @brief Calculates the rate of soot oxidation in (kg soot)/(m^3-s), which is required to
 *         calculate source terms for the transport equations  of soot
 *         mass, tar mass, and soot particles per volume.
 *
 *  The rate of soot formation is given as
 * \f[
 *     SA_{\mathrm{soot}}\frac{P_{\mathrm{O_{2}}}}{T^{1/2}}A_{O,\mathrm{soot}}\exp\left(\frac{-E_{O,\mathrm{soot}}}{RT}\right),
 * \f]
 *  <ul>
 *
 *  <li> where \f$ SA_{\mathrm{soot}} \f$ is the total surface area of the soot particles.
 *
 *  </ul>
 *
 *  The above equation assumes that all tar in the system considered
 *  is composed of monomers of a single "tar" molecule.
 *
 *  Source for parameters:
 *
 *  [1]    Brown, A. L.; Fletcher, T. H.
 *         "Modeling Soot Derived from Pulverized Coal"
 *         Energy & Fuels, 1998, 12, 745-757
 *
 *
 * @param density_:        system density
 * @param temp_:           system temperature
 * @param yO2_:            (kg O2)/(kg total)
 * @param ySoot_:          (kg soot)/(kg total)
 * @param mixMW_           [g/mol] molecular weight of mixture
 * @param A_               [kg*K^(0.5)*m^(-2)*Pa*s^(-1)] Arrhenius preexponential factor [1]
 * @param E_               [J/mol] Activation energy                [1]
 * @param o2MW_            [g/mol] molecular weight of O2
 * @param sootDensity_     [kg/m^3] soot density                    [1]
 */

template< typename ScalarT >
class SootOxidationRate
 : public Expr::Expression<ScalarT>
{
  DECLARE_FIELDS(ScalarT, yO2_, ySoot_, nSootParticle_,
                 mixMW_, press_, density_, temp_ )
  const double A_, E_, o2MW_, sootDensity_;

    SootOxidationRate( const Expr::Tag& yO2Tag,
                       const Expr::Tag& ySootTag,
                       const Expr::Tag& nSootParticleTag,
                       const Expr::Tag& mixMWTag,
                       const Expr::Tag& pressTag,
                       const Expr::Tag& densityTag,
                       const Expr::Tag& tempTag,
                       const double     sootDensity );
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag yO2Tag_, ySootTag_, nSootParticleTag_, mixMWTag_, pressTag_, densityTag_, tempTag_;
    const double sootDensity_;
  public:
    /**
     *  @brief Build a SootOxidationRate expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& yO2Tag,
             const Expr::Tag& ySootTag,
             const Expr::Tag& nSootParticleTag,
             const Expr::Tag& mixMWTag,
             const Expr::Tag& pressTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& tempTag,
             const double     sootDensity )
     : Expr::ExpressionBuilder( resultTag ),
        yO2Tag_          ( yO2Tag           ),
        ySootTag_        ( ySootTag         ),
        nSootParticleTag_( nSootParticleTag ),
        mixMWTag_        ( mixMWTag         ),
        pressTag_        ( pressTag         ),
        densityTag_      ( densityTag       ),
        tempTag_         ( tempTag          ),
        sootDensity_     ( sootDensity      )
     {}

    Expr::ExpressionBase* build() const{
      return new SootOxidationRate<ScalarT>( yO2Tag_, ySootTag_, nSootParticleTag_, mixMWTag_, pressTag_, densityTag_, tempTag_, sootDensity_ );
    }
  };

  ~SootOxidationRate(){}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}
  void evaluate();
};

#endif // SootOxidationRate_h
