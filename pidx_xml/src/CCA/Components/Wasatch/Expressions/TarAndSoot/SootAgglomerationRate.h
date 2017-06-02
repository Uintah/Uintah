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

#ifndef SootAgglomerationRate_h
#define SootAgglomerationRate_h

#include <expression/Expression.h>

/**
 *  @class SootFormationRate
 *  @author Josh McConnell
 *  @date   January, 2015
 *  @brief Calculates the rate of soot agglomeration in (kg soot)/(m^3-s), which is required to
 *         calculate source term for the transport equations  of soot particle number density
 *
 *  The rate of soot formation is given as
 * \f[

 *     r_{Agg\mathrm{soot}} = SA_{/mathrm{soot particle}}* P_{O_{2}}* T^{-1/2} * A_{Agg\mathrm{soot}} * exp(-E_{Agg\mathrm{soot}} / RT),
 *
 * \f]
 *  <ul>
 *
 *  <li> where \f$ SA_{/mathrm{soot particle}} \f$ is the total surface area of the soot particles
 *
 *   <li> and \f$ P_{O_{2}} \f$ is the partial pressure of dioxygen.
 *
 *  </ul>
 *
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
 * @param ySoot_:          (kg soot)/(kg total)
 * @param nSootParticle_:  (soot particles)/(kg total)
 * @param boltzConst_      (J/K) Boltzmann's constant
 * @param ca_              collision constant                    [1]
 * @param sootDensity_        (kg/m^3) soot density              [1]
 */
template< typename ScalarT >
class SootAgglomerationRate
 : public Expr::Expression<ScalarT>
{
  DECLARE_FIELDS(ScalarT, density_, temp_, ySoot_, nSootParticle_ )
  const double boltzConst_, ca_, sootDensity_;

  SootAgglomerationRate( const Expr::Tag& densityTag,
                         const Expr::Tag& tempTag,
                         const Expr::Tag& ySootTag,
                         const Expr::Tag& nSootParticleTag,
                         const double     sootDensity );
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag densityTag_, tempTag_, ySootTag_, nSootParticleTag_;
    const double sootDensity_;
  public:
    /**
     *  @brief Build a SootAgglomerationRate expression
     *  @param result the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& tempTag,
             const Expr::Tag& ySootTag,
             const Expr::Tag& nSootParticleTag,
             const double     sootDensity )
    : Expr::ExpressionBuilder( resultTag ),
      densityTag_      ( densityTag        ),
      tempTag_         ( tempTag           ),
      ySootTag_        ( ySootTag          ),
      nSootParticleTag_( nSootParticleTag  ),
      sootDensity_     ( sootDensity       )
    {}

    Expr::ExpressionBase* build() const{
      return new SootAgglomerationRate<ScalarT>( densityTag_,tempTag_,ySootTag_,nSootParticleTag_,sootDensity_ );
    }
  };

  ~SootAgglomerationRate(){}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}
  void evaluate();
};

#endif // SootAgglomerationRate_h
