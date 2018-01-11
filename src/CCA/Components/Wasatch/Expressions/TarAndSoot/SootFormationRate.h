/*
 * The MIT License
 *
 * Copyright (c) 2012-2018 The University of Utah
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
#ifndef SootFormationRate_h
#define SootFormationRate_h

#include <expression/Expression.h>

/**
 *  @class SootFormationRate
 *  @author Josh McConnell
 *  @date   January, 2015
 *  @brief Calculates the rate of soot formation in (kg soot)/(m^3)-s, which is required to
 *         calculate source terms for the transport equations  of soot 
 *         mass, tar mass, and soot particles per volume.
 *
 *  The rate of soot formation is given as
 * \f[

 *     r_{\mathrm{F,soot}} = [\mathrm{tar}] * A_{F,\mathrm{soot}} * exp(-E_{F,\mathrm{soot}} / RT),
 *
 * \f]
 *  <ul>
 *
 *  <li> where \f$ [\mathrm{tar}] \f$ is the concentration of tar.
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
 * @param density_:     system density
 * @param temp:         system temperature (K)
 * @param ytar:         (kg tar)/(kg total)
 * @param A_            (1/s) Arrhenius preexponential factor [1]
 * @param E_            (J/mol) Activation energy             [1]
 */

template< typename ScalarT >
class SootFormationRate
 : public Expr::Expression<ScalarT>
{
  DECLARE_FIELDS( ScalarT, yTar_, temp_, density_ )
  const double A_, E_;

    SootFormationRate( const Expr::Tag& yTarTag,
                       const Expr::Tag& densityTag,
                       const Expr::Tag& tempTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag yTarTag_, tempTag_, densityTag_;
  public:
    /**
     *  @brief Build a SootFormationRate expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& yTarTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& tempTag )
    : Expr::ExpressionBuilder( resultTag ),
      yTarTag_   ( yTarTag    ),
      tempTag_   ( tempTag    ),
      densityTag_( densityTag )
    {}

    Expr::ExpressionBase* build() const{
      return new SootFormationRate<ScalarT>( yTarTag_, densityTag_, tempTag_ );
    }
  };

  ~SootFormationRate(){}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}
  void evaluate();
};

#endif // SootFormationRate_h
