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

#ifndef TarOxidationRate_h
#define TarOxidationRate_h

#include <expression/Expression.h>

#ifndef GAS_CONSTANT
#  define GAS_CONSTANT 8.3144621 // J/(mol*K)
#endif

/**
 *  @class TarOxidationRate
 *  @author Josh McConnell
 *  @date   January, 2015
 *  @brief Calculates the rate of tar oxidation in (kg tar)/(m^3-s), which is required to
 *         calculate source terms for the transport equations  of soot
 *         mass, tar mass, and soot particles per volume.
 *
 *  The rate of tar oxidation is given as
 * \f[

 *     r_{\mathrm{O,tar}} = [\mathrm{tar}] [O_{2}]* A_{O,\mathrm{tar}} * exp(-E_{O,\mathrm{tar}} / RT),
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
 *
 * @param density_:        system density
 * @param temp_:           system temperature
 * @param yO2_:            (kg O2)/(kg total)
 * @param yTar_:           (kg tar)/(kg total)
 * @param A_               (m^3/kmol-s) Arrhenius preexponential factor [1]
 * @param E_               (J/mol) Activation energy                    [1]
 * @param o2MW_            (g/mol) molecular weight of O2
 */

template< typename ScalarT >
class TarOxidationRate
 : public Expr::Expression<ScalarT>
{
  DECLARE_FIELDS(ScalarT, density_, yO2_, yTar_, temp_ )
  const double A_, E_;

  /* declare operators associated with this expression here */

  TarOxidationRate( const Expr::Tag& densityTag,
                    const Expr::Tag& yO2Tag,
                    const Expr::Tag& yTarTag,
                    const Expr::Tag& tempTag );
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag densityTag_, yO2Tag_, yTarTag_, tempTag_;
  public:
    /**
     *  @brief Build a TarOxidationRate expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& densityTag,
             const Expr::Tag& yO2Tag,
             const Expr::Tag& yTarTag,
             const Expr::Tag& tempTag )
    : Expr::ExpressionBuilder( resultTag ),
      densityTag_( densityTag ),
      yO2Tag_     ( yO2Tag    ),
      yTarTag_    ( yTarTag   ),
      tempTag_    ( tempTag   )
    {}


    Expr::ExpressionBase* build() const{
      return new TarOxidationRate<ScalarT>( densityTag_, yO2Tag_, yTarTag_,tempTag_ );
    }
  };

  ~TarOxidationRate(){}
  void bind_operators( const SpatialOps::OperatorDatabase& opDB ){}
  void evaluate();
};

#endif // TarOxidationRate_h
