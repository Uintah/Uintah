/*
 * The MIT License
 *
 * Copyright (c) 2016-2018 The University of Utah
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

/**
 *  \file   TarTransportEquation.h
 *  \date   November, 2016
 *  \author Josh McConnell
 */

#ifndef Wasatch_TarTransport_h
#define Wasatch_TarTransport_h

#include <CCA/Components/Wasatch/Transport/PseudospeciesTransportEquation.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error Tar transport requires PoKiTT.
#endif

namespace WasatchCore{

EqnTimestepAdaptorBase*
setup_tar_equation( Uintah::ProblemSpecP params,
                    const TurbulenceParameters& turbParams,
                    const Expr::Tag densityTag,
                    GraphCategories& gc );

class TarTransportEquation : public PseudospeciesTransportEquation<SVolField>
{
public:
  /**
   *  \brief Construct an TarTransportEquation
   *  \param params the tag from the input file specifying the transport equation.
   *  \param gc
   *  \param densityTag a tag containing density for necessary cases. it will be empty where
   *         it is not needed.
   *  \param turbulenceParams
   */
  TarTransportEquation( Uintah::ProblemSpecP params,
                        GraphCategories& gc,
                        const Expr::Tag densityTag,
                        const TurbulenceParameters& turbulenceParams );

  // Tar is assumed to have the chemical formula $C_{10}H_{8}$

  // get molecular weight of tar
  const double get_tar_molecular_weight() const { return 128.17; };

  // return the stoichiometric coefficient of carbon in the tar
  const double get_tar_carbon() const { return 10; };

  // return the stoichiometric coefficient of hydrogen in the tar
  const double get_tar_hydrogen() const { return 8; };

  ~TarTransportEquation();

protected:
  void setup_diffusive_flux( FieldTagInfo& );
  void setup_source_terms( FieldTagInfo&, Expr::TagList& );
  Expr::ExpressionID setup_rhs( FieldTagInfo&, Expr::TagList& );

private:
  const Expr::Tag densityTag_, tarOxRateTag_, sootFormRateTag_;

};

} /* namespace WasatchCore */

#endif /* Wasatch_TarTransport_h */
