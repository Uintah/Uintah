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
 *  \file   SootParticleTransportEquation.h
 *  \date   November, 2016
 *  \author Josh McConnell
 */

#ifndef Wasatch_SootParticleTransport_h
#define Wasatch_SootParticleTransport_h

#include <CCA/Components/Wasatch/Transport/PseudospeciesTransportEquation.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Transport/EquationAdaptors.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error Soot particle transport requires PoKiTT.
#endif

namespace WasatchCore{

EqnTimestepAdaptorBase*
setup_soot_particle_equation( Uintah::ProblemSpecP params,
                              const TurbulenceParameters& turbParams,
                              const Expr::Tag densityTag,
                              GraphCategories& gc );

class SootParticleTransportEquation : public PseudospeciesTransportEquation<SVolField>
{
public:
  /**
   *  \brief Construct an SootParticleTransportEquation
   *  \param params the tag from the input file specifying the transport equation.
   *  \param gc
   *  \param densityTag a tag containing density for necessary cases. it will be empty where
   *         it is not needed.
   *  \param turbulenceParams
   */
  SootParticleTransportEquation( Uintah::ProblemSpecP params,
                                 GraphCategories& gc,
                                 const Expr::Tag densityTag,
                                 const TurbulenceParameters& turbulenceParams );

  ~SootParticleTransportEquation();

protected:
  void setup_diffusive_flux( FieldTagInfo& );
  void setup_source_terms( FieldTagInfo&, Expr::TagList& );
  Expr::ExpressionID setup_rhs( FieldTagInfo&, Expr::TagList& );

private:
  const Expr::Tag densityTag_, sootAgglomRateTag_;

};

} /* namespace WasatchCore */

#endif /* Wasatch_SootParticleTransport_h */
