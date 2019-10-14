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
 *  \file   SpeciesTransport.h
 *  \date   May 16, 2016
 *  \author james
 */

#ifndef WASATCH_SPECIESTRANSPORT_H_
#define WASATCH_SPECIESTRANSPORT_H_

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error Species transport requires PoKiTT.
#endif

#include <pokitt/kinetics/AnalyticalJacobian.h>

#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>

namespace WasatchCore{

class DualTimeMatrixInfo;  // forward declaration

class EqnTimestepAdaptorBase;

std::vector<EqnTimestepAdaptorBase*>
setup_species_equations( Uintah::ProblemSpecP params,
                         Uintah::ProblemSpecP wasatchSpec,
                         const TurbulenceParameters& turbParams,
                         const Expr::Tag densityTag,
                         const Expr::TagList velTags,
                         const Expr::Tag temperatureTag,
                         GraphCategories& gc,
                         WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                         const bool computeKineticsJacobian );

/**
 * \class SpeciesTransportEquation
 * \author James C. Sutherland
 * \date May, 2016
 *
 * This class requires PoKiTT and Cantera
 */
class SpeciesTransportEquation : public TransportEquation
{
  Uintah::ProblemSpecP params_;
  Uintah::ProblemSpecP wasatchSpec_;
  const TurbulenceParameters turbParams_;
  const int specNum_;
  const Expr::Tag primVarTag_;
  const Expr::Tag densityTag_, temperatureTag_, mmwTag_;
  const Expr::TagList velTags_;
  const int nspec_;
  Expr::TagList yiTags_;
  WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo_;
  boost::shared_ptr<pokitt::ChemicalSourceJacobian> jacobian_;


public:

  SpeciesTransportEquation( Uintah::ProblemSpecP params,
                            Uintah::ProblemSpecP wasatchSpec,
                            const TurbulenceParameters& turbParams,
                            const int specNum,
                            GraphCategories& gc,
                            const Expr::Tag densityTag,
                            const Expr::TagList velTags,
                            const Expr::Tag temperatureTag,
                            const Expr::Tag mmwTag,
                            WasatchCore::DualTimeMatrixInfo& dualTimeMatrixInfo,
                            const bool computeKineticsJacobian );

  ~SpeciesTransportEquation();

  Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                  GraphCategories& graphCat );

  void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                          WasatchBCHelper& bcHelper );

  void apply_boundary_conditions( const GraphHelper& graphHelper,
                                  WasatchBCHelper& bcHelper );

  void setup_diffusive_flux( FieldTagInfo& );

  virtual void setup_convective_flux( FieldTagInfo& );

  void setup_source_terms( FieldTagInfo&, Expr::TagList& );

  Expr::ExpressionID setup_rhs( FieldTagInfo& info,
                                const Expr::TagList& srcTags );

};

} // namespace WasatchCore

#endif /* WASATCH_SPECIESTRANSPORT_H_ */
