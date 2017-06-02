/**
 *  \file   EnthalpyTransportEquation.h
 *  \date   Nov 12, 2013
 *  \author "James C. Sutherland"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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
 *
 */
#ifndef TotalInternalEnergy_Transport_Equation_H_
#define TotalInternalEnergy_Transport_Equation_H_

#include <sci_defs/wasatch_defs.h>
#include <CCA/Components/Wasatch/Transport/ScalarTransportEquation.h>

namespace WasatchCore {

  /**
   *  \class  WasatchCore::TotalInternalEnergyTransportEquation
   *  \date   November, 2013
   *  \author James C. Sutherland, Tony Saad
   *
   *  The total internal energy transport equation has the form:
   *  \f[
   *  \frac{\partial \rho e_0}{\partial t} = - \nabla\cdot(\rho e_0\vec{v})
   *      - \nabla\cdot(\tau\cdot\vec{v}+p\vec{v}) - \nabla\cdot\vec{q}
   *  \f]
   *  where \f$e_0 = e + \frac{1}{2}\mathbf{v}\cdot\mathbf{v}\f$ is the specific
   *  total internal energy and \f$e=h-\frac{p}{\rho}\f$ is the specific
   *  internal energy.
   */
  class TotalInternalEnergyTransportEquation : public ScalarTransportEquation<SVolField>
  {
  public:
    /**
     *  \brief Construct a TotalInternalEnergyTransportEquation
     *  \param e0Name the name for the total internal energy
     *  \param wasatchSpec the block from the input file for Wasatch.
     *  \param energyEqnSpec the block from the input file specifying the transport equation.
     *  \param gc
     *  \param densityTag a tag containing density
     *  \param temperatureTag a tag for the temperature
     *  \param pressureTag a tag for the temperature
     *  \param velTags tags for the x-, y- and z-components of velocity.
     *  \param bodyForceTags the TagList of body forces for each of the three directions
     *  \param viscTag the tag for the viscosity
     *  \param dilTag the tag for the dilatation
     *  \param turbulenceParams
     */
    TotalInternalEnergyTransportEquation( const std::string rhoe0Name,
                                          Uintah::ProblemSpecP wasatchSpec,
                                          Uintah::ProblemSpecP energyEqnSpec,
                                          GraphCategories& gc,
                                          const Expr::Tag& densityTag,
                                          const Expr::Tag& temperatureTag,
                                          const Expr::Tag& pressureTag,
                                          const Expr::TagList& velTags,
                                          const Expr::TagList& bodyForceTags,
                                          const Expr::Tag& viscTag,
                                          const Expr::Tag& dilTag,
                                          const TurbulenceParameters& turbulenceParams );

    ~TotalInternalEnergyTransportEquation();
    
    void apply_boundary_conditions( const GraphHelper& graphHelper, WasatchBCHelper& bcHelper );
    void setup_boundary_conditions( WasatchBCHelper& bcHelper, GraphCategories& graphCat );

  protected:
    void setup_diffusive_flux ( FieldTagInfo& );
    void setup_convective_flux( FieldTagInfo& );
    void setup_source_terms( FieldTagInfo&, Expr::TagList& );

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    Uintah::ProblemSpecP wasatchSpec_;
    const Expr::Tag kineticEnergyTag_, temperatureTag_, pressureTag_;
    const Expr::TagList velTags_, bodyForceTags_;
#   ifdef HAVE_POKITT
    Expr::TagList massFracTags_;
#   endif
  };

} /* namespace WasatchCore */

#endif /* TotalInternalEnergy_Transport_Equation_H_ */
