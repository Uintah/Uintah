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

#ifndef Wasatch_ThermodymamicPressureTransportEquation_h
#define Wasatch_ThermodymamicPressureTransportEquation_h

#include <sci_defs/wasatch_defs.h>

#ifndef HAVE_POKITT
// kill compilation if we don't have pokitt.
#error transport equation for thermodynamic pressure requires PoKiTT.
#endif

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentDiffusivity.h>

namespace WasatchCore{

  /**
   *  \ingroup WasatchCore
   *  \class ThermodynamicPressureTransportEquation
   *  \author Josh McConnell
   *  \date November 2018
   *
   *  The thermodynamic pressure transport equation has the form:
   *
   *  \f[
   *  \frac{\partial P}{\partial t} = \frac{Dp}{Dt} - \mathbf{v} \cdot \nabla P,
   *  \f]
   *
   *  where \f$\mathbf{v}\f$ is velocity. \f$ \frac{Dp}{Dt} \f$ is obtained by
   *  differentiating the ideal equation of state. Expressions required to calculate
   *  \f$ \frac{Dp}{Dt} \f$ are registered in the setup_rhs() method of
   *  EnthalpyTransportEquation.cc
   */
  class ThermodymamicPressureTransportEquation : public WasatchCore::TransportEquation
  {
  public:

    /**
     *  \brief Construct a ThermodynamicPressureTransportEquation
     *  \param solnVarName the name of the solution variable for this ThermodynamicPressureTransportEquation
     *  \param wasatchSpec the tag from the input file specifying Wasatch
     *         specification.
     *  \param gc an enum containing GraphHelpers
     *  \param velTags an Expr::TagList with tags to velocities
     *  \param momRHSTags an Expr::TagList with tags to momentum RHSs
     */
    ThermodymamicPressureTransportEquation( Uintah::ProblemSpecP   wasatchSpec,
                                            GraphCategories&       gc,
                                            std::set<std::string>& persistentFields );

     ~ThermodymamicPressureTransportEquation(){};

    /**
     *  \brief Used to check the validity of the boundary conditions specified
     *   by the user at a given boundary and also to infer/add new BCs on the
     *   type of boundary.  Example: at a stationary impermeable wall, we can
     *   immediately infer zero-velocity boundary conditions and check whether
     *   the user has specified any velocity BCs at that boundary. See examples
     *   in the momentum transport equation.
     */
    void setup_boundary_conditions( WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat );
    
    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            WasatchBCHelper& bcHelper );

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper );

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  private:
    Expr::ExpressionID setup_rhs( FieldTagInfo&, const Expr::TagList& );

    // we won't be setting up source terms, diffusive fluxes, nor convective fluxes
    void setup_diffusive_flux( FieldTagInfo& ){};
    void setup_convective_flux( FieldTagInfo& ){};
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){};

    Uintah::ProblemSpecP wasatchSpec_;
    std::set<std::string>& persistentFields_;

    Uintah::ProblemSpecP enthalpyParams_;

  };

} // namespace WasatchCore

#endif // Wasatch_ThermodymamicPressureTransportEquation_h
