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

#ifndef Wasatch_LowMachMomentumTransportEquation_h
#define Wasatch_LowMachMomentumTransportEquation_h

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Transport/MomentumTransportEquationBase.h>
#include <CCA/Components/Wasatch/Transport/TransportEquation.h>
#include <CCA/Components/Wasatch/Transport/EquationBase.h>
//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SolverInterface.h>

namespace WasatchCore{

  /**
   *  \ingroup WasatchCore
   *  \class LowMachMomentumTransportEquation
   *  \authors James C. Sutherland, Tony Saad
   *  \date January, 2011
   *
   *  \brief Creates a momentum transport equation for solving low-Mach problems on staggered grids
   *
   */
  template< typename FieldT >
  class LowMachMomentumTransportEquation : public WasatchCore::MomentumTransportEquationBase<FieldT>
  {
  public:

    typedef typename FaceTypes<FieldT>::XFace  XFace; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFace; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFace; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a LowMachMomentumTransportEquation
     *  \param momComponent the direction of this component of momentum
     *  \param velName the name of the velocity component solved by this LowMachMomentumTransportEquation
     *  \param momName the name of the momentum component solved by this LowMachMomentumTransportEquation
     *  \param densTag the tag for the mixture mass density
     *  \param isConstDensity
     *  \param bodyForceTag tag for body force
     *  \param srcTermTag tag for any source terms present
     *  \param gc
     *  \param params Parser information for this momentum equation
     *  \param turbulenceParams
     *  \param linSolver the linear solver object for the pressure solve
     *  \param sharedState contains useful stuff like the value of timestep, etc.
     */
    LowMachMomentumTransportEquation( const Direction momComponent,
                                      const std::string velName,
                                      const std::string momName,
                                      const Expr::Tag densTag,
                                      const bool isConstDensity,
                                      const Expr::Tag bodyForceTag,
                                      const Expr::Tag srcTermTag,
                                      GraphCategories& gc,
                                      Uintah::ProblemSpecP params,
                                      TurbulenceParameters turbulenceParams,
                                      Uintah::SolverInterface& linSolver,
                                      Uintah::SimulationStateP sharedState );

    ~LowMachMomentumTransportEquation();

    void setup_boundary_conditions(WasatchBCHelper& bcHelper,
                                    GraphCategories& graphCat);
    
    /**
     *  \brief apply the boundary conditions on the initial condition
     *         associated with this transport equation
     */
    void apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                           WasatchBCHelper& bcHelper );

    /**
     *  \brief setup the boundary conditions associated with this momentum equation
     */
    void apply_boundary_conditions( const GraphHelper& graphHelper,
                                   WasatchBCHelper& bcHelper );
    /**
     *  \brief setup the initial conditions for this momentum equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     *  \brief Parse the input file to get the name of this LowMachMomentumTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_phi_name( Uintah::ProblemSpecP params );

  protected:

    void setup_diffusive_flux( FieldTagInfo& ){}
    void setup_convective_flux( FieldTagInfo& ){}
    void setup_source_terms( FieldTagInfo&, Expr::TagList& ){}
    Expr::ExpressionID setup_rhs( FieldTagInfo&,
                                  const Expr::TagList& srcTags  );

  private:
    Uintah::SolverParameters* solverParams_;
    

  };

} // namespace WasatchCore

#endif // Wasatch_LowMachMomentumTransportEquation_h
