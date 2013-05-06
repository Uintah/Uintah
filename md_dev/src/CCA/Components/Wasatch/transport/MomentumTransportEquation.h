/*
 * The MIT License
 *
 * Copyright (c) 2012 The University of Utah
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

#ifndef Wasatch_MomentumTransportEquation_h
#define Wasatch_MomentumTransportEquation_h

//-- ExprLib includes --//
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Parallel/UintahParallelComponent.h>
#include <CCA/Ports/SolverInterface.h>

namespace Wasatch{

  void register_turbulence_expressions (const TurbulenceParameters& turbParams,
                                        Expr::ExpressionFactory& factory,
                                        const Expr::TagList& velTags,
                                        const Expr::Tag densTag);    
  /**
   *  \ingroup WasatchCore
   *  \class MomentumTransportEquation
   *  \authors James C. Sutherland, Tony Saad
   *  \date January, 2011
   *
   *  \brief Creates a momentum transport equation
   *
   *  \todo Allow more flexibility in specifying initial and boundary conditions for momentum.
   */
  template< typename FieldT >
  class MomentumTransportEquation : public Wasatch::TransportEquation
  {
  public:

    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a MomentumTransportEquation
     *  \param velName the name of the velocity component solved by this MomentumTransportEquation
     *  \param momName the name of the momentum component solved by this MomentumTransportEquation
     *  \param factory the Expr::ExpressionFactory that will hold expressions registered by this transport equation.
     *  \param params Parser information for this momentum equation
     *  \param linSolver the linear solver object for the pressure solve
     */
    MomentumTransportEquation( const std::string velName,
                               const std::string momName,
                               const Expr::Tag densTag,
                               const Expr::Tag bodyForceTag,
                               const Expr::Tag srcTermTag,
                               Expr::ExpressionFactory& factory,
                               Uintah::ProblemSpecP params,
                               TurbulenceParameters turbulenceParams,
                               const bool hasEmbeddedGeometry,
                               const bool hasMovingGeometry,
                               const Expr::ExpressionID rhsID,
                               Uintah::SolverInterface& linSolver,
                               Uintah::SimulationStateP sharedState);


    ~MomentumTransportEquation();

    static Expr::ExpressionID
    get_mom_rhs_id( Expr::ExpressionFactory& factory,
                   const std::string velName,
                   const std::string momName,
                   Uintah::ProblemSpecP params,
                   const bool hasEmbeddedGeometry,
                   Uintah::SolverInterface& linSolver );

    /**
     *  \brief apply the boundary conditions on the initial condition
     *         associated with this transport equation
     */
    void setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            const Uintah::PatchSet* const localPatches,
                                            const PatchInfoMap& patchInfoMap,
                                            const Uintah::MaterialSubset* const materials,
                                            const std::map<std::string, std::set<std::string> >& bcFunctorMap);


    /**
     *  \brief setup the boundary conditions associated with this momentum equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials,
                                    const std::map<std::string, std::set<std::string> >& bcFunctorMap);
    /**
     *  \brief setup the initial conditions for this momentum equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     *  \brief Parse the input file to get the name of this MomentumTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_phi_name( Uintah::ProblemSpecP params );

  private:

    Uintah::SolverParameters* solverParams_;
    const bool isviscous_;
    const bool isTurbulent_;
    const Expr::Tag thisVelTag_, densityTag_;
    Expr::ExpressionID normalStrainID_, normalConvFluxID_, pressureID_;
    std::string thisMomName_;
    Expr::TagList velTags_;  ///< TagList for the velocity expressions

  };

} // namespace Wasatch

#endif // Wasatch_MomentumTransportEquation_h
