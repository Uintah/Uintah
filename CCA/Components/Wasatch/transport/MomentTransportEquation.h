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

#ifndef Wasatch_MomentTransportEquation_h
#define Wasatch_MomentTransportEquation_h

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \class MomentTransportEquation
   *  \date April, 2011
   *  \author Tony Saad
   *
   *  \brief Moment transport equation for population balances with a single
   *				 internal coordinate.
   *
   *  Sets up solution for a transport equation of the form:
   *
   *  \f[
   *    \frac{\partial m_k}{\partial t} =
   *    - \frac{\partial m_k u_x }{\partial x}
   *    - \frac{\partial m_k u_y }{\partial y}
   *    - \frac{\partial m_k u_z }{\partial z}
   *    - \frac{\partial J_{m_k,x}}{\partial x}
   *    - \frac{\partial J_{m_k,y}}{\partial y}
   *    - \frac{\partial J_{m_k,z}}{\partial z}
   *    + s_\phi
   *  \f]
   *
   *  Any or all of the terms in the RHS above may be activated
   *  through the input file.
   *
   *  \par Notes & Restrictions
   *
   *  - The source term of momentum equation is \f$\nabla P \f$ and
   *    the diffusive Flux is \f$\nabla \cdot \tau \f$. Right now in the
   *    momentumTransportEqu we do not require density when it is
   *    constant. So, you should be careful about these 2 terms in
   *    constant density cases. You need to use kinematic viscosity
   *    instead of dynamic viscosity for calculating \f$ \tau \f$ and
   *    use density wieghted pressure instead of normal pressure.
   *
   */
  template<typename FieldT>
  class MomentTransportEquation : public Wasatch::TransportEquation
  {
  public:
    /**
     *  \brief Construct a MomentTransportEquation
     *  \param basePhiName This equation will be created n-times where n is a user
     *         specified number in the input file. The basePhiName refers to the
     *         base name of the solution variable. The n-equations that are created
     *         will correspond to basePhiName0, basePhiName1, etc...
     *  \param thisPhiName The name of the solution variable for this ScalarTransportEquation
     *  \param id The Expr::ExpressionID for the RHS expression for this ScalarTransportEquation
     *
     *  Note that the static member method get_rhs_expr_id can be useful to
     *  obtain the appropriate input arguments here.
     */
    MomentTransportEquation(const std::string thisPhiName,
                            const Expr::ExpressionID id,
                            const bool hasEmbeddedGeometry,                            
                            Uintah::ProblemSpecP params);

    ~MomentTransportEquation();

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
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                   const Uintah::PatchSet* const localPatches,
                                   const PatchInfoMap& patchInfoMap,
                                   const Uintah::MaterialSubset* const materials,
                                   const std::map<std::string, std::set<std::string> >& bcFunctorMap);

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     * \brief Parse the input file to determine the rhs expression id.
     *        Also registers convective flux, diffusive flux, and
     *        source term expressions.
     *
     *  \param factory The Expr::ExpressionFactory object that
     *         terms associated with the RHS of this transport
     *         equation should be registered on.
     *
     *  \param params The Uintah::ProblemSpec XML description for this
     *         equation.  Scope should be within the ScalabilityTest tag.
     */
    static Expr::ExpressionID  get_moment_rhs_id(Expr::ExpressionFactory& factory,
                                                  Uintah::ProblemSpecP params,
                                                  Expr::TagList& weightsTags,
                                                  Expr::TagList& abscissaeTags,
                                                  const double momentOrder,
                                                  const double initialMoment);

    /**
     *  \brief Parse the input file to get the name of this ScalarTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    //static std::string get_phi_name( Uintah::ProblemSpecP params );

  private:
    //Expr::TagList weightsTagList_, abscissaeTagList_, weightsAndAbscissaeTagList_;
    };

} // namespace Wasatch
#endif // Wasatch_MomentTransportEquation_h


