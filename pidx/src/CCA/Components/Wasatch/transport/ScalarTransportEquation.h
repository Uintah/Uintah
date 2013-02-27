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

#ifndef Wasatch_ScalarTransportEquation_h
#define Wasatch_ScalarTransportEquation_h

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulenceParameters.h>
#include <CCA/Components/Wasatch/Expressions/Turbulence/TurbulentDiffusivity.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \class ScalarTransportEquation
   *  \authors James C. Sutherland, Tony Saad, Amir Biglari
   *  \date July, 2011. (Originally Created: June 2010).
   *
   *  \brief Support for a generic scalar transport equation.
   *
   *  Sets up solution for a transport equation of the general form of:
   *
   *  \f[
   *    \frac{\partial \rho \phi}{\partial t} =
   *    - \frac{\partial \rho \phi u_x }{\partial x}
   *    - \frac{\partial \rho \phi u_y }{\partial y}
   *    - \frac{\partial \rho \phi u_z }{\partial z}
   *    - \frac{\partial J_{\phi,x}}{\partial x}
   *    - \frac{\partial J_{\phi,y}}{\partial y}
   *    - \frac{\partial J_{\phi,z}}{\partial z}
   *    + s_\phi
   *  \f]
   *
   *  Any or all of the terms in the RHS above may be activated
   *  through the input file. Also, it can be define in input file
   *  wether we have a constant or variable density and strong or
   *  weak form of this equation.
   *
   *  The other forms of the equation are:
   *
   *  constant density strong form:
   *  \f[
   *    \frac{\partial \phi}{\partial t} =
   *    - \frac{\partial \phi u_x }{\partial x}
   *    - \frac{\partial \phi u_y }{\partial y}
   *    - \frac{\partial \phi u_z }{\partial z}
   *    - \frac{\partial V_{\phi,x}}{\partial x}
   *    - \frac{\partial V_{\phi,y}}{\partial y}
   *    - \frac{\partial V_{\phi,z}}{\partial z}
   *    + (1 / \rho) * s_\phi
   *  \f]
   *
   *  constant density weak form:
   *  \f[
   *    \frac{\partial \phi}{\partial t} =
   *    - u_x * \frac{\partial \phi}{\partial x}
   *    - u_y * \frac{\partial \phi}{\partial y}
   *    - u_z * \frac{\partial \phi}{\partial z}
   *    - \frac{\partial V_{\phi,x}}{\partial x}
   *    - \frac{\partial V_{\phi,y}}{\partial y}
   *    - \frac{\partial V_{\phi,z}}{\partial z}
   *    + (1 / \rho) * s_\phi
   *  \f]
   *
   *  variable density weak form:
   *  \f[
   *    \frac{\partial \phi}{\partial t} =
   *    - u_x * \frac{\partial \phi}{\partial x}
   *    - u_y * \frac{\partial \phi}{\partial y}
   *    - u_z * \frac{\partial \phi}{\partial z}
   *    - (1 / \rho) * \frac{\partial J_{\phi,x}}{\partial x}
   *    - (1 / \rho) * \frac{\partial J_{\phi,y}}{\partial y}
   *    - (1 / \rho) * \frac{\partial J_{\phi,z}}{\partial z}
   *    + (1 / \rho) * s_\phi
   *  \f]
   *
   *  \par Notes & Restrictions
   *
   *  - In the above equations "J" represents the diffusive flux which
   *    is equal to \f$\rho * \Gamma_\phi \nabla \phi \f$, while "V"
   *    shows the diffusive velocity which is equal to \f$\Gamma_\phi
   *    \Gamma_\phi \nabla \phi \f$.
   *
   *  - Currently, only basic forms for the scalar diffusive flux are
   *    supported.  Specifically, either an expression for the
   *    diffusion coefficient, \f$\Gamma_\phi\f$ is required or the
   *    diffusion coefficient must be a constant value.  See
   *    DiffusiveFlux and DiffusiveFlux2 classes.
   *
   *  - Source terms can only be added if the expression to evaluate
   *    them has been constructed elsewhere.
   *
   *  - In the case that we are solving a scalar transport equation with
   *    constant density we move out the density by division from all
   *    terms except the source term. For the source term in this case we
   *    devide the specified source term expression by density here in
   *    ScalarRHS.
   *    So, you should be carfule with the cases that source terms are
   *    NOT defined in the INPUT FILE but they will be added to the RHS
   *    automatically during the solution process and they are almost
   *    impossible to track (e.g. in ODT solver).
   *
   *  \todo Need to hook in parser support for boundary and initial conditions.
   */
  template<typename FieldT>
  class ScalarTransportEquation : public Wasatch::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a ScalarTransportEquation
     *  \param solnVarName the name of the solution variable for this ScalarTransportEquation
     *
     *  \param params the tag from the input file specifying the
     *         transport equation.
     *
     *  \param densityTag a tag containing density for necessary cases. it will be empty where
     *         it is not needed.
     *
     *  \param isConstDensity true for constant density
     *
     *  \param id the Expr::ExpressionID for the RHS expression for this ScalarTransportEquation
     *
     *  Note that the static member methods get_rhs_expr_id,
     *  get_primvar_name and get_solnvar_name can be useful
     *  to obtain the appropriate input arguments here.
     */
    ScalarTransportEquation( const std::string solnVarName,
                             Uintah::ProblemSpecP params,
                             const bool hasEmbeddedGeometry,
                             const Expr::Tag densityTag,
                             const bool isConstDensity,
                             const Expr::ExpressionID id );

    ~ScalarTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_initial_boundary_conditions( const GraphHelper& graphHelper,
                                            const Uintah::PatchSet* const localPatches,
                                            const PatchInfoMap& patchInfoMap,
                                            const Uintah::MaterialSubset* const materials,
                                           const std::map<std::string, std::set<std::string> >& bcFunctorMap_);


    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                    const Uintah::PatchSet* const localPatches,
                                    const PatchInfoMap& patchInfoMap,
                                    const Uintah::MaterialSubset* const materials,
                                   const std::map<std::string, std::set<std::string> >& bcFunctorMap_);

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

    /**
     * \brief Parse the input file to determine the rhs expression id.
     *        Also registers convective flux, diffusive flux, and
     *        source term expressions.
     *
     *  \param densityTag a tag containing density for necessary cases. it will be empty where
     *         it is not needed.
     *
     *  \param factory the Expr::ExpressionFactory object that terms
     *         associated with the RHS of this transport equation
     *         should be registered on.
     *
     *  \param isConstDensity true for constant density
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation.  Scope should be within the TransportEquation tag.
     *
     *  \param turbulenceParams information on the turbulence models being used
     */
    static Expr::ExpressionID get_rhs_expr_id( const Expr::Tag densityTag,
                                               const bool isConstDensity,
                                               Expr::ExpressionFactory& factory,
                                               Uintah::ProblemSpecP params,
                                               const bool hasEmbeddedGeometry,
                                               TurbulenceParameters turbulenceParams);

    /**
     *  \brief Parse the input file to get the name of the solution
     *         variable for this ScalarTransportEquation
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_solnvar_name( Uintah::ProblemSpecP params );

    /**
     *  \brief Parse the input file to get the primitive variable name
     *         for this ScalarTransportEquation.
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation. Scope should be within the TransportEquation tag.
     */
    static std::string get_primvar_name( Uintah::ProblemSpecP params );

  private:
    const bool isConstDensity_;
    const Expr::Tag densityTag_;
    Expr::Tag primVarTag_, solnVarTag_;
    bool isStrong_;

  };

} // namespace Wasatch

#endif // Wasatch_ScalarTransportEquation_h
