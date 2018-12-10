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

#ifndef Wasatch_ScalarTransportEquation_h
#define Wasatch_ScalarTransportEquation_h

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
   *    - \nabla\cdot\rho\phi\vec{u}
   *    - \nabla\cdot\vec{J}_\phi
   *    + s_\phi
   *  \f]
   *
   *  Any or all of the terms in the RHS above may be activated
   *  through the input file. Also, it can be define in input file
   *  whether we have a constant or variable density and strong or
   *  weak form of this equation.
   *
   *  The other supported forms of the equation are:
   *
   *  - Constant density and variable density weak form:
   *  \f[
   *    \frac{\partial \phi}{\partial t} =
   *     \frac{1}{\rho}\left[
   *     - \phi \frac{\partial \rho}{\partial t}
   *     - \nabla\cdot\rho\phi\vec{u}
   *     - \nabla\cdot\vec{J}_\phi
   *     + s_\phi
   *    \right]
   *  \f]
   *   Note that in this form, \f$\frac{\partial \rho}{\partial t}\f$ is retained
   *   and the strong form of the convective flux is also retained.  The model
   *   for \f$\frac{\partial \rho}{\partial t}\f$ being used in the pressure
   *   Poisson equation is used here.
   *
   *  \par Notes & Restrictions
   *
   *  - In the above equations, the diffusive flux is assumed to have the form
   *    \f[ \vec{J}_\phi = -\rho \Gamma_\phi \nabla \phi. \f]
   *    This can also be written in terms of a diffusive velocity,
   *    \f[ \vec{V}_\phi = \frac{\vec{J}}{\rho}. \f]
   *    The user must provide an expression or a constant value for
   *    \f$\Gamma_\phi\f$.  See the DiffusiveFlux and DiffusiveFlux2 classes
   *    for further details.
   *
   *  - Source terms, \f$s_\phi\f$, can only be added if the expression to
   *    evaluate them has been constructed elsewhere.
   *
   *  - When solving constant density or weak forms of the scalar transport
   *    equation, the source term(s) must be divided by \f$\rho\f$.  This is
   *    handled for all source terms that this transport equation is made aware
   *    of at construction.  However, if source terms are added to the RHS then
   *    they need to be scaled by density in the case where the weak or constant
   *    density form of the equation is being solved!  This is NOT done
   *    automatically.
   */
  template<typename FieldT>
  class ScalarTransportEquation : public WasatchCore::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    typedef FieldT MyFieldT; ///< The type of field for this transported variable

    /**
     *  \brief Construct a ScalarTransportEquation
     *  \param solnVarName the name of the solution variable for this ScalarTransportEquation
     *  \param params the tag from the input file specifying the
     *         transport equation.
     *  \param gc
     *  \param densityTag a tag containing density for necessary cases. it will be empty where
     *         it is not needed.
     *  \param isConstDensity true for constant density
     *  \param turbulenceParams information on turbulence models
     *  \param callSetup for objects that derive from ScalarTransportEquation,
     *         this flag should be set to false, and those objects should call
     *         setup() at the end of their constructor.
     *
     *  Note that the static member methods get_rhs_expr_id,
     *  get_primvar_name and get_solnvar_name can be useful
     *  to obtain the appropriate input arguments here.
     */
    ScalarTransportEquation( const std::string solnVarName,
                             Uintah::ProblemSpecP params,
                             GraphCategories& gc,
                             const Expr::Tag densityTag,
                             const TurbulenceParameters& turbulenceParams,
                             std::set<std::string>& persistentFields,
                             const bool callSetup=true );

    virtual ~ScalarTransportEquation();

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

    bool is_weak_form() const{ return !isStrong_; }
    bool is_strong_form() const{ return isStrong_; }
    
  protected:
    virtual void setup_diffusive_flux( FieldTagInfo& );
    virtual void setup_convective_flux( FieldTagInfo& );
    virtual void setup_source_terms( FieldTagInfo&, Expr::TagList& );
    virtual Expr::ExpressionID setup_rhs( FieldTagInfo&,
                                          const Expr::TagList& srcTags  );

    Uintah::ProblemSpecP params_;
    const bool hasConvection_;
    const Expr::Tag densityInitTag_, densityTag_, densityNP1Tag_;
    const bool enableTurbulence_;
    Expr::Tag primVarInitTag_, primVarTag_, primVarNP1Tag_;
    Expr::Tag turbDiffTag_;
    bool isStrong_;
    FieldTagInfo infoNP1_;  // needed to form predicted scalar quantities
    std::set<std::string>& persistentFields_;

  };

} // namespace WasatchCore

#endif // Wasatch_ScalarTransportEquation_h
