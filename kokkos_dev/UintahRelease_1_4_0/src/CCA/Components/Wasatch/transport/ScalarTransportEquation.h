#ifndef Wasatch_ScalarTransportEquation_h
#define Wasatch_ScalarTransportEquation_h

//-- ExprLib includes --//
#include <CCA/Components/Wasatch/transport/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ScalarRHS.h>
#include <CCA/Components/Wasatch/BCHelperTools.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \class ScalarTransportEquation
   *  \date June, 2010
   *  \author James C. Sutherland
   *  \modifier Amir Biglari
   *  \date July, 2011
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
   *    constant density we move out the density by devision from all  
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
     *  \param id the Expr::ExpressionID for the RHS expression for this ScalarTransportEquation
     *
     *  Note that the static member methods get_rhs_expr_id,
     *  get_primvar_name and get_solnvar_name can be useful
     *  to obtain the appropriate input arguments here.
     */
    ScalarTransportEquation( const std::string solnVarName,
                             Uintah::ProblemSpecP params,
                             const Expr::Tag densityTag,
                             const bool isConstDensity,
                             const Expr::ExpressionID id );

    ~ScalarTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                           const Uintah::PatchSet* const localPatches,
                                           const PatchInfoMap& patchInfoMap,
                                           const Uintah::MaterialSubset* const materials);
    
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
     *  \param params the Uintah::ProblemSpec XML description for this
     *         equation.  Scope should be within the TransportEquation tag.
     */
    static Expr::ExpressionID get_rhs_expr_id( const Expr::Tag densityTag,
                                               const bool isConstDensity,
                                               Expr::ExpressionFactory& factory,
                                               Uintah::ProblemSpecP params );

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
    Expr::Tag primVarTag_, densityTag_, solnVarTag_;
    bool isStrong_, isConstDensity_;

  };


  template< typename FieldT>
  void setup_diffusive_flux_expression( Uintah::ProblemSpecP diffFluxParams,
                                        const Expr::Tag densityTag,
                                        const Expr::Tag primVarTag,
                                        const bool isStrong,
                                        Expr::ExpressionFactory& factory,
                                        typename ScalarRHS<FieldT>::FieldTagInfo& info );
  template< typename FieldT>
  void setup_diffusive_velocity_expression( Uintah::ProblemSpecP diffVelParams,
                                            const Expr::Tag primVarTag,
                                            Expr::ExpressionFactory& factory,
                                            typename ScalarRHS<FieldT>::FieldTagInfo& info );

  template< typename FieldT >
  void setup_convective_flux_expression( Uintah::ProblemSpecP convFluxParams,
                                         const Expr::Tag solnVarName,
                                         Expr::ExpressionFactory& factory,
                                         typename ScalarRHS<FieldT>::FieldTagInfo& info );
  
} // namespace Wasatch
#endif // Wasatch_ScalarTransportEquation_h
