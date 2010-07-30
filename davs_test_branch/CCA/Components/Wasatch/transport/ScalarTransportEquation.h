#ifndef Wasatch_ScalarTransportEquation_h
#define Wasatch_ScalarTransportEquation_h


//-- ExprLib includes --//
#include <expression/TransportEquation.h>


//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredTypes.h>


//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>


//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Expr{
  class ExpressionFactory;
  class ExpressionID;
}


namespace Wasatch{

  /**
   *  \class ScalarTransportEquation
   *  \date June, 2010
   *  \author James C. Sutherland
   *
   *  \brief Support for a generic scalar transport equation.
   *
   *  Sets up solution for a transport equation of the form:
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
   *  through the input file.
   *
   *  \par Notes & Restrictions
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
   *  \todo Need to hook in parser support for boundary and initial conditions.
   */
  class ScalarTransportEquation : public Expr::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef ScalarVolField            FieldT; ///< The type of field for the solution variable.
    typedef FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a ScalarTransportEquation
     *
     *  \param solnExprFactory the Expr::ExpressionFactory object that
     *         terms associated with the RHS of this transport
     *         equation should be registered on.
     *
     *  \param params the Uintah::ProblemSpec XML description for this
     *         transport equation.
     */
    ScalarTransportEquation( Expr::ExpressionFactory& solnExprFactory,
                             Uintah::ProblemSpecP params );

    ~ScalarTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( Expr::ExpressionFactory& factory );

    /**
     *  \brief setup the initial conditions for this transport equation.
     */
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  private:

  };

} // namespace Wasatch


#endif // Wasatch_ScalarTransportEquation_h
