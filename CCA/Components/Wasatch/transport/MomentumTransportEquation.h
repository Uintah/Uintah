#ifndef Wasatch_MomentumTransportEquation_h
#define Wasatch_MomentumTransportEquation_h

//-- ExprLib includes --//
#include <expression/TransportEquation.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>

//-- Uintah includes --//
#include <Core/ProblemSpec/ProblemSpecP.h>


namespace Wasatch{

  /**
   *  \ingroup WasatchCore
   *  \class MomentumTransportEquation
   *  \date January, 2011
   *
   *  \brief Creates a momentum transport equation
   */
  template< typename FieldT >
  class MomentumTransportEquation : public Expr::TransportEquation
  {
  public:

    // these typedefs are provided for convenience.
    typedef typename FaceTypes<FieldT>::XFace  XFaceT; ///< The type of field on the x-faces of the volume.
    typedef typename FaceTypes<FieldT>::YFace  YFaceT; ///< The type of field on the y-faces of the volume.
    typedef typename FaceTypes<FieldT>::ZFace  ZFaceT; ///< The type of field on the z-faces of the volume.

    /**
     *  \brief Construct a MomentumTransportEquation
     *  \param velName the name of the solution variable for this MomentumTransportEquation
     *  \param momName
     *  \param params Parser information for this momentum equation
     */
    MomentumTransportEquation( const std::string velName,
                               const std::string momName,
                               Expr::ExpressionFactory& factory,
                               Uintah::ProblemSpecP params );

    ~MomentumTransportEquation();

    /**
     *  \brief setup the boundary conditions associated with this transport equation
     */
    void setup_boundary_conditions( Expr::ExpressionFactory& factory );

    /**
     *  \brief setup the initial conditions for this transport equation.
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

  };

} // namespace Wasatch

#endif // Wasatch_MomentumTransportEquation_h
