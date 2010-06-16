#ifndef Wasatch_TemperatureTransportEquation_h
#define Wasatch_TemperatureTransportEquation_h


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

  class TemperatureTransportEquation : public Expr::TransportEquation
  {
  public:

    typedef ScalarVolField            FieldT;
    typedef FaceTypes<FieldT>::XFace  XFaceT;
    typedef FaceTypes<FieldT>::YFace  YFaceT;
    typedef FaceTypes<FieldT>::ZFace  ZFaceT;

    TemperatureTransportEquation( Expr::ExpressionFactory& solnExprFactory,
                                  Uintah::ProblemSpecP params );

    ~TemperatureTransportEquation();

    void setup_boundary_conditions( Expr::ExpressionFactory& factory );

    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  private:

    static Expr::ExpressionID get_rhs_id( Expr::ExpressionFactory& );
                       
  };

} // namespace Wasatch


#endif // Wasatch_TemperatureTransportEquation_h
