#ifndef Wasatch_TemperatureTransportEquation_h
#define Wasatch_TemperatureTransportEquation_h


//-- ExprLib includes --//
#include <expression/TransportEquation.h>


//-- SpatialOps includes --//
#include <spatialops/structured/FVStaggeredTypes.h>


//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldTypes.h>


namespace Expr{
  class ExpressionFactory;
  class ExpressionID;
}


namespace Wasatch{

  /**
   *  \class TemperatureTransportEquation
   *  \author James C. Sutherland
   *  \date June, 2010
   *
   *  \brief A basic transport equation for temperature.
   *
   */
  class TemperatureTransportEquation : public Wasatch::TransportEquation
  {
  public:

    // define the types of fields that this TransportEquation deals with.
    typedef SVolField            FieldT;
    typedef FaceTypes<FieldT>::XFace  XFaceT;
    typedef FaceTypes<FieldT>::YFace  YFaceT;
    typedef FaceTypes<FieldT>::ZFace  ZFaceT;

    TemperatureTransportEquation( Expr::ExpressionFactory& solnExprFactory );

    ~TemperatureTransportEquation();

    void setup_boundary_conditions( const GraphHelper& graphHelper,
                                   const Uintah::PatchSet* const localPatches,
                                   const PatchInfoMap& patchInfoMap,
                                   const Uintah::MaterialSubset* const materials);
    
    Expr::ExpressionID initial_condition( Expr::ExpressionFactory& icFactory );

  private:

    static Expr::ExpressionID get_rhs_id( Expr::ExpressionFactory& );
                       
  };

} // namespace Wasatch


#endif // Wasatch_TemperatureTransportEquation_h
