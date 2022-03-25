#ifndef ActuatorDisk_Expr_h
#define ActuatorDisk_Expr_h

#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>

template< typename FieldT >
class ActuatorDisk 
  : public Expr::Expression<FieldT>
{
  typedef typename SpatialOps::BasicOpTypes<FieldT>  OpTypes;

  DECLARE_FIELD(FieldT, volFrac_)
  WasatchCore::UintahPatchContainer* patchContainer_;
  
  const double payloadMass_;
  const int rotors_;
  const double thrustDir_;
  const double radius_;

  
  ActuatorDisk( const Expr::Tag& volFracTag,
                  const double payload,
                  const int rotors,
                  const double thrustDir,
                  const double radius );



 public: 
  class Builder : public Expr::ExpressionBuilder
  {
      const Expr::Tag volfract_;
      const double payloadmass_;
      const int numrotors_;
      const double thrustdir_;
      const double rotor_radius_;

      public:
        Builder( const Expr::Tag& result,
                 const Expr::Tag& volFracTag,
                 const double payloadmass,
                 const int numrotors,
                 const double thrustdir,
                 const double rotor_radius);
        
      Expr::ExpressionBase* build() const;
  };

    ~ActuatorDisk();

    void bind_operators( const SpatialOps::OperatorDatabase& opDB );
    void evaluate();

};
 #endif // ActuatorDisk_Expr_h