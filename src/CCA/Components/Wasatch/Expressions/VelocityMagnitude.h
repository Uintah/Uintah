#ifndef Velocity_Magnitude_Expr_h
#define Velocity_Magnitude_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

/**
 *  \class 	Velocity Magnitude
 *  \author Tony Saad
 *  \date 	 February, 2012
 *  \ingroup	Expressions
 *
 *  \brief Calculates a cell centered velocity magnitude.
 *
 *  The Velocity magnitude is given by:
 *  \f[ \left |  \mathbf{u} \right | = \sqrt(\mathbf u \cdot \mathbf u) \f]     
 *
 *  \tparam FieldT  The type of field for the velocity magnitue - usually SVOL.
 *  \tparam Vel1T   The type of field for the first velocity component.
 *  \tparam Vel2T   The type of field for the second velocity component.
 *  \tparam ViscT   The type of field for the viscosity.
 */
template< typename FieldT,
typename Vel1T,
typename Vel2T,
typename Vel3T >
class VelocityMagnitude
: public Expr::Expression<FieldT>
{
  const Expr::Tag vel1t_, vel2t_, vel3t_;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1T, FieldT >::type InpterpVel1T2FieldT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2T, FieldT >::type InpterpVel2T2FieldT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, Vel3T, FieldT >::type InpterpVel3T2FieldT;
    
  const Vel1T* vel1_;
  const Vel2T* vel2_;
  const Vel3T* vel3_;
  
  const InpterpVel1T2FieldT* InpterpVel1T2FieldTOp_;
  const InpterpVel2T2FieldT* InpterpVel2T2FieldTOp_;
  const InpterpVel3T2FieldT* InpterpVel3T2FieldTOp_;
  
  VelocityMagnitude( const Expr::Tag& vel1tag,
                     const Expr::Tag& vel2tag,
                     const Expr::Tag& vel3tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     *  \param vel3tag the velocity corresponding to the Vel3T template parameter
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& vel1tag,
            const Expr::Tag& vel2tag,
            const Expr::Tag& vel3tag );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_, v3t_;
  };
  
  ~VelocityMagnitude();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Stress_Expr_h
