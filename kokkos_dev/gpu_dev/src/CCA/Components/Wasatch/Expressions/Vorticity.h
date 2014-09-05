#ifndef Vorticity_Expr_h
#define Vorticity_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/SpatialFieldStore.h>

// helper struct to identify which faces or directions we take derivatives on
template< typename Vel1T, typename Vel2T> struct VorticityFaceSelector;

// for omega_x = dw/dy - dv/dz
template<> struct VorticityFaceSelector<SpatialOps::structured::ZVolField, SpatialOps::structured::YVolField>
{
private:
  typedef SpatialOps::structured::ZVolField Vel1T;
  typedef SpatialOps::structured::YVolField Vel2T;  
public:
  typedef SpatialOps::structured::FaceTypes<Vel1T>::YFace Vel1FaceT;
  typedef SpatialOps::structured::FaceTypes<Vel2T>::ZFace Vel2FaceT;  
};

// for omega_y = du/dz - dw/dx
template<> struct VorticityFaceSelector<SpatialOps::structured::XVolField, SpatialOps::structured::ZVolField>
{
private:
  typedef SpatialOps::structured::XVolField Vel1T;
  typedef SpatialOps::structured::ZVolField Vel2T;  
public:
  typedef SpatialOps::structured::FaceTypes<Vel1T>::ZFace Vel1FaceT;
  typedef SpatialOps::structured::FaceTypes<Vel2T>::XFace Vel2FaceT;  
};

// for omega_z = dv/dx - du/dy
template<> struct VorticityFaceSelector<SpatialOps::structured::YVolField, SpatialOps::structured::XVolField>
{
private:
  typedef SpatialOps::structured::YVolField Vel1T;
  typedef SpatialOps::structured::XVolField Vel2T;  
public:
  typedef SpatialOps::structured::FaceTypes<Vel1T>::XFace Vel1FaceT;
  typedef SpatialOps::structured::FaceTypes<Vel2T>::YFace Vel2FaceT;  
};

/**
 *  \class 	Vorticity
 *  \author Tony Saad
 *  \date 	 February, 2012
 *  \ingroup	Expressions
 *
 *  \brief Calculates a cell centered vorticity. Note that the order of the
 *         velocity tags is important as it dictates fieldtypes.
 *
 */
template< typename FieldT,
typename Vel1T,
typename Vel2T >
class Vorticity
: public Expr::Expression<FieldT>
{
  const Expr::Tag vel1t_, vel2t_;
  
  typedef typename VorticityFaceSelector<Vel1T, Vel2T>::Vel1FaceT Vel1FaceT;
  typedef typename VorticityFaceSelector<Vel1T, Vel2T>::Vel2FaceT Vel2FaceT;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, Vel1T, Vel1FaceT >::type Vel1GradT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, Vel2T, Vel2FaceT >::type Vel2GradT;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, Vel1FaceT, FieldT >::type InpterpVel1FaceT2FieldT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, Vel2FaceT, FieldT >::type InpterpVel2FaceT2FieldT;
  
  const Vel1T* vel1_;
  const Vel2T* vel2_;
    
  const Vel1GradT* Vel1GradTOp_;
  const Vel2GradT* Vel2GradTOp_;

  const InpterpVel1FaceT2FieldT* InpterpVel1FaceT2FieldTOp_;
  const InpterpVel2FaceT2FieldT* InpterpVel2FaceT2FieldTOp_;
      
  Vorticity( const Expr::Tag& vel1tag,
             const Expr::Tag& vel2tag );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \param vel1tag the velocity corresponding to the Vel1T template parameter
     *  \param vel2tag the velocity corresponding to the Vel2T template parameter
     */
    Builder( const Expr::Tag& result,
            const Expr::Tag& vel1tag,
            const Expr::Tag& vel2tag);
    ~Builder(){}
    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::Tag v1t_, v2t_;
  };
  
  ~Vorticity();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

#endif // Stress_Expr_h
