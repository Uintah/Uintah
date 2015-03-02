#ifndef VelEst_Expr_h
#define VelEst_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  VelEst
 *  \author Amir Biglari
 *  \date	Sep, 2012
 *
 *  \brief Estimates the value of one of the velocity components in the next time step
 *         using a simple forward euler method and the partial rhs of the weak momentum equation. 
 *
 *  Note that this requires the current velocity value, \f$u\f$, convective term of the weak from 
 *       momentum equation, \f$\textbf{\overrightarrow{u}} \cdot \nabla u \f$, density,   
 *       \f$\rho\f$, stress terms related to this component of velocity, \f$\tau_{iu}\f$, and the 
 *       time step at the current RK stage.
 */
template< typename FieldT >
class VelEst
  : public Expr::Expression<FieldT>
{  
  typedef SpatialOps::SingleValueField TimeField;
  typedef SpatialOps::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFace; ///< The type of field for the x-face of FieldT.
  typedef typename FaceTypes::YFace YFace; ///< The type of field for the y-face of FieldT.
  typedef typename FaceTypes::ZFace ZFace; ///< The type of field for the z-face of FieldT.
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type ScalarInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XFace >::type  S2XFInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YFace >::type  S2YFInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZFace >::type  S2ZFInterpT;

  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, FieldT >::type  GradPT; 
  
  typedef SpatialOps::BasicOpTypes<FieldT> OpTypes;
  typedef typename OpTypes::DivX DivXT; ///< Divergence operator (surface integral) in the x-direction
  typedef typename OpTypes::DivY DivYT; ///< Divergence operator (surface integral) in the y-direction
  typedef typename OpTypes::DivZ DivZT; ///< Divergence operator (surface integral) in the z-direction
  
  // interpolant operators
  const ScalarInterpT* scalarInterpOp_;
  const S2XFInterpT* s2XFInterpOp_;
  const S2YFInterpT* s2YFInterpOp_;
  const S2ZFInterpT* s2ZFInterpOp_;

  // gradient operators
  const GradPT* gradPOp_;
  
  // divergence operators
  const DivXT* divXOp_;
  const DivYT* divYOp_;
  const DivZT* divZOp_;
  
  DECLARE_FIELDS(SVolField, density_, pressure_, visc_);
  DECLARE_FIELDS(FieldT, vel_, convTerm_);
  DECLARE_FIELD(XFace, tauxi_);
  DECLARE_FIELD(YFace, tauyi_);
  DECLARE_FIELD(ZFace, tauzi_);
  DECLARE_FIELD(TimeField, dt_);
  
  const bool doX_, doY_, doZ_, is3d_;
  
  VelEst( const Expr::Tag velTag,
          const Expr::Tag convTermTag,
          const Expr::TagList tauTags,
          const Expr::Tag densityTag,
          const Expr::Tag viscTag,
          const Expr::Tag pressureTag,
          const Expr::Tag timeStepTag );
public:
  
  /**
   *  \brief Builder for the estimation of one of the components of velocity at the next time step
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \brief Constructs a builder for one of the velocity components estimation
     *  \param result the tag for the velocity estimate
     *  \param velTag a tag for the component of the velocity that we are advancing
     *  \param convTermTag a tag for the convective term
     *  \param tauTags a tag list holding stress tensor components related to the
     *         component of the velocity which exists in velTag.
     *  \param densityTag a tag for density at the current time step
     *  \param viscTag a tag for viscosity
     *  \param pressureTag a tag for pressure field at the previous time step.
     *  \param timeStepTag a tag for the time step at the current RK stage
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag velTag,
             const Expr::Tag convTermTag,
             const Expr::TagList tauTags,
             const Expr::Tag densityTag,
             const Expr::Tag viscTag,
             const Expr::Tag pressureTag,
             const Expr::Tag timeStepTag  );
    
    ~Builder(){}

    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::TagList tauts_;
    const Expr::Tag velt_, convTermt_, densityt_, visct_, pt_;
    const Expr::Tag tstpt_;

  };

  ~VelEst();

  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // VelEst_Expr_h
