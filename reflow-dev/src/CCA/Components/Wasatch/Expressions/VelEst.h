#ifndef VelEst_Expr_h
#define VelEst_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/structured/FVStaggeredOperatorTypes.h>


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
  typedef SpatialOps::structured::SingleValueField TimeField;
  typedef SpatialOps::structured::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFace; ///< The type of field for the x-face of FieldT.
  typedef typename FaceTypes::YFace YFace; ///< The type of field for the y-face of FieldT.
  typedef typename FaceTypes::ZFace ZFace; ///< The type of field for the z-face of FieldT.
  
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, FieldT >::type ScalarInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, XFace >::type  S2XFInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, YFace >::type  S2YFInterpT;
  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Interpolant, SVolField, ZFace >::type  S2ZFInterpT;

  typedef typename SpatialOps::structured::OperatorTypeBuilder< SpatialOps::Gradient, SVolField, FieldT >::type  GradPT; 
  
  typedef SpatialOps::structured::BasicOpTypes<FieldT> OpTypes;
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
  
  const XFace *tauxi_;
  const YFace *tauyi_;
  const ZFace *tauzi_;
  const FieldT *vel_, *convTerm_;
  const SVolField *density_, *pressure_, *visc_;
  const TimeField *tStep_;

  const Expr::Tag velt_, convTermt_, densityt_, visct_, tauxit_, tauyit_, tauzit_, pressuret_, tStept_;
  const bool is3d_;
  
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
     *
     *  \param the velTag a tag for the component of the velocity that we are advancing
     *
     *  \param the tauTags a tag list holding stress tensor components related to the 
     *         component of the velocity which exists in velTag.
     *
     *  \param the densityTag a tag for density at the current time step
     *
     *  \param the pressureTag a tag for pressure field at the previous time step.
     *
     *  \param timeStepTag a tag for the time step at the current RK stage
     *
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
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // VelEst_Expr_h
