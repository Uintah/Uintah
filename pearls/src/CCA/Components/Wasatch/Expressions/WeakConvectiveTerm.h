#ifndef WeakConvectiveTerm_Expr_h
#define WeakConvectiveTerm_Expr_h

//-- ExprLib Includes --//
#include <expression/Expression.h>

#include <spatialops/structured/FVStaggered.h>

/**
 *  \ingroup WasatchExpressions
 *  \class  WeakConvectiveTerm
 *  \author Amir Biglari
 *  \date	Sep, 2012
 *
 *  \brief calculates the convective term of the weak momentum transport equation using a 
 *         corresponding component of the velocity and the velocity vector. 
 *  Note that this requires the current velocity value, \f$u\f$, and the whole velocity vector
 *       \f$\textbf{\overrightarrow{u}}\f$
 */
template< typename FieldT >
class WeakConvectiveTerm
  : public Expr::Expression<FieldT>
{  
  typedef SpatialOps::FaceTypes<FieldT> FaceTypes;
  typedef typename FaceTypes::XFace XFace; ///< The type of field for the x-face of FieldT.
  typedef typename FaceTypes::YFace YFace; ///< The type of field for the y-face of FieldT.
  typedef typename FaceTypes::ZFace ZFace; ///< The type of field for the z-face of FieldT.
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XVolField, FieldT >::type  XInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YVolField, FieldT >::type  YInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZVolField, FieldT >::type  ZInterpT;  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, XFace, FieldT >::type  XFaceInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, YFace, FieldT >::type  YFaceInterpT;
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Interpolant, ZFace, FieldT >::type  ZFaceInterpT;  
  
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, FieldT, XFace >::type  GradXT; 
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, FieldT, YFace >::type  GradYT; 
  typedef typename SpatialOps::OperatorTypeBuilder< SpatialOps::Gradient, FieldT, ZFace >::type  GradZT; 
    
  // interpolant operators
  const XInterpT* xInterpOp_;
  const YInterpT* yInterpOp_;
  const ZInterpT* zInterpOp_;
  const XFaceInterpT* xFaceInterpOp_;
  const YFaceInterpT* yFaceInterpOp_;
  const ZFaceInterpT* zFaceInterpOp_;

  // gradient operators
  const GradXT* gradXOp_;
  const GradYT* gradYOp_;
  const GradZT* gradZOp_;
    
  const XVolField *velx_;
  const YVolField *vely_;
  const ZVolField *velz_;  
  const FieldT *vel_;

  const Expr::Tag velt_, velxt_, velyt_, velzt_;
  const bool is3d_;
  
  WeakConvectiveTerm( const Expr::Tag velTag,
          const Expr::TagList velTags );
public:
  
  /**
   *  \brief Builder for the convective term of the weak momentum transport equation
   */
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    
    /**
     *  \brief Constructs a builder for the convective term of the weak momentum transport equation
     *  \param result the convective term
     *  \param velTag a tag for the component of the velocity that we are advancing
     *  \param velTags a tag list holding velocity components
     */
    Builder( const Expr::Tag& result,
             const Expr::Tag velTag,
             const Expr::TagList velTags );
    
    ~Builder(){}

    Expr::ExpressionBase* build() const;
    
  private:
    const Expr::TagList velts_;
    const Expr::Tag velt_;

  };

  ~WeakConvectiveTerm();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();

};

#endif // WeakConvectiveTerm_Expr_h
