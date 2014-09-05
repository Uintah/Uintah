/* ---------------------------------------------------------------------------------------------- 
   %%%%%%      %%%%%%    %%      %%  %%      %%  %%%%%%%%    %%%%%%  %%%%%%%%%%  %%      %%%%%%    %%      %%
 %%          %%      %%  %%%%    %%  %%      %%  %%        %%            %%      %%    %%      %%  %%%%    %%
 %%          %%      %%  %%  %%  %%    %%  %%    %%%%%%    %%            %%      %%    %%      %%  %%  %%  %%
 %%          %%      %%  %%  %%  %%    %%  %%    %%        %%            %%      %%    %%      %%  %%  %%  %%
 %%          %%      %%  %%    %%%%    %%  %%    %%        %%            %%      %%    %%      %%  %%    %%%%
   %%%%%%      %%%%%%    %%      %%      %%      %%%%%%%%    %%%%%%      %%      %%      %%%%%%    %%      %%
 ---------------------------------------------------------------------------------------------- */
#ifndef ConvectiveFlux_h
#define ConvectiveFlux_h

//-- ExprLib includes --//
#include <expression/Expr_Expression.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>

/**
 *  \ingroup Expressions
 *  \class   ConvectiveFlux
 *  \author  Tony Saad
 *  \date    July, 2010
 *
 *  \brief Creates an expression for the convective flux of a scalar
 *  given a velocity field \f$\mathbf{u}\f$. We write the convective
 *  flux in conservation form as \f$ J_i = \rho \varphi u_i = \phi u_i
 *  \f$ where \f$i=1,2,3\f$ is the coordinate direction. This requires
 *  knowledge of the velocity field.
 *
 *  Here, we are constructing the convective flux J_i, therefore, it
 *  is convenient to set \f$ \rho \varphi \equiv \phi\f$
 *
 *  \tparam PhiInterpT The type of operator used in forming
 *          \f$\frac{\partial \phi}{\partial x}\f$
 *
 *  \tparam VelInterpT The type of operator used in interpolating the
 *          velocity from volume to face fields
 */
template< typename PhiInterpT, typename VelInterpT > // scalar interpolant and velocity interpolant
class ConvectiveFlux
  : public Expr::Expression<typename PhiInterpT::DestFieldType>
{  
  // PhiInterpT: an interpolant from staggered or non-staggered volume field to staggered or non-staggered face field
  typedef typename PhiInterpT::SrcFieldType  PhiVolT;  ///< source field is a scalar volume
  typedef typename PhiInterpT::DestFieldType PhiFaceT; ///< destination field is scalar face
  
  // VelInterpT: an interpolant from Staggered volume field to scalar face field
  typedef typename VelInterpT::SrcFieldType  VelVolT;  ///< source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;
  // the destination field of VelInterpT should be a PhiFaceT

  const Expr::Tag phiTag_, velTag_;
  const PhiVolT* phi_;
  const VelVolT* vel_;
  PhiInterpT* phiInterpOp_;
  const VelInterpT* velInterpOp_;
  
public:
  ConvectiveFlux( const Expr::Tag phiTag,
                  const Expr::Tag velTag,
                  const Expr::ExpressionID& id,
                  const Expr::ExpressionRegistry& reg  );
  ~ConvectiveFlux();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag phiT_, velT_;
    
  public:
    /**
     *  \brief Construct a convective flux given an expression for
     *         \f$\phi\f$.
     *
     *  \param phiTag  the Expr::Tag for the scalar field.
     *         This is located at cell centroids.
     *
     *  \param velTag the Expr::Tag for the velocity field.
     *         The velocity field is a face field.
     */
    Builder( const Expr::Tag phiTag,
             const Expr::Tag velTag )
      : phiT_(phiTag), velT_(velTag)
    {}
		
    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;

    ~Builder(){}
  };
};

/**
 *  \ingroup WasatchExpressions
 *  \class   ConvectiveFluxLimiter
 *  \author  Tony Saad
 *  \date    January, 2011
 *
 *  \brief Creates an expression for the convective flux of a scalar
 *  given a velocity field \f$\mathbf{u}\f$. We write the convective
 *  flux in conservation form as \f$ J_i = \rho \varphi u_i = \phi u_i
 *  \f$ where \f$i=1,2,3\f$ is the coordinate direction. This requires
 *  knowledge of the velocity field. The ConvectiveFluxLimiter is used for
 *  all interpolants that are dependent on a velocity field such as the
 *  Upwind scheme or any of the flux limiters such as Superbee.
 *
 *  Here, we are constructing the convective flux J_i, therefore, it
 *  is convenient to set \f$ \rho \varphi \equiv \phi\f$
 *
 *  \par Template Parameters
 *  <ul>
 *  <li> \b PhiInterpT The type of operator used in forming
 *       \f$\frac{\partial \phi}{\partial x}\f$.  Interpolates from
 *       staggered or non-staggered volume field to staggered or
 *       non-staggered face field
 *  <li> \b VelInterpT The type of operator used in interpolating the
 *        velocity from volume to face fields. Interpolates from
 *        staggered volume field to scalar face field
 *  </ul>
 */
template< typename PhiInterpT, typename VelInterpT > // scalar interpolant and velocity interpolant
class ConvectiveFluxLimiter
  : public Expr::Expression<typename PhiInterpT::DestFieldType>
{  
  typedef typename PhiInterpT::SrcFieldType  PhiVolT;  ///< source field is a scalar volume
  typedef typename PhiInterpT::DestFieldType PhiFaceT; ///< destination field is scalar face
  
  typedef typename VelInterpT::SrcFieldType  VelVolT;  ///< source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;

  const Expr::Tag phiTag_, velTag_;
  const Wasatch::ConvInterpMethods limiterType_;

  const PhiVolT* phi_;
  const VelVolT* vel_;

  PhiInterpT* phiInterpOp_;
  const VelInterpT* velInterpOp_;

  ConvectiveFluxLimiter( const Expr::Tag phiTag,
                         const Expr::Tag velTag,
                         Wasatch::ConvInterpMethods limiterType,
                         const Expr::ExpressionID& id,
                         const Expr::ExpressionRegistry& reg );
  
public:
  class Builder : public Expr::ExpressionBuilder
  {
    const Expr::Tag phiT_, velT_;
    Wasatch::ConvInterpMethods limiterType_;
  public:
    /**
     *  \brief Construct an convective flux limiter given an expression
     *         for \f$\phi\f$.
     *
     *  \param phiTag the Expr::Tag for the scalar field.  This is
     *         located at cell centroids.
     *
     *  \param velTag the Expr::Tag for the velocity field.  The
     *         velocity field is a face field.
     *
     *  \param limiterType the type of flux limiter to use.
     */
    Builder( const Expr::Tag phiTag,
             const Expr::Tag velTag,
            Wasatch::ConvInterpMethods limiterType)
      : phiT_( phiTag ), velT_( velTag ), limiterType_( limiterType )
    {}
    
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                const Expr::ExpressionRegistry& reg ) const
    {
      return new ConvectiveFluxLimiter<PhiInterpT,VelInterpT>( phiT_, velT_, limiterType_, id, reg );
    }
    ~Builder(){}
  };
  
  ~ConvectiveFluxLimiter();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};


#endif // /ConvectiveFlux_Expr_h
