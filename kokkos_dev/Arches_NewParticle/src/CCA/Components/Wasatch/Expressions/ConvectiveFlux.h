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

/**
 *  \ingroup WasatchExpressions
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
 *  \par Template Parameters
 *  <ul>
 *  <li> \b PhiInterpT The type of operator used in forming 
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *  <li> \b VelInterpT The type of operator used in interpolating the
 *       velocity from volume to face fields
 *  </ul>
 */
template< typename PhiInterpT, typename VelInterpT > // scalar interpolant and velocity interpolant
class ConvectiveFlux
  : public Expr::Expression<typename PhiInterpT::DestFieldType>
{  
  // PhiInterpT: an interpolant from staggered or non-staggered volume field to staggered or non-staggered face field
  typedef typename PhiInterpT::SrcFieldType  PhiVolT; // source field is a scalar volume
  typedef typename PhiInterpT::DestFieldType PhiFaceT; // destination field is scalar face
  
  // VelInterpT: an interpolant from Staggered volume field to scalar face field
  typedef typename VelInterpT::SrcFieldType  VelVolT; // source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;
  // the destination field of VelInterpT should be a PhiFaceT
  
protected:

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
  virtual ~ConvectiveFlux();
  
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  
  class Builder : public Expr::ExpressionBuilder
  {
    
  protected:
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
             const Expr::Tag velTag ):phiT_(phiTag), velT_(velTag) {}
		
    virtual Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                         const Expr::ExpressionRegistry& reg ) const;

    virtual ~Builder(){}
  };
};

/**
 *  \ingroup WasatchExpressions
 *  \class   ConvectiveFluxUpwind
 *  \author  James C. Sutherland
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
 *  \par Template Parameters
 *  <ul>
 *  <li> \b PhiInterpT The type of operator used in forming 
 *       \f$\frac{\partial \phi}{\partial x}\f$
 *  <li> \b VelInterpT The type of operator used in interpolating the
 *        velocity from volume to face fields
 *  </ul>
 */
template< typename PhiInterpT, typename VelInterpT > // scalar interpolant and velocity interpolant
class ConvectiveFluxUpwind
  : public ConvectiveFlux<PhiInterpT, VelInterpT>
{  
  // PhiInterpT: an interpolant from staggered or non-staggered volume field to staggered or non-staggered face field
  typedef typename PhiInterpT::SrcFieldType  PhiVolT; // source field is a scalar volume
  typedef typename PhiInterpT::DestFieldType PhiFaceT; // destination field is scalar face
  
  // VelInterpT: an interpolant from Staggered volume field to scalar face field
  typedef typename VelInterpT::SrcFieldType  VelVolT; // source field is always a staggered volume field.
  typedef typename VelInterpT::DestFieldType VelFaceT;
  
public:
  class Builder : public ConvectiveFlux<PhiInterpT,VelInterpT>::Builder
  {
  public:
    /**
     *  \brief Construct an upwind convective flux given an expression
     *         for \f$\phi\f$.
     *
     *  \param phiTag the Expr::Tag for the scalar field.  This is
     *         located at cell centroids.
     *
     *  \param velTag the Expr::Tag for the velocity field.  The
     *         velocity field is a face field.
     */
    Builder( const Expr::Tag phiTag,
             const Expr::Tag velTag ) : ConvectiveFlux<PhiInterpT,VelInterpT>::Builder(phiTag,velTag) {}
    
    Expr::ExpressionBase* build( const Expr::ExpressionID& id,
                                 const Expr::ExpressionRegistry& reg ) const
    {
      return new ConvectiveFluxUpwind<PhiInterpT,VelInterpT>( this->phiT_, this->velT_, id, reg );
    }		
  };
  
  ConvectiveFluxUpwind( const Expr::Tag phiTag,
                        const Expr::Tag velTag,
                        const Expr::ExpressionID& id,
                        const Expr::ExpressionRegistry& reg  );
  ~ConvectiveFluxUpwind();
  
  void evaluate();
};

#endif // /ConvectiveFlux_Expr_h
