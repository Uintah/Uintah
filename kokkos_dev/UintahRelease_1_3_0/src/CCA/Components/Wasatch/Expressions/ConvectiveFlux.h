#ifndef ConvectiveFlux_h
#define ConvectiveFlux_h

//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- SpatialOps includes --//
#include <spatialops/OperatorDatabase.h>
#include <spatialops/SpatialFieldStore.h>

/*!
    @class	ConvectiveFlux
    @author James C. Sutherland
    @author Tony Saad
    @date   July, 2010
  
    @brief	Creates an expression for the convective flux of a scalar given a 
 velocity field \f$\mathbf{u}\f$. We write the convective flux in conservation 
 form as \f$ J_i = \rho \varphi u_i = \phi u_i \f$ where \f$i=1,2,3\f$ is the 
 coordinate direction. This requires knowledge of the velocity field. 
 
 Here, we are constructing the convective flux J_i, therefore, it is convenient 
 to set \f$ \rho \varphi \equiv \phi\f$
 
    @par Template Parameters
      <ul>
        <li> \b PhiInterpT The type of operator used in forming 
             \f$\frac{\partial \phi}{\partial x}\f$
        <li> \b VelInterpT The type of operator used in interpolating the
             velocity from volume to face fields
      </ul>
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

private:
  // member variables
	const Expr::Tag phiTag_, velTag_;
	const PhiVolT* phi_; // pointer to const memory location (memory location cannot be modified)
	const VelVolT* vel_;
  PhiInterpT* phiInterpOp_; // pointer to operator
  const VelInterpT* velInterpOp_;
  
public:
	ConvectiveFlux( const Expr::Tag phiTag,
						      const Expr::Tag velTag,
 						      const Expr::ExpressionID& id,
				          const Expr::ExpressionRegistry& reg  );
  ~ConvectiveFlux();
  
	// member functions
	void advertise_dependents( Expr::ExprDeps& exprDeps );
	void bind_fields( const Expr::FieldManagerList& fml );
	void bind_operators( const SpatialOps::OperatorDatabase& opDB );
	void evaluate();
  
	class Builder : public Expr::ExpressionBuilder
	{
    
  private:
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
		
		Expr::ExpressionBase* build( const Expr::ExpressionID& id,
			                           const Expr::ExpressionRegistry& reg ) const;		
	};
	
};

// ###################################################################
//
//                          Implementation
//
// ###################################################################

template< typename PhiInterpT, typename VelInterpT >
ConvectiveFlux<PhiInterpT, VelInterpT>::
ConvectiveFlux( const Expr::Tag phiTag,
                const Expr::Tag velTag,
                const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg  )
: Expr::Expression<PhiFaceT>(id,reg), phiTag_( phiTag ), velTag_( velTag )
{
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
ConvectiveFlux<PhiInterpT, VelInterpT>::
~ConvectiveFlux()
{
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
	exprDeps.requires_expression(phiTag_);
	exprDeps.requires_expression(velTag_);
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
bind_fields( const Expr::FieldManagerList& fml )
{
	const Expr::FieldManager<PhiVolT>& phiVolFM = fml.template field_manager<PhiVolT>(); // get references to field manager
	phi_ = &phiVolFM.field_ref( phiTag_ ); // get reference to phiTag_ and have phi_ point to it

  const Expr::FieldManager<VelVolT>& velVolFM = fml.template field_manager<VelVolT>();
  vel_ = &velVolFM.field_ref( velTag_ );
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  velInterpOp_ = opDB.retrieve_operator<VelInterpT>();
  phiInterpOp_ = opDB.retrieve_operator<PhiInterpT>();
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
void ConvectiveFlux<PhiInterpT, VelInterpT>::evaluate()
{
    PhiFaceT& result = this->value();
  // create a scratch field to hold the temporary interpolated velocity
  // note that PhiFaceT and VelFaceT should on the same mesh location
  SpatialOps::SpatFldPtr<VelFaceT> velInterp = SpatialOps::SpatialFieldStore<VelFaceT>::self().get( result );
  // move the velocity from staggered volume to phi faces
  velInterpOp_->apply_to_field( *vel_, *velInterp );
  // set the advective velocity
  phiInterpOp_->set_advective_velocity( *velInterp );
  
  phiInterpOp_->apply_to_field( *phi_, result );
  result *= *velInterp;
}

//--------------------------------------------------------------------

template< typename PhiInterpT, typename VelInterpT > 
Expr::ExpressionBase* ConvectiveFlux<PhiInterpT, VelInterpT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
	return new ConvectiveFlux<PhiInterpT,VelInterpT>( phiT_, velT_, id, reg );
}


#endif // /ConvectiveFlux_Expr_h