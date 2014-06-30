#ifndef ParticleResponseTime_Expr_h
#define ParticleResponseTime_Expr_h

#include <expression/Expression.h>
//==================================================================
/**
 *  \class ParticleResponseTime
    \author Tony Saad, ODT
    \date June 2014
 *  \brief Calculates the particle Response time \f$\tau_\text{p}\f$. 
\f[ 
  \tau_\text{p} \equiv \frac{ \rho_\text{p} }{ 18 \mu_\text{g} }
 \f]
 */

template< typename ViscT >
class ParticleResponseTime
 : public Expr::Expression<ParticleField>
{

  const Expr::Tag pDensityTag_, pSizeTag_, gViscTag_;
  const Expr::TagList pPosTags_;
  const ParticleField  *pdensity_,  *psize_, *px_, *py_, *pz_ ;
  const ViscT *gVisc_;

  typedef typename SpatialOps::Particle::CellToParticle<ViscT> Scal2POpT;
  Scal2POpT* sOp_;

  ParticleResponseTime( const Expr::Tag& particleDensityTag,
                        const Expr::Tag& particleSizeTag,
                        const Expr::Tag& gasViscosityTag,
                       const Expr::TagList& particlePositionTags);

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder(const Expr::Tag& resultTag,
            const Expr::Tag& particleDensityTag,
            const Expr::Tag& particleSizeTag,
            const Expr::Tag& gasViscosityTag,
            const Expr::TagList& particlePositionTags );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
  private:
    const Expr::Tag pDensityTag_, pSizeTag_, gViscTag_;
    const Expr::TagList pPosTags_;
  };

  ~ParticleResponseTime();

  void advertise_dependents( Expr::ExprDeps& exprDeps);
  void bind_fields( const Expr::FieldManagerList& fml );
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
};

 // ###################################################################
  //
  //                          Implementation
  //
  // ###################################################################

template< typename ViscT >
ParticleResponseTime<ViscT>::
ParticleResponseTime( const Expr::Tag& particleDensityTag,
                     const Expr::Tag& particleSizeTag,
                     const Expr::Tag& gasViscosityTag,
                     const Expr::TagList& particlePositionTags )
  : Expr::Expression<ParticleField>(),
    pDensityTag_( particleDensityTag ),
    pSizeTag_   ( particleSizeTag    ),
    gViscTag_ ( gasViscosityTag   ),
    pPosTags_ (particlePositionTags)
{}

//--------------------------------------------------------------------

template< typename ViscT >
ParticleResponseTime<ViscT>::~ParticleResponseTime()
{}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( pDensityTag_ );
  exprDeps.requires_expression( pSizeTag_ );
  exprDeps.requires_expression( pPosTags_[0]  );
  exprDeps.requires_expression( pPosTags_[1]  );
  exprDeps.requires_expression( pPosTags_[2]  );
  exprDeps.requires_expression( gViscTag_ );
}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::bind_fields( const Expr::FieldManagerList& fml )
{
  const typename Expr::FieldMgrSelector<ParticleField>::type& fm = fml.field_manager<ParticleField>();
  
  pdensity_ = &fm.field_ref( pDensityTag_ );
  psize_    = &fm.field_ref( pSizeTag_    );
  
  px_     = &fm.field_ref( pPosTags_[0]     );
  py_     = &fm.field_ref( pPosTags_[1]     );
  pz_     = &fm.field_ref( pPosTags_[2]     );
  
  gVisc_ = &fml.field_ref<ViscT>( gViscTag_ );
}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  sOp_ = opDB.retrieve_operator<Scal2POpT>();
}

//--------------------------------------------------------------------

template< typename ViscT >
void
ParticleResponseTime<ViscT>::evaluate()
{
  ParticleField& result = this->value();
  SpatialOps::SpatFldPtr<ParticleField> tmpvisc = SpatialOps::SpatialFieldStore::get<ParticleField>( result );
  
  sOp_->set_coordinate_information(px_,py_,pz_,psize_);
  sOp_->apply_to_field(*gVisc_, *tmpvisc );

  using namespace SpatialOps;
  result <<= *pdensity_ * *psize_ * *psize_ / ( 18.0 * *tmpvisc );
}

//--------------------------------------------------------------------

template< typename ViscT >
ParticleResponseTime<ViscT>::
Builder::Builder( const Expr::Tag& resultTag,
                 const Expr::Tag& particleDensityTag,
                 const Expr::Tag& particleSizeTag,
                 const Expr::Tag& gasViscosityTag,
                 const Expr::TagList& particlePositionTags)
  : ExpressionBuilder( resultTag ),
    pDensityTag_( particleDensityTag ),
    pSizeTag_   ( particleSizeTag    ),
    gViscTag_ ( gasViscosityTag    ),
    pPosTags_ (particlePositionTags)
{}

//--------------------------------------------------------------------

template< typename ViscT >
Expr::ExpressionBase*
ParticleResponseTime<ViscT>::Builder::build() const
{
  return new ParticleResponseTime( pDensityTag_, pSizeTag_, gViscTag_, pPosTags_ );
}

#endif // ParticleResponseTime_Expr_h
