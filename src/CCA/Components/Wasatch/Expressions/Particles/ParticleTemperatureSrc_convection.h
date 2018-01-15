#ifndef ParticleTemperatureSrc_convection_Expr_h
#define ParticleTemperatureSrc_convection_Expr_h

#include <expression/Expression.h>
#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/particles/ParticleOperators.h>
#include <spatialops/OperatorDatabase.h>

  /**
   *  \class ParticleTemperatureSrc_convection
   *  \ingroup WasatchParticles
   *  \brief Evaluates the convection term in the particle temperature equation.
   *
   *  \author Naveen Punati, Josh McConnell
   *
   *
   */

  template< typename FieldT >
  class ParticleTemperatureSrc_convection
  : public Expr::Expression<ParticleField>
  {
    DECLARE_FIELDS( ParticleField, coefh_, pxpos_, pmass_, pRe_, ptemp_, psize_, pcp_ )
    DECLARE_FIELDS( FieldT, gtemp_ )

    typedef typename SpatialOps::Particle::CellToParticle<FieldT> C2POpT;
    C2POpT* c2pOp_;

    ParticleTemperatureSrc_convection(const Expr::Tag& coefhtag,
                                      const Expr::Tag& pmassTag,
                                      const Expr::Tag& ptempTag,
                                      const Expr::Tag& psizeTag,
                                      const Expr::Tag& pcoalcpTag,
                                      const Expr::Tag& gtempTag );

  public:
    class Builder : public Expr::ExpressionBuilder
    {
    public:
     Builder( const Expr::Tag& pTempConvTag,
              const Expr::Tag& coefhtag,
              const Expr::Tag& pmassTag,
              const Expr::Tag& ptempTag,
              const Expr::Tag& psizeTag,
              const Expr::Tag& pcoalcpTag,
              const Expr::Tag& gtempTag );
        
        
      ~Builder(){}
      Expr::ExpressionBase* build() const;
    private:
      const Expr::Tag coefhtag_, pmassTag_,ptempTag_,psizeTag_,pcoalcpTag_,gtempTag_;
    };

    ParticleTemperatureSrc_convection(){}
    void bind_operators( const SpatialOps::OperatorDatabase& opDB );
    void evaluate();

  };

  //---------------------------------------------------------------------
  //
  //                   Implementation
  //
  //---------------------------------------------------------------------
  template<typename FieldT>
  ParticleTemperatureSrc_convection<FieldT>::
  ParticleTemperatureSrc_convection(const Expr::Tag& coefhtag,
                                    const Expr::Tag& pmassTag,
                                    const Expr::Tag& ptempTag,
                                    const Expr::Tag& psizeTag,
                                    const Expr::Tag& pcoalcpTag,
                                    const Expr::Tag& gtempTag)
      : Expr::Expression<ParticleField>()
  {
    this->set_gpu_runnable(false);  // waiting for GPU-enabled particle interpolants
    
    coefh_ = this->template create_field_request<ParticleField>( coefhtag   );
    pmass_ = this->template create_field_request<ParticleField>( pmassTag   );
    ptemp_ = this->template create_field_request<ParticleField>( ptempTag   );
    psize_ = this->template create_field_request<ParticleField>( psizeTag   );
    pcp_   = this->template create_field_request<ParticleField>( pcoalcpTag );

    gtemp_ = this->template create_field_request<FieldT>( gtempTag );
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  ParticleTemperatureSrc_convection<FieldT>::
  bind_operators( const SpatialOps::OperatorDatabase& opDB )
  {
    c2pOp_ = opDB.retrieve_operator<C2POpT>();
  }

  //------------------------------------------------------------------

  template<typename FieldT>
  void
  ParticleTemperatureSrc_convection<FieldT>::
  evaluate()
  {
    using namespace SpatialOps;

    ParticleField& convterm = this->value();

    const ParticleField& pmass = pmass_->field_ref();
      
    const ParticleField& coefh = coefh_->field_ref();

    const ParticleField& ptemp = ptemp_->field_ref();
    const ParticleField& psize = psize_->field_ref();
    const ParticleField& pcp   = pcp_  ->field_ref();

    const FieldT& gtemp   = gtemp_  ->field_ref();


    //interpolate gas phase temperature to particle locations
    SpatialOps::SpatFldPtr<ParticleField> int_temp = SpatialOps::SpatialFieldStore::get<ParticleField>( convterm );
    c2pOp_->apply_to_field( gtemp, *int_temp );

    //time to construct the convective heat transfer term for particle temperature equation
    // ( A_p / (m_p*c_p)) * h * (T_g-T_p)
    convterm <<= ((3.1416 * pow(psize,2)) / (pmass * pcp) ) * ( coefh * ( *int_temp - ptemp)) ;

  }

  //------------------------------------------------------------------

  template<typename FieldT>
  ParticleTemperatureSrc_convection<FieldT>::
  Builder::Builder( const Expr::Tag& pTempConvTag,
                    const Expr::Tag& coefhtag,
                    const Expr::Tag& pmassTag,
                    const Expr::Tag& ptempTag,
                    const Expr::Tag& psizeTag,
                    const Expr::Tag& pcoalcpTag,
                    const Expr::Tag& gtempTag)
  : ExpressionBuilder( pTempConvTag ),
    coefhtag_  ( coefhtag  ),
    pmassTag_  ( pmassTag  ),
    ptempTag_  ( ptempTag  ),
    psizeTag_  ( psizeTag  ),
    pcoalcpTag_( pcoalcpTag),
    gtempTag_  ( gtempTag  )
  {}

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionBase*
  ParticleTemperatureSrc_convection<FieldT>::Builder::build() const
  {
    return new ParticleTemperatureSrc_convection<FieldT>( coefhtag_,pmassTag_,ptempTag_,psizeTag_,pcoalcpTag_,gtempTag_ );
  }

#endif // ParticleTemperatureSrc_convection_Expr_h
