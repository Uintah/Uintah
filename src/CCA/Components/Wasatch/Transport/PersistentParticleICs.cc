
#include <CCA/Components/Wasatch/Transport/PersistentParticleICs.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <expression/ExprLib.h>
namespace WasatchCore{


#ifndef PI
#  define PI 3.1415926535897932384626433832795
#endif

/**
 *  \class ParticleMassIC
 *  \ingroup WasatchParticles
 *  \brief Initial condition for particle mass jtm: This is duplicated from ParticleMassEquation.cc
 *  todo: remove duplicated code here
 */
class ParticleMassIC : public Expr::Expression<ParticleField>
{
  DECLARE_FIELDS(ParticleField, pRho_, pDiameter_)

    ParticleMassIC( const Expr::Tag& pRhoTag,
                    const Expr::Tag& pDiameterTag )
      : Expr::Expression<ParticleField>()
    {
      this->set_gpu_runnable(true);
      pRho_ = create_field_request<ParticleField>(pRhoTag);
      pDiameter_ = create_field_request<ParticleField>(pDiameterTag);
    }

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a ParticleMassIC expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& pRhoTag,
             const Expr::Tag& pDiameterTag )
    : ExpressionBuilder( resultTag ),
      pRhoTag_( pRhoTag ),
      pDiameterTag_( pDiameterTag )
    {}

    Expr::ExpressionBase* build() const{
      return new ParticleMassIC( pRhoTag_,pDiameterTag_ );
    }

  private:
    const Expr::Tag pRhoTag_, pDiameterTag_;
  };

  void evaluate()
  {
    using namespace SpatialOps;
    ParticleField& result = this->value();
    const ParticleField& pRho = pRho_->field_ref();
    const ParticleField& pDiameter = pDiameter_->field_ref();
    result <<= (PI/6.0) * pRho * pow(pDiameter,3);
  }
};


  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################




  PersistentInitialParticleSize::
  PersistentInitialParticleSize( const std::string&   solnVarName,
                                 const Expr::TagList& particlePositionTags,
                                 const Expr::Tag&     particleSizeTag,
                                 GraphCategories&     gc )
  : ParticleEquationBase(solnVarName, WasatchCore::NODIR, particlePositionTags, particleSizeTag, gc)
  {
    setup();
  }

  //------------------------------------------------------------------

  void PersistentInitialParticleSize::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }

  //------------------------------------------------------------------

  Expr::ExpressionID PersistentInitialParticleSize::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    typedef Expr::ConstantExpr<ParticleField>::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, 0.0 ) );
  }

  //------------------------------------------------------------------

  PersistentInitialParticleSize::~PersistentInitialParticleSize()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  PersistentInitialParticleSize::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    exprFactory.register_expression( scinew Expr::ConstantExpr<ParticleField>::Builder(initial_condition_tag(),1e-4));
    return exprFactory.get_id( initial_condition_tag() );
  }

  //==================================================================


  PersistentInitialParticleMass::
   PersistentInitialParticleMass( const std::string&   solnVarName,
                                  const Expr::TagList& particlePositionTags,
                                  const Expr::Tag&     particleSizeTag,
                                  Uintah::ProblemSpecP particleEqsSpec,
                                  GraphCategories&     gc )
   : ParticleEquationBase( solnVarName, WasatchCore::NODIR, particlePositionTags, particleSizeTag, gc ),
     pRhoTag_( parse_nametag(particleEqsSpec->findBlock("ParticleDensity")) )
   {
     setup();
   }

   //------------------------------------------------------------------

   void PersistentInitialParticleMass::setup()
   {
     rhsExprID_ = setup_rhs();
     gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
   }

   //------------------------------------------------------------------

   Expr::ExpressionID PersistentInitialParticleMass::setup_rhs()
   {
     Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
     typedef Expr::ConstantExpr<ParticleField>::Builder RHSBuilder;
     return factory.register_expression( scinew RHSBuilder(rhsTag_, 0.0 ) );
   }

   //------------------------------------------------------------------

   PersistentInitialParticleMass::~PersistentInitialParticleMass()
   {}

   //------------------------------------------------------------------

   Expr::ExpressionID
   PersistentInitialParticleMass::
   initial_condition( Expr::ExpressionFactory& exprFactory )
   {
     // register expression to calculate the momentum initial condition from the initial conditions on
     // velocity and density in the cases that we are initializing velocity in the input file
     const Expr::Tag pRhoInit(pRhoTag_.name(), Expr::STATE_NONE);
     const Expr::Tag pSizeInit(pSizeTag_.name(), Expr::STATE_NONE);
     exprFactory.register_expression( new ParticleMassIC::Builder( initial_condition_tag(),
                                                                   pRhoInit, pSizeInit ) );
     return exprFactory.get_id( initial_condition_tag() );
   }

} // namespace Particle
