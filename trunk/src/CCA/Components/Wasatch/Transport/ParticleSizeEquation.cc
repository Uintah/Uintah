
#include <CCA/Components/Wasatch/Transport/ParticleSizeEquation.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <expression/ExprLib.h>
namespace WasatchCore{


  // #################################################################
  //
  //                          Implementation
  //
  // #################################################################




  ParticleSizeEquation::
  ParticleSizeEquation( const std::string& solnVarName,
                            const Direction pdir,
                            const Expr::TagList& particlePositionTags,
                           const Expr::Tag& particleSizeTag,
                           Uintah::ProblemSpecP particleEqsSpec,
                           GraphCategories& gc )
  : ParticleEquationBase(solnVarName, pdir, particlePositionTags, particleSizeTag, gc),
    pSrcTag_( parse_nametag(particleEqsSpec->findBlock("ParticleSize")->findBlock("SourceTerm")) )
  {
    setup();
  }

  //------------------------------------------------------------------
  
  void ParticleSizeEquation::setup()
  {
    rhsExprID_ = setup_rhs();
    gc_[ADVANCE_SOLUTION]->rootIDs.insert( rhsExprID_ );
  }
  
  //------------------------------------------------------------------
  
  Expr::ExpressionID ParticleSizeEquation::setup_rhs()
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    typedef Expr::ConstantExpr<ParticleField>::Builder RHSBuilder;
    return factory.register_expression( scinew RHSBuilder(rhsTag_, 0.0 ) );
  }
  
  //------------------------------------------------------------------
  
  ParticleSizeEquation::~ParticleSizeEquation()
  {}

  //------------------------------------------------------------------

  Expr::ExpressionID
  ParticleSizeEquation::
  initial_condition( Expr::ExpressionFactory& exprFactory )
  {
    return exprFactory.get_id( initial_condition_tag() );
  }

  //==================================================================

} // namespace Particle
