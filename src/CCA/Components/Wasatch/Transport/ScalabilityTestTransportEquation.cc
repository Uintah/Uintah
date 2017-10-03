/*
 * The MIT License
 *
 * Copyright (c) 2012-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/Transport/ScalabilityTestTransportEquation.h>
#include <CCA/Components/Wasatch/Operators/OperatorTypes.h>
#include <CCA/Components/Wasatch/ParseTools.h>
#include <CCA/Components/Wasatch/TagNames.h>
#include <CCA/Components/Wasatch/Expressions/DiffusiveVelocity.h>
#include <CCA/Components/Wasatch/Expressions/ConvectiveFlux.h>
#include <CCA/Components/Wasatch/Expressions/ScalabilityTestSrc.h>
#include <CCA/Components/Wasatch/ConvectiveInterpolationMethods.h>
#include <CCA/Components/Wasatch/Transport/ParseEquationHelper.h>
#include <CCA/Components/Wasatch/Expressions/MonolithicRHS.h>
#include <CCA/Components/Wasatch/BCHelper.h>
#include <CCA/Components/Wasatch/WasatchBCHelper.h>
//-- ExprLib includes --//
#include <expression/ExprLib.h>

//-- Uintah includes --//
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>

using std::endl;

namespace WasatchCore{

  //------------------------------------------------------------------

  template< typename FieldT >
  void setup_diffusive_velocity_expression( const std::string dir,
                                            const std::string thisPhiName,
                                            Expr::ExpressionFactory& factory,
                                            FieldTagInfo& info )
  {
    const TagNames& tagNames = TagNames::self();
    const Expr::Tag diffFluxTag( thisPhiName + tagNames.diffusiveflux + dir, Expr::STATE_NONE );
    const Expr::Tag phiTag( thisPhiName, Expr::STATE_DYNAMIC );

    Expr::ExpressionBuilder* builder = nullptr;
    FieldSelector fs;

    if( dir=="X" ){
      typedef typename DiffusiveVelocity<typename FaceTypes<FieldT>::XFace>::Builder XFlux;
      builder = scinew XFlux( diffFluxTag,  phiTag, 1.0 );
      fs = DIFFUSIVE_FLUX_X;
    }
    else if( dir=="Y" ){
      typedef typename DiffusiveVelocity<typename FaceTypes<FieldT>::YFace>::Builder YFlux;
      builder = scinew YFlux( diffFluxTag, phiTag, 1.0 );
      fs = DIFFUSIVE_FLUX_Y;
    }
    else if( dir=="Z" ){
      typedef typename DiffusiveVelocity<typename FaceTypes<FieldT>::ZFace>::Builder ZFlux;
      builder = scinew ZFlux( diffFluxTag, phiTag, 1.0 );
      fs = DIFFUSIVE_FLUX_Z;
    }
    else{
      std::ostringstream msg;
      msg << "Invalid direction selection for diffusive flux expression" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    if( builder == nullptr ){
      std::ostringstream msg;
      msg << "Could not build a diffusive velocity expression for '" << thisPhiName << "'" << endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    factory.register_expression( builder );
    info[ fs ] = diffFluxTag;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::
  ScalabilityTestTransportEquation( GraphCategories& gc,
                                    const std::string thisPhiName,
                                    Uintah::ProblemSpecP params )
  : WasatchCore::TransportEquation( gc,
                                thisPhiName,
                                get_staggered_location<FieldT>(),
                                true),   // always constant density
    params_( params )
  {
    setup();
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  ScalabilityTestTransportEquation<FieldT>::
  ~ScalabilityTestTransportEquation()
  {}

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  setup_diffusive_flux( FieldTagInfo& info )
  {
    bool doDiffusion = false;
    params_->get("DoDiffusion",doDiffusion);

    if( doDiffusion ){
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      setup_diffusive_velocity_expression<FieldT>( "X", solnVarName_, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Y", solnVarName_, factory, info );
      setup_diffusive_velocity_expression<FieldT>( "Z", solnVarName_, factory, info );
    }
    else{
      info[DIFFUSIVE_FLUX_X] = Expr::Tag();
      info[DIFFUSIVE_FLUX_Y] = Expr::Tag();
      info[DIFFUSIVE_FLUX_Z] = Expr::Tag();
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  setup_convective_flux( FieldTagInfo& info )
  {
    bool doConvection = false;
    params_->get("DoConvection",doConvection);
    if( doConvection ){
      const Expr::Tag empty;
      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;

      setup_convective_flux_expression<FieldT>( "X",
                                                solnVarTag_,
                                                empty, // convective flux (empty to build it)
                                                CENTRAL,
                                                parse_nametag( params_->findBlock("X-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
      setup_convective_flux_expression<FieldT>( "Y",
                                                solnVarTag_,
                                                empty, // convective flux (empty to build it)
                                                CENTRAL,
                                                parse_nametag( params_->findBlock("Y-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
      setup_convective_flux_expression<FieldT>( "Z",
                                                solnVarTag_,
                                                empty, // convective flux (empty to build it)
                                                CENTRAL,
                                                parse_nametag( params_->findBlock("Z-Velocity" )->findBlock( "NameTag" ) ),
                                                factory,
                                                info );
     }
    else{
      info[CONVECTIVE_FLUX_X] = Expr::Tag();
      info[CONVECTIVE_FLUX_Y] = Expr::Tag();
      info[CONVECTIVE_FLUX_Z] = Expr::Tag();
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  setup_source_terms( FieldTagInfo& info, Expr::TagList& sourceTags )
  {
    bool doSrc = false;
    params_->get("DoSourceTerm",doSrc);
    Expr::Tag srcTag;

    if( doSrc ){
      Uintah::ProblemSpecP srcSpec = params_->findBlock("DoSourceTerm");
      std::string kind;
      srcSpec->getAttribute("kind", kind);

      Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
      srcTag = Expr::Tag( solnVarName_ + "_src", Expr::STATE_NONE );
      
      if (kind == "COUPLED") {
        int nEqs=0;
        params_->get( "NumberOfEquations", nEqs );
        
        std::string basePhiName;
        params_->get( "SolutionVariable", basePhiName );
        const Expr::Tag basePhiTag ( basePhiName, Expr::STATE_DYNAMIC );        
        typedef typename ScalabilityTestSrc<FieldT>::Builder coupledSrcTerm;
        factory.register_expression( scinew coupledSrcTerm( srcTag, basePhiTag, nEqs) );
      } else {
        typedef typename ScalabilityTestSrcUncoupled<FieldT>::Builder uncoupledSrcTerm;
        factory.register_expression( scinew uncoupledSrcTerm( srcTag, this->solution_variable_tag()) );
      }
    }
    info[SOURCE_TERM] = srcTag;
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  Expr::ExpressionID ScalabilityTestTransportEquation<FieldT>::
  setup_rhs( FieldTagInfo& info,
             const Expr::TagList& srcTags  )
  {
    Expr::ExpressionFactory& factory = *gc_[ADVANCE_SOLUTION]->exprFactory;
    bool monolithic = false;
    if( params_->findBlock("MonolithicRHS") )  params_->get("MonolithicRHS",monolithic);
    if( monolithic ){
      proc0cout << "ScalabilityTestTransportEquation " << solnVarName_ << " MONOLITHIC RHS ACTIVE - diffusion is always on!" << endl;
      const Expr::Tag dcoefTag( solnVarName_+"DiffCoeff", Expr::STATE_NONE );
      factory.register_expression( scinew typename Expr::ConstantExpr<FieldT>::Builder( dcoefTag, 1.0 ) );
      return factory.register_expression(
          scinew typename MonolithicRHS<FieldT>::
          Builder( Expr::Tag(solnVarName_+"_rhs", Expr::STATE_NONE),
                   dcoefTag,
                   info.find(CONVECTIVE_FLUX_X)->second,
                   info.find(CONVECTIVE_FLUX_Y)->second,
                   info.find(CONVECTIVE_FLUX_Z)->second,
                   solnVarTag_,
                   info.find(SOURCE_TERM)->second ) );
    }
    else{
      const Expr::Tag densT;
      const bool tempConstDens = false;
      Expr::Tag srcTag = info.find(SOURCE_TERM)->second;
      if (srcTag == Expr::Tag()) {
        return factory.register_expression( scinew typename ScalarRHS<FieldT>::Builder( rhsTag_, info, densT, tempConstDens) );
      } else {
        return factory.register_expression( scinew typename ScalarRHS<FieldT>::Builder( rhsTag_, info, tag_list(srcTag), densT, tempConstDens) );
      }
    }
  }

  //------------------------------------------------------------------

  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  apply_initial_boundary_conditions( const GraphHelper& graphHelper,
                                    WasatchBCHelper& bcHelper )
  {}
  
  //------------------------------------------------------------------
  
  template< typename FieldT >
  void ScalabilityTestTransportEquation<FieldT>::
  apply_boundary_conditions( const GraphHelper& graphHelper,
                             WasatchBCHelper& bcHelper )
  {}

  //------------------------------------------------------------------

  template<typename FieldT>
  Expr::ExpressionID
  ScalabilityTestTransportEquation<FieldT>::
  initial_condition( Expr::ExpressionFactory& icFactory )
  {
    return icFactory.get_id( initial_condition_tag() );
  }

  //------------------------------------------------------------------

  //==================================================================
  // Explicit template instantiation
  template class ScalabilityTestTransportEquation< SVolField >;
  //==================================================================


} // namespace WasatchCore
