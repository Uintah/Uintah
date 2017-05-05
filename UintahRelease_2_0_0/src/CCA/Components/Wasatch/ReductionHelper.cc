/**
 *  \file   ReductionHelper.cc
 *  \date   June, 2013
 *  \author "Tony Saad"
 *
 *
 * The MIT License
 *
 * Copyright (c) 2013-2017 The University of Utah
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

#include "ReductionHelper.h"

//-- Boost includes --//
#include <boost/foreach.hpp>

//-- ExprLib includes --//
#include <expression/Expression.h>
#include <expression/ExpressionFactory.h>

//-- Wasatch includes --//
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <CCA/Components/Wasatch/Wasatch.h>
#include <CCA/Components/Wasatch/FieldTypes.h>
#include <CCA/Components/Wasatch/Expressions/ReductionBase.h>
#include <CCA/Components/Wasatch/Expressions/Reduction.h>
#include <CCA/Components/Wasatch/ParseTools.h>

//-- Uintah Includes --//
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Variables/Reductions.h>
#include <Core/Grid/Variables/VarTypes.h>

namespace so = SpatialOps;

namespace WasatchCore {

  //==================================================================

  ReductionHelper::ReductionHelper()
  {
    wasatchSync_ = false;
    hasDoneSetup_ = false;
  }

  //------------------------------------------------------------------

  ReductionHelper&
  ReductionHelper::self()
  {
    static ReductionHelper redHelp;
    return redHelp;
  }

  //------------------------------------------------------------------

  ReductionHelper::~ReductionHelper()
  {}

  //------------------------------------------------------------------

  template< typename SrcT, typename ReductionOpT >
  void
  ReductionHelper::add_variable( const Category category,
                                 const Expr::Tag& resultTag,
                                 const Expr::Tag& srcTag,
                                 const bool printVar,
                                 const bool reduceOnAllRKStages )
  {
    reduceOnAllRKStages_ = reduceOnAllRKStages;
    proc0cout << "adding reduction variable " << resultTag << std::endl;
    if( hasDoneSetup_ ){
      std::ostringstream msg;
      msg << "ReductionHelper error: cannot add new variables after tasks have been registered!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    if( !wasatchSync_ ){
      std::ostringstream msg;
      msg << "ReductionHelper error: must call sync_with_wasatch() prior to adding variables!" << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }

    Expr::ExpressionFactory& factory = *(wasatch_->graph_categories()[category]->exprFactory);

    // if this expression has already been registered, then return
    if( factory.have_entry( resultTag ) )
      return;

    const Expr::ExpressionID myID = factory.register_expression( new typename Reduction< SrcT, ReductionOpT >::Builder(resultTag,
                                                                                                srcTag,
                                                                                                printVar) );

    ReductionBase::reductionTagList.insert(std::pair< Expr::Tag, bool >(resultTag, reduceOnAllRKStages) );
    factory.cleave_from_parents(myID);
    factory.cleave_from_children(myID);

    (wasatch_->graph_categories()[category]->rootIDs).insert( myID );

    // don't allow the ExpressionTree to reclaim memory for this field since
    // it will need to be seen by the task that copies it to the "old" value.
    if( wasatch_->persistent_fields().find( srcTag.name() ) == wasatch_->persistent_fields().end() ){
      wasatch_->persistent_fields().insert( srcTag.name() );
    }

    if( wasatch_->persistent_fields().find( resultTag.name() ) == wasatch_->persistent_fields().end() ){
      wasatch_->persistent_fields().insert( resultTag.name() );
    }
  }

  //------------------------------------------------------------------

  void
  ReductionHelper::sync_with_wasatch( Wasatch* const wasatch )
  {
    wasatch_ = wasatch;
    wasatchSync_ = true;
  }

  //------------------------------------------------------------------
  
  ReductionEnum
  ReductionHelper::select_reduction_enum( const std::string& strRedOp )
  {
    ReductionEnum redEnum;
    if      (strRedOp.compare("min")==0) redEnum = ReduceMin;
    else if (strRedOp.compare("max")==0) redEnum = ReduceMax;
    else if (strRedOp.compare("sum")==0) redEnum = ReduceSum;
    else {
      std::ostringstream msg;
      msg << "ERROR: unsupported reduction operatioin specified: '" << strRedOp << "'. ";
      msg << "Supported reduction operations are: min, max, and sum." << std::endl;
      throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
    }
    return redEnum;
  }

  //------------------------------------------------------------------
  
  void
  ReductionHelper::parse_reduction_spec( Uintah::ProblemSpecP wasatchSpec )
  {    
    for( Uintah::ProblemSpecP reductionSpec=wasatchSpec->findBlock("Reduction");
        reductionSpec != nullptr;
        reductionSpec=reductionSpec->findNextBlock("Reduction") ) {
      
      const Category cat = parse_tasklist(reductionSpec,true);
      
      std::string reductionOperation;
      reductionSpec->getAttribute("op",reductionOperation);
      ReductionEnum redEnum = select_reduction_enum(reductionOperation);
      
      bool printVar=false;
      reductionSpec->getAttribute("output",printVar);
      
      bool perStage=false;
      reductionSpec->getAttribute("perstage",perStage);
      
      const Expr::Tag resultTag = parse_nametag(reductionSpec->findBlock("NameTag"));
      
      Uintah::ProblemSpecP srcFldParams = reductionSpec->findBlock("Source");
      std::string fieldType;
      srcFldParams->getAttribute("type",fieldType);
      const Expr::Tag srcTag = parse_nametag(srcFldParams->findBlock("NameTag"));
      
      switch( get_field_type(fieldType) ){
        case SVOL :
        {
          switch (redEnum) {
            case ReduceMin: self().add_variable<SVolField,ReductionMinOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceMax: self().add_variable<SVolField,ReductionMaxOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceSum: self().add_variable<SVolField,ReductionSumOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            default:
            {
              std::ostringstream msg;
              msg << "ERROR: unsupported reduction operation specified: '" << reductionOperation <<"'. ";
              msg << "Supported reduction operations are: min, max, and sum." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
              break;
          }
        }
          break;
        case XVOL :
        {
          switch (redEnum) {
            case ReduceMin: self().add_variable<XVolField,ReductionMinOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceMax: self().add_variable<XVolField,ReductionMaxOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceSum: self().add_variable<XVolField,ReductionSumOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            default:
            {
              std::ostringstream msg;
              msg << "ERROR: unsupported reduction operation specified: '" << reductionOperation <<"'. ";
              msg << "Supported reduction operations are: min, max, and sum." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
              break;
          }
        }        
          break;
        case YVOL :
        {
          switch (redEnum) {
            case ReduceMin: self().add_variable<YVolField,ReductionMinOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceMax: self().add_variable<YVolField,ReductionMaxOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceSum: self().add_variable<YVolField,ReductionSumOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            default:
            {
              std::ostringstream msg;
              msg << "ERROR: unsupported reduction operation specified: '" << reductionOperation <<"'. ";
              msg << "Supported reduction operations are: min, max, and sum." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
              break;
          }
        }
          break;
        case ZVOL :
        {
          switch (redEnum) {
            case ReduceMin: self().add_variable<ZVolField,ReductionMinOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceMax: self().add_variable<ZVolField,ReductionMaxOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceSum: self().add_variable<ZVolField,ReductionSumOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            default:
            {
              std::ostringstream msg;
              msg << "ERROR: unsupported reduction operation specified: '" << reductionOperation <<"'. ";
              msg << "Supported reduction operations are: min, max, and sum." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
              break;
          }
        }
          break;
        case PERPATCH :
        {
          switch (redEnum) {
            case ReduceMin: self().add_variable<so::SingleValueField, ReductionMinOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceMax: self().add_variable<so::SingleValueField, ReductionMaxOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            case ReduceSum: self().add_variable<so::SingleValueField, ReductionSumOpT>(cat, resultTag, srcTag, printVar, perStage); break;
            default:
            {
              std::ostringstream msg;
              msg << "ERROR: unsupported reduction operation specified: '" << reductionOperation <<"'. ";
              msg << "Supported reduction operations are: min, max, and sum." << std::endl;
              throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
            }
              break;
          }
        }
          break;          
        default:
          std::ostringstream msg;
          msg << "ERROR: unsupported field type '" << fieldType << "'" << std::endl;
          throw Uintah::ProblemSetupException( msg.str(), __FILE__, __LINE__ );
      }      
    }
  }
  
  //------------------------------------------------------------------
  
  void
  ReductionHelper::schedule_tasks( const Uintah::LevelP& level,
                                   Uintah::SchedulerP sched,
                                   const Uintah::MaterialSet* const materials,
                                   const Expr::ExpressionTree::TreePtr tree,
                                   const int patchID,
                                   const int rkStage )
  {
    // go through reduction variables that are computed in this Wasatch Task
    // and insert a Uintah task immediately after.
    Expr::ExpressionFactory& factory = tree->get_expression_factory();
    typedef std::map<Expr::Tag, bool> MapT;
    BOOST_FOREACH( MapT::value_type& pair, ReductionBase::reductionTagList ){
      const Expr::Tag rtag = pair.first;
      const bool compute = (pair.second) ? true : rkStage==1;
      if (tree->computes_field( rtag) && compute) {
        ReductionBase& redExpr = dynamic_cast<ReductionBase&>( factory.retrieve_expression( rtag, patchID, false ) );
        redExpr.schedule_set_reduction_vars( level, sched, materials, rkStage  );
      }
    }    
  }

} /* namespace WasatchCore */

//==========================================================================
// Explicit template instantiation for supported versions of this expression
#include <spatialops/structured/FVStaggered.h>

#define DECLARE_REDUCTION_VARIANTS(T) \
template void WasatchCore::ReductionHelper::add_variable<T,ReductionMinOpT>( const Category, const Expr::Tag&, const Expr::Tag&, const bool, const bool ); \
template void WasatchCore::ReductionHelper::add_variable<T,ReductionMaxOpT>( const Category, const Expr::Tag&, const Expr::Tag&, const bool, const bool ); \
template void WasatchCore::ReductionHelper::add_variable<T,ReductionSumOpT>( const Category, const Expr::Tag&, const Expr::Tag&, const bool, const bool );

DECLARE_REDUCTION_VARIANTS(so::SingleValueField)
DECLARE_REDUCTION_VARIANTS(SVolField)
DECLARE_REDUCTION_VARIANTS(XVolField)
DECLARE_REDUCTION_VARIANTS(YVolField)
DECLARE_REDUCTION_VARIANTS(ZVolField)
//====================================================================
