/*
 * The MIT License
 *
 * Copyright (c) 2012-2016 The University of Utah
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

#ifndef NestedExpression_h
#define NestedExpression_h

#include <expression/ExprLib.h>
#include <expression/Expression.h>
#include <expression/ExpressionFactory.h>
#include <CCA/Components/Wasatch/Expressions/ExprAlgebra.h>
#include <expression/Functions.h>
#include <expression/ManagerTypes.h>

#include <spatialops/structured/SpatialFieldStore.h>
#include <spatialops/OperatorDatabase.h>

#include <CCA/Components/Wasatch/NestedGraphHelper.h>
#include <CCA/Components/Wasatch/PatchInfo.h>
#include <CCA/Components/Wasatch/FieldAdaptor.h>
#include <spatialops/structured/FieldHelper.h>

namespace WasatchCore{

template< typename FieldT >
class NestedExpression : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, f_)

  NestedGraphHelper     helper_;
  UintahPatchContainer* patchContainer_;
  Expr::ExpressionTree* treePtr_;

  bool setupHasRun_;

  const Expr::Tag& resultTag_;
  const Expr::Tag intermediateTag_;
  Expr::TagList inputTags_;

  const unsigned numIter_; // maximum number of iterations

  NestedExpression( const Expr::Tag&     resultTag,
                    const Expr::TagList& inputTags,
                    const unsigned       maxIter );

  void setup();
  Expr::IDSet register_local_expressions();
  void set_initial_guesses();
  void bind_operators( const SpatialOps::OperatorDatabase& opDB );
  void evaluate();
  ~NestedExpression();

public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    /**
     *  @brief Build a NestedExpression expression
     *  @param resultTag the tag for the value that this expression computes
     */
    Builder( const Expr::Tag&     resultTag,
             const Expr::TagList& inputTags,
             const unsigned        maxIter );

    Expr::ExpressionBase* build() const{
      return scinew NestedExpression<FieldT>( resultTag_, inputTags_, numIter_ );
    }

    private:
      const Expr::Tag resultTag_;
      const Expr::TagList inputTags_;
      const unsigned numIter_; // maximum number of iterations
    };
  };


// ###################################################################
//
//                          Implementation
//
// ###################################################################

template<typename FieldT>
NestedExpression<FieldT>::
NestedExpression( const Expr::Tag&     resultTag,
                  const Expr::TagList& inputTags,
                  const unsigned       maxIter )
  : Expr::Expression<FieldT>(),
    helper_(),
    treePtr_      ( nullptr ),
    setupHasRun_  ( false   ),
    resultTag_    (resultTag),
    intermediateTag_( "intermediate", Expr::STATE_NONE ),
    inputTags_    (inputTags),
    numIter_      ( maxIter )
{
  this->set_gpu_runnable(true);
  this->template create_field_vector_request<FieldT>(inputTags, f_);
}

//--------------------------------------------------------------------

template<typename FieldT>
NestedExpression<FieldT>::
Builder::Builder( const Expr::Tag&     resultTag,
                  const Expr::TagList& inputTags,
                  const unsigned       maxIter )
  : ExpressionBuilder( resultTag ),
    resultTag_  ( resultTag ),
    inputTags_  ( inputTags ),
    numIter_    ( maxIter   )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::IDSet
NestedExpression<FieldT>::
register_local_expressions( )
{
  Expr::IDSet rootIDs;
  Expr::ExpressionFactory& factory = *helper_.factory_;

  typedef typename Expr::PlaceHolder<FieldT>::Builder PHBuilder;
  for(const Expr::Tag& tag : inputTags_ ){
    factory.register_expression( new PHBuilder(tag) );
  }

  typedef ExprAlgebra<FieldT> Algebra;
  factory.register_expression( new typename Algebra::Builder(intermediateTag_, inputTags_, Algebra::SUM) );

  typedef typename Expr::LinearFunction<FieldT>::Builder LinBuilder;
  rootIDs.insert(factory.register_expression( new LinBuilder(resultTag_, intermediateTag_, 1., 0.) ));

  return rootIDs;
}


//--------------------------------------------------------------------

template< typename FieldT >
void
NestedExpression<FieldT>::
setup()
{
  using namespace SpatialOps;

  const Uintah::Patch* patch = patchContainer_->get_uintah_patch();

  helper_.set_alloc_info(patch, this->is_gpu_runnable());

  IDSet rootIDs = register_local_expressions();
  treePtr_ = helper_.new_tree("nested_tree", rootIDs);

  helper_.finalize();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
NestedExpression<FieldT>::
set_initial_guesses()
{
  Expr::UintahFieldManager<FieldT>& fieldManager = helper_.fml_-> template field_manager<FieldT>();
  for(int i=0; i<inputTags_.size(); ++i)
  {
    FieldT& fLocal = fieldManager.field_ref( inputTags_[i] );
    const FieldT& f = f_[i]->field_ref();
    fLocal <<= f;
  }
}

//--------------------------------------------------------------------
template < typename FieldT >
void
NestedExpression<FieldT>::
bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  patchContainer_ = opDB.retrieve_operator<UintahPatchContainer>();
}

//--------------------------------------------------------------------

template< typename FieldT >
void
NestedExpression<FieldT>::
evaluate()
{
  using namespace SpatialOps;
  
  if( !setupHasRun_ ){ setup(); setupHasRun_=true; }

   set_initial_guesses();

  int numIter = 0;

  while( numIter<numIter_ )
  {
    treePtr_->execute_tree();
    ++numIter;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
NestedExpression<FieldT>::
~NestedExpression()
{}

//====================================================================
}//namespace WasatchCore

#endif /*NestedExpression_h */
