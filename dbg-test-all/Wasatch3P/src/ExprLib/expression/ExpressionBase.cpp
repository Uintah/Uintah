/*
 * Copyright (c) 2011 The University of Utah
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
#include <set>
#include <stdexcept>
#include <sstream>

#include <expression/ExpressionBase.h>
#include <expression/ExprDeps.h>
#include <expression/FieldDeps.h>

#include <boost/foreach.hpp>

namespace Expr{

#define DEFAULT_FML_ID -9999999
//#define DEBUG_CUDA_HANDLES

//--------------------------------------------------------------------

ExpressionBase::
ExpressionBase()
  : cleaveFromParents_ (false),
    cleaveFromChildren_(false),
    exprGpuRunnable_(false)
{
  fmlID_ = DEFAULT_FML_ID;
# ifdef ENABLE_CUDA
  deviceID_ = 0;
  cudaStreamAlreadyExists_ = false;
  cudaStream_ = NULL;
# endif
}

//--------------------------------------------------------------------

#ifdef ENABLE_CUDA

void ExpressionBase::create_cuda_stream( int deviceIndex )
{
# ifndef NDEBUG
  if( !IS_GPU_INDEX(deviceIndex) ){
    std::ostringstream msg;
    msg << "Invalid device Index : " << deviceIndex << " found while creating stream for expression : " << this->get_tags()[0]
        << "at " << __FILE__ << " : " << __LINE__ << std::endl;
    throw(std::runtime_error(msg.str()));
  }
# endif

  // Stream created for the first time
  if( !cudaStreamAlreadyExists_ ){
    deviceID_ = deviceIndex;
    cudaSetDevice(deviceID_);
    cudaError err;
    if (cudaSuccess != (err = cudaStreamCreate( &cudaStream_ ))) {
      std::ostringstream msg;
      msg << "Failed to create stream for expression : " << this->get_tags()[0] << std::endl
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
    }
    cudaStreamAlreadyExists_ = true;
  }

  // Stream already exists but there is a change in device context
  else if( cudaStreamAlreadyExists_ && (deviceID_ != deviceIndex) ){
    cudaSetDevice(deviceID_);
    cudaError err;
    if (cudaSuccess != (err = cudaStreamDestroy( cudaStream_ ))) {
      std::ostringstream msg;
      msg << "Failed to destroy existing stream for expression : " << this->get_tags()[0] << std::endl
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
    }
    cudaStream_ = NULL;

    deviceID_ = deviceIndex;
    cudaSetDevice(deviceID_);
    if (cudaSuccess != (err = cudaStreamCreate( &cudaStream_ ))){
      std::ostringstream msg;
      msg << "Failed to create stream for expression : " << this->get_tags()[0] << std::endl
          << " at " << __FILE__ << " : " << __LINE__ << std::endl;
      msg << "\t - " << cudaGetErrorString(err);
      throw(std::runtime_error(msg.str()));
    }
    cudaStreamAlreadyExists_ = true;
  }

  // For other cases
  else{
    return;
  }

# ifdef DEBUG_CUDA_HANDLES
  std::cout << "Creating CudaStream with context : " << deviceID_
            << ", for expr : " << this->get_tags()[0] << " & streamID : " << cudaStream_ << std::endl;
# endif
}

//--------------------------------------------------------------------

cudaStream_t ExpressionBase::get_cuda_stream()
{
  return cudaStream_;
}

#endif

//--------------------------------------------------------------------

ExpressionBase::~ExpressionBase()
{
# ifdef ENABLE_CUDA

# ifdef DEBUG_CUDA_HANDLES
  std::cout << "Destroying Stream with context : " << deviceID_
            << ", for expr " << this->get_tags()[0] << " & streamID : " << cudaStream_ << std::endl;
# endif
    // set the device context before invoking CUDA API
    cudaSetDevice(deviceID_);
    cudaError err;
    if( cudaStream_ != NULL ){
      if( cudaSuccess != (err = cudaStreamDestroy( cudaStream_ )) ){
        std::ostringstream msg;
        msg << "Failed to destroy stream, at " << __FILE__ << " : " << __LINE__
            << std::endl;
        msg << "\t - " << cudaGetErrorString(err);
        throw(std::runtime_error(msg.str()));
      }
      cudaStream_ = NULL;
    }
# endif
}

//--------------------------------------------------------------------

void ExpressionBase::set_computed_tag( const TagList& tags )
{
  exprNames_ = tags;
}

//--------------------------------------------------------------------

void ExpressionBase::set_gpu_runnable( const bool b )
{
  exprGpuRunnable_ = b;
}

//--------------------------------------------------------------------

void
ExpressionBase::
base_advertise_dependents( ExprDeps& exprDeps,
                           FieldDeps& fieldDeps )
{
  // Ensure that this expression has been associated with a field.
  // This should be automatically done when the expression is
  // constructed by the factory.
  if( exprNames_.empty() ){
    std::ostringstream msg;
    msg << "ERROR!  An expression exists that has not been associated with any fields!" << std::endl
        << "       " << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( msg.str() );
  }

  for( ExtraSrcTerms::iterator isrc=srcTerms_.begin(); isrc!=srcTerms_.end(); ++isrc ){
    exprDeps.requires_expression( isrc->srcExpr->get_tags() );
  }

  // NOTE: modifiers do not call base_advertise_dependents called because we
  // already have this information. Of course, perhaps we want to attach source
  // expressions on the modifiers as well?  Hmmmm...
  BOOST_FOREACH( ExpressionBase* modifier, modifiers_ ){
    modifier->advertise_dependents( exprDeps );
#   ifndef NDEBUG
    if( modifier->srcTerms_.size() > 0 ){
      std::ostringstream msg;
      msg << __FILE__ << " : " << __LINE__ << std::endl
          << "ERROR: Modifier expressions do not currently support arbitrary source term addition" << std::endl;
      throw std::runtime_error( msg.str() );
    }
#   endif
  }

  set_computed_fields( fieldDeps );
  this->advertise_dependents(exprDeps);
}

//--------------------------------------------------------------------

void
ExpressionBase::base_bind_fields( FieldManagerList& fldMgrList )
{
  this->bind_computed_fields( fldMgrList );

  BOOST_FOREACH( ExpressionBase* modifier, modifiers_ ){
    modifier->base_bind_fields( fldMgrList );
  }

  // on cleaved graphs, it is possible to have the source term expressions live
  // on different graphs.  In this case, the field binding must be redone to
  // ensure that it is current.
  BOOST_FOREACH( const SrcTermInfo& src, srcTerms_ ){
    ExpressionBase* expr = const_cast<ExpressionBase*>( src.srcExpr );
    expr->bind_computed_fields( fldMgrList );
  }

  // only allow const access to the bind_fields() method.  This will
  // help ensure that the fields obtained there are const.
  const FieldManagerList& fml = fldMgrList;
  this->bind_fields( fml );
}

void
ExpressionBase::base_bind_fields( FMLMap& fmls )
{
# ifndef NDEBUG
  if( fmls.size()>1 ) assert( fmlID_ != DEFAULT_FML_ID );
# endif

  FieldManagerList& fldMgrList = *extract_field_manager_list( fmls, fmlID_ );

  BOOST_FOREACH( ExpressionBase* modifier, modifiers_ ){
    modifier->base_bind_fields( fldMgrList );
  }

  this->bind_computed_fields( fldMgrList );

  // only allow const access to the bind_fields() method.  This will
  // help ensure that the fields obtained there are const.
  const FieldManagerList& fml = fldMgrList;
  this->bind_fields( fml );
}

//--------------------------------------------------------------------

void
ExpressionBase::base_bind_operators( const SpatialOps::OperatorDatabase& opDB )
{
  this->bind_operators( opDB );

  BOOST_FOREACH( ExpressionBase* modifier, modifiers_ ){
    modifier->bind_operators( opDB );
  }
}

//--------------------------------------------------------------------

void
ExpressionBase::add_modifier( ExpressionBase* const modifier, const Tag& tag )
{
  modifiers_.push_back(modifier);
  modifierNames_.push_back( tag );
}

//--------------------------------------------------------------------

void
ExpressionBase::attach_source_expression( const ExpressionBase* const src,
                                          const SourceExprOp op,
                                          const int myFieldIx,
                                          const int srcFieldIx )
{
  assert( !is_placeholder() );
  srcTerms_.insert( SrcTermInfo(src,op,myFieldIx,srcFieldIx) );
}

//--------------------------------------------------------------------

const Tag&
ExpressionBase::get_tag() const
{
  if( exprNames_.size() > 1 ){
    std::ostringstream msg;
    msg << "ERROR from " << __FILE__ << " : " << __LINE__ << std::endl
        << "name() called in an expression which evaluates a vector of fields:" << std::endl;
    for( TagList::const_iterator i=exprNames_.begin(); i!=exprNames_.end(); ++i ){
      msg << "  " << *i;
    }
    msg << "Use get_tags() instead" << std::endl;
    throw std::runtime_error( msg.str() );
  }
  return exprNames_[0];
}

//--------------------------------------------------------------------

} // namespace Expr
