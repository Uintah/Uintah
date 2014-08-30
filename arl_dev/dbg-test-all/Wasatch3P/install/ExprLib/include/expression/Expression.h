/**
 * \file Expression.h
 * \author James C. Sutherland
 *
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
#ifndef Expression_h
#define Expression_h

#include <map>

#include <boost/signals2.hpp>
#include <boost/foreach.hpp>
#include <boost/lexical_cast.hpp>

#include <expression/ExpressionBase.h>
#include <expression/FieldManager.h>

#include <spatialops/Nebo.h>

namespace Expr{

//====================================================================


/**
 *  @class  Expression
 *  @author James C. Sutherland
 *  @date   May, 2007
 *
 *  The Expression class (and the parent class ExpressionBase) is the
 *  fundamental computational kernel.  It provides a hierarchical way
 *  to generate complex expressions that allow efficient evaluation of
 *  functions as well as sensitivities.
 *
 *
 *  @par Evaluation and values of Expressions
 *
 *  Evaluation of an expression is separate from value-retrieval,
 *  since we may retrieve the value of the expression many times until
 *  we must recompute it.
 *
 *
 *  @par Expression dependencies
 *
 *  An Expression should advertise its dependency on other expressions
 *  through the <code>advertise_dependents()</code> method.
 *
 *
 *  @par Construction of Expression Objects
 *
 *  Expression objects should be constructed using an
 *  ExpressionBuilder object.  This provides a standard interface to
 *  the ExpressionFactory.  The construction process is as follows:
 *
 *   \li Register the expression's ExpressionBuilder class with the
 *       ExpressionFactory.  The factory is then responsible for
 *       building the expression and managing deletion.
 *
 *   \li When an ExpressionTree is created, the root expression is
 *       interrogated for its dependencies by calling its
 *       <code>advertise_dependents()</code> method.  The resulting
 *       dependent expressions are recursively descended to construct a
 *       dependency graph.  This graph is then compiled and each
 *       expression in the graph is constructed through the
 *       ExpressionFactory.
 *
 *
 *  @par External modification of values computed by an Expression
 *
 *  Occasionally we need to modify values computed by an Expression.
 *  An example is imposing boundary conditions, where we must modify
 *  some of the field values.  The process_after_evaluate() method
 *  facilitates this process.  T he user provides a call-back functor
 *  which is evaluated after the expression is computed.  Note that
 *  multiple such functors may be added, but there is no guarantee
 *  for ordering of these functors when the callback occurs.
 *
 *
 *  @tparam ValT The type of value that this expression computes and
 *          returns.  This may be an integral value or a class.
 *
 *  @todo Need to rework the signal approach so that connections can be
 *        destroyed.  This is particularly important for mesh changes,
 *        where BC objects will be re-created.  The old ones must be
 *        destroyed and the signal must be disconnected.
 */
template< typename ValT >
class Expression : public ExpressionBase
{
  typedef typename boost::signals2::signal< void(ValT&) > Signal;

public:

  typedef std::vector<ValT*> ValVec;
  virtual ~Expression();

  /**
   * Trigger evaluation of an expression. It should only be called by the
   * scheduler - not by user code.
   */
  inline void base_evaluate();

  /**
   *  This can be called to load additional work on this expression.
   *  It is primarily intended for applying boundary conditions to a
   *  field after it has been evaluated.
   *
   *  Any subscribers loaded on here will be executed after the
   *  expression is evaluated.  The argument to the functor is the
   *  field that the expression evaluates.
   *
   *  \param subscriber the functor to evaluate
   *  \param isGPUReady true if the functor can handle GPU call-backs.
   */
  void process_after_evaluate( typename Signal::slot_function_type subscriber,
                               const bool isGPUReady );

  /**
   *  This can be called to load additional work on this expression.
   *  It is primarily intended for applying boundary conditions to a
   *  field after it has been evaluated.
   *
   *  Any subscribers loaded on here will be executed after the
   *  expression is evaluated.  The argument to the functor is the
   *  field that the expression evaluates.
   *
   *  \param fieldName In the case of an expression that evaluates
   *         multiple fields, this supplies the name of the field to
   *         process.
   *  \param subscriber the functor to evaluate
   *  \param isGPUReady true if the functor can handle GPU call-backs.
   */
  void process_after_evaluate( const std::string fieldName,
                               typename Signal::slot_function_type subscriber,
                               const bool isGPUReady );

  /**
   * Obtain the number of modifiers and "process_after_evaluate()" functors
   * associated with this expression.
   */
  int num_post_processors() const{ return postEvalSignal_.size() + modifiers_.size(); }

  /**
   * Obtain the names of the modifiers and "process_after_evaluate()" functors
   * associated with this expression.
   */
  std::vector<std::string> post_proc_names() const;

  const char* field_typeid() const{ return typeid(ValT).name(); }

  ExpressionBase* as_placeholder_expression() const;

protected:

  virtual void set_computed_fields( FieldDeps& );
  virtual void bind_computed_fields( FieldManagerList& );

  Expression();

  inline ValT& value();
  inline ValVec& get_value_vec(){ return exprValues_; }
  inline const ValVec& get_value_vec() const{ return exprValues_; }

private:

  typedef std::map<size_t,Signal*> SignalMap;
  SignalMap postEvalSignal_;

  Expression( const Expression& ); // no copying
  Expression& operator=( const Expression& ); // no assignment

  std::vector<ValT*> exprValues_; ///< The value(s) that this expression computes.

};



/**
 *  @class  PlaceHolder
 *  @author James C. Sutherland
 *  @date   June, 2008
 *
 *  @brief This expression does nothing.  It simply wraps a field.
 */
template< typename FieldT >
class PlaceHolder : public Expression<FieldT>
{
public:
  inline void evaluate(){}

  void advertise_dependents( ExprDeps& exprDeps ){}

  inline void bind_fields( const FieldManagerList& fml ){}

  bool is_placeholder() const{ return true; }

  /**
   *  @class Builder
   *  @brief Constructs PlaceHolder objects.
   */
  struct Builder : public ExpressionBuilder
  {
    Builder( const Tag& name      ) : ExpressionBuilder(name ) {}
    Builder( const TagList& names ) : ExpressionBuilder(names) {}
    ~Builder(){}
    ExpressionBase* build() const { return new PlaceHolder<FieldT>(); }
  };

protected:

  PlaceHolder()
    : Expression<FieldT>()
  {
    this->set_gpu_runnable(true);
  }

  virtual ~PlaceHolder(){}

};

//====================================================================




//####################################################################
//
//
//                          IMPLEMENTATION
//
//
//####################################################################




//====================================================================


//--------------------------------------------------------------------
template<typename ValT>
Expression<ValT>::Expression()
  : ExpressionBase()
{}
//--------------------------------------------------------------------
template<typename ValT>
Expression<ValT>::~Expression()
{
  for( typename SignalMap::iterator isig=postEvalSignal_.begin(); isig!=postEvalSignal_.end(); ++isig ){
    delete isig->second;
  }
}
//--------------------------------------------------------------------
template<typename ValT>
ValT&
Expression<ValT>::value()
{
# ifndef NDEBUG
  if( exprValues_.size() != 1 ){
    std::ostringstream msg;
    msg << std::endl << __FILE__ << " : " << __LINE__
        << "\n\nERROR! Expression::value() called on \n\t"
        << this->get_tags()
        << "\nbut this expression does not compute just one value.\n"
        << "Apparently, it computes " << exprValues_.size() << " values\n"
        << "Use Expression:get_value_vec() instead\n";
    throw std::runtime_error( msg.str() );
  }
# endif
  return *exprValues_[0];
}
//--------------------------------------------------------------------
template<typename ValT>
void
Expression<ValT>::set_computed_fields( FieldDeps& fieldDeps )
{
  const TagList& exprNames = this->get_tags();
  for( TagList::const_iterator itag=exprNames.begin(); itag!=exprNames.end(); ++itag ){
    fieldDeps.requires_field<ValT>( *itag );
  }
}
//--------------------------------------------------------------------
template<typename ValT>
inline void
Expression<ValT>::bind_computed_fields( FieldManagerList& fml )
{
  typename FieldMgrSelector<ValT>::type& fm = fml.field_manager<ValT>();
  const TagList& exprNames = this->get_tags();
  exprValues_.clear();
  BOOST_FOREACH( const Tag& tag, exprNames ){
    ValT& field = fm.field_ref( tag );
#   ifdef ENABLE_CUDA
    field.set_stream( cudaStream_ );
#   endif
    this->exprValues_.push_back( &field );

    if( tag.context() == CARRY_FORWARD && !is_placeholder() ){
      fm.copy_field_forward( tag, field );
    }
  }
}
//--------------------------------------------------------------------
template<typename ValT>
inline void
Expression<ValT>::base_evaluate()
{
  using namespace SpatialOps;
  this->evaluate();

  // add extra terms to this expression as needed.
  if( !is_placeholder() ){  // placeholder expressions don't have source terms
    for( ExtraSrcTerms::const_iterator isrc=srcTerms_.begin(); isrc!=srcTerms_.end(); ++isrc ){
      const Expression<ValT>* const src = dynamic_cast<const Expression<ValT>*>( isrc->srcExpr );
      const ValT& rhsVal = *(src->get_value_vec()[isrc->srcIx]);
      ValT& exprVal = *(this->exprValues_[isrc->myIx]);
      switch( isrc->op ){
        case ADD_SOURCE_EXPRESSION     :  exprVal <<= exprVal+rhsVal;  break;
        case SUBTRACT_SOURCE_EXPRESSION:  exprVal <<= exprVal-rhsVal;  break;
      }
    }
  }

  BOOST_FOREACH( ExpressionBase* modifier, this->modifiers_ ){
    // NOTE: modifiers should not call base_evaluate because modifiers shouldn't
    //       be messing with this stuff.  But the evaluate() method is not
    //       publicly accessible here...
    modifier->base_evaluate();
  }

  for( typename SignalMap::iterator isig=postEvalSignal_.begin(); isig!=postEvalSignal_.end(); ++isig ){
    Signal& sig = *(isig->second);
    sig( *this->exprValues_[isig->first] );
  }

}
//--------------------------------------------------------------------
template<typename ValT>
void
Expression<ValT>::process_after_evaluate( typename Signal::slot_function_type subscriber,
                                          const bool isGPUReady )
{
  if( exprValues_.size() > 1 ){
    std::ostringstream msg;
    msg << "ERROR from " << __FILE__ << " : " << __LINE__ << std::endl
        << "      Cannot add post processing for expressions with multiple fields to be evaluated!" << std::endl
        << "      Expression name(s): ";
    const TagList& names = this->get_tags();
    for( TagList::const_iterator inm=names.begin(); inm!=names.end(); ++inm ){
      msg << "'" << inm->field_name() << "', ";
    }
    msg << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }
  process_after_evaluate( this->get_tag().field_name(), subscriber, isGPUReady );
}
//--------------------------------------------------------------------
template<typename ValT>
void
Expression<ValT>::process_after_evaluate( const std::string fieldName,
                                          typename Signal::slot_function_type subscriber,
                                          const bool isGPUReady )
{
  const TagList& names = this->get_tags();
  size_t ii=0;
  for( typename TagList::const_iterator inm=names.begin(); inm!=names.end(); ++inm, ++ii ){
    if( inm->field_name() == fieldName ) break;
  }
  if( ii == names.size() ){
    std::ostringstream msg;
    msg << "ERROR from " << __FILE__ << " : " << __LINE__ << std::endl
        << "      Expression evaluating " << std::endl
        << names << std::endl
        << "      does not have a field named '" << fieldName << "'" << std::endl;
  }
  typename SignalMap::iterator ism = postEvalSignal_.find( ii );
  if( ism == postEvalSignal_.end() ){
    Signal* sig = new Signal;
    ism = postEvalSignal_.insert( std::make_pair(ii,sig) ).first;
  }
  ism->second->connect(subscriber);

  if( this->is_gpu_runnable() ){
    if( !isGPUReady ) this->set_gpu_runnable( false );
  }
}
//--------------------------------------------------------------------

template<typename ValT>
std::vector<std::string>
Expression<ValT>::post_proc_names() const
{
  std::vector<std::string> names;
  BOOST_FOREACH( const Tag mod, modifierNames_ ){
    names.push_back( mod.name() );
  }
  for( size_t i=0; i<postEvalSignal_.size(); ++i ){
    names.push_back( "ProcAfterEval_" + boost::lexical_cast<std::string>(i+1) );
  }
  return names;
}

//--------------------------------------------------------------------

template<typename FieldT>
ExpressionBase*
Expression<FieldT>::as_placeholder_expression() const
{
  typedef typename Expr::PlaceHolder<FieldT>::Builder Builder;
  const Builder builder( this->get_tags() );
  ExpressionBase* expr = builder.build();
  expr->set_computed_tag( this->get_tags() );
  return expr;
}

} // namespace Expr


#endif // Expression_h
