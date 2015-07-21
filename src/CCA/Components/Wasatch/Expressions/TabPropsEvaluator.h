/*
 * The MIT License
 *
 * Copyright (c) 2012-2015 The University of Utah
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

#ifndef TabPropsEvaluator_Expr_h
#define TabPropsEvaluator_Expr_h

#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>

#include <spatialops/Nebo.h>
#include <expression/Expression.h>

/**
 *  \ingroup	Expressions
 *  \class  	TabPropsEvaluator
 *  \author 	James C. Sutherland
 *  \date   	June, 2010
 *
 *  \brief Evaluates one or more fields using TabProps, an external
 *         library for tabular property evaluation.
 *
 *  \tparam FieldT the type of field for both the independent and
 *          dependent variables.
 */
template< typename FieldT >
class TabPropsEvaluator
 : public Expr::Expression<FieldT>
{
  DECLARE_VECTOR_OF_FIELDS(FieldT, indepVars_)
  const InterpT& evaluator_;

  TabPropsEvaluator( const InterpT& interp,
                     const Expr::TagList& ivarNames );

  class Functor1D{
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval ){
      get_evaluator() = eval;
    }
    double operator()( const double x ) const{
      return get_evaluator()->value(&x);
    }
  };

  class Functor2D{
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval ){ get_evaluator() = eval; }
    double operator()( const double x1, const double x2 ) const{
      double vals[2] = {x1,x2};
      return get_evaluator()->value( vals );
    }
  };

  class Functor3D{
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval ){ get_evaluator() = eval; }
    double operator()( const double x1, const double x2, const double x3 ) const{
      double vals[3] = {x1,x2,x3};
      return get_evaluator()->value( vals );
    }
  };

  class Functor4D{
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval ){ get_evaluator() = eval; }
    double operator()( const double x1, const double x2, const double x3, const double x4 ) const{
      double vals[4] = {x1,x2,x3,x4};
      return get_evaluator()->value( vals );
    }
  };

  class Functor5D{
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval ){ get_evaluator() = eval; }
    double operator()( const double x1, const double x2, const double x3, const double x4, const double x5 ) const{
      double vals[5] = {x1,x2,x3,x4,x5};
      return get_evaluator()->value( vals );
    }
  };

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const InterpT* const interp_;
    const Expr::TagList ivarNames_;
  public:
    Builder( const Expr::Tag& result,
             const InterpT& interp,
             const Expr::TagList& ivarNames );
    ~Builder(){ delete interp_; }
    Expr::ExpressionBase* build() const;
  };

  ~TabPropsEvaluator();

  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
TabPropsEvaluator<FieldT>::
TabPropsEvaluator( const InterpT& interp,
                   const Expr::TagList& ivarNames )
  : Expr::Expression<FieldT>(),
    evaluator_( interp )
{
  this->template create_field_vector_request<FieldT>(ivarNames, indepVars_);
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsEvaluator<FieldT>::
~TabPropsEvaluator()
{}

//--------------------------------------------------------------------

template< typename FieldT >
void
TabPropsEvaluator<FieldT>::
evaluate()
{
  FieldT& result = this->value();

  switch( indepVars_.size() ){
    case 1:{
      Functor1D::set_evaluator( &evaluator_ );
      result <<= SpatialOps::apply_pointwise<Functor1D>( indepVars_[0]->field_ref() );
      break;
    }
    case 2:{
      Functor2D::set_evaluator( &evaluator_ );
      result <<= SpatialOps::apply_pointwise<Functor2D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref() );
      break;
    }
    case 3:{
      Functor3D::set_evaluator( &evaluator_ );
      result <<= SpatialOps::apply_pointwise<Functor3D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref() );
      break;
    }
    case 4:{
      Functor4D::set_evaluator( &evaluator_ );
      result <<= SpatialOps::apply_pointwise<Functor4D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref() );
      break;
    }
    case 5:{
      Functor5D::set_evaluator( &evaluator_ );
      result <<= SpatialOps::apply_pointwise<Functor5D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref(), indepVars_[4]->field_ref() );
      break;
    }
    default:
      throw std::invalid_argument( "Unsupported dimensionality for interpolant in TabPropsEvaluator" );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsEvaluator<FieldT>::
Builder::Builder( const Expr::Tag& result,
                  const InterpT& interp,
                  const Expr::TagList& ivarNames )
  : ExpressionBuilder(result),
    interp_   ( interp.clone() ),
    ivarNames_( ivarNames )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TabPropsEvaluator<FieldT>::
Builder::build() const
{
  return new TabPropsEvaluator<FieldT>( *interp_, ivarNames_ );
}

#endif // TabPropsEvaluator_Expr_h
