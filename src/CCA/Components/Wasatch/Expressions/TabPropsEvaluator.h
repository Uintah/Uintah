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
  DECLARE_VECTOR_OF_FIELDS( FieldT, indepVars_ )
  const InterpT& evaluator_;
  const bool doDerivative_;
  const size_t derIndex_;

  TabPropsEvaluator( const InterpT& interp,
                     const Expr::TagList& ivarNames,
                     const Expr::Tag derVarName=Expr::Tag() );

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

  class DerFunctor1D{
    static int& get_dim(){ static int dim; return dim; }
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval, const int dim ){
      get_evaluator() = eval;
      get_dim() = dim;
    }
    double operator()( const double x ) const{
      return get_evaluator()->derivative(&x,get_dim());
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
  class DerFunctor2D{
    static int& get_dim(){ static int dim; return dim; }
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval, const int dim ){
      get_evaluator() = eval;
      get_dim() = dim;
    }
    double operator()( const double x1, const double x2 ) const{
      double vals[2] = {x1,x2};
      return get_evaluator()->derivative( vals, get_dim() );
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
  class DerFunctor3D{
    static int& get_dim(){ static int dim; return dim; }
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval, const int dim ){
      get_evaluator() = eval;
      get_dim() = dim;
    }
    double operator()( const double x1, const double x2, const double x3 ) const{
      double vals[3] = {x1,x2,x3};
      return get_evaluator()->derivative( vals, get_dim() );
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
  class DerFunctor4D{
    static int& get_dim(){ static int dim; return dim; }
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval, const int dim ){
      get_evaluator() = eval;
      get_dim() = dim;
    }
    double operator()( const double x1, const double x2, const double x3, const double x4 ) const{
      double vals[4] = {x1,x2,x3,x4};
      return get_evaluator()->derivative( vals, get_dim() );
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
  class DerFunctor5D{
    static int& get_dim(){ static int dim; return dim; }
    static const InterpT*& get_evaluator(){
      static const InterpT* eval = NULL;
      return eval;
    }
  public:
    static void set_evaluator( const InterpT* eval, const int dim ){
      get_evaluator() = eval;
      get_dim() = dim;
    }
    double operator()( const double x1, const double x2, const double x3, const double x4, const double x5 ) const{
      double vals[5] = {x1,x2,x3,x4,x5};
      return get_evaluator()->derivative( vals, get_dim() );
    }
  };

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const InterpT* const interp_;
    const Expr::TagList ivarNames_;
    const Expr::Tag derVarTag_;
  public:
    /**
     * Calculate property \f$\phi=\mathcal{G}(\vec{\eta})\f$
     *
     * @param result the value of the property
     * @param interp the TabProps interpolator to use
     * @param ivarNames the tags for the independent variables (ordered appropriately)
     */
    Builder( const Expr::Tag& result,
             const InterpT& interp,
             const Expr::TagList& ivarNames )
      : ExpressionBuilder(result),
        interp_   ( interp.clone() ),
        ivarNames_( ivarNames ),
        derVarTag_()
    {}

    /**
     * For property \f$\phi=\mathcal{G}(\vec{\eta})\f$, compute \f$\frac{\partial \phi}{\partial \eta_j} \f$
     *
     * @brief Build a tabular property evaluator that computes the partial derivative with respect to the given independent variable.
     * @param result the partial derivative value
     * @param interp the TabProps interpolator to use
     * @param ivarNames the tags for the independent variables (ordered appropriately)
     * @param derVar the independent variable \f$\eta_j\f$ to differentiate with respect to
     */
    Builder( const Expr::Tag& result,
             const InterpT& interp,
             const Expr::TagList& ivarNames,
             const Expr::Tag derVar )
      : ExpressionBuilder(result),
        interp_   ( interp.clone() ),
        ivarNames_( ivarNames ),
        derVarTag_( derVar )
    {}

    ~Builder(){ delete interp_; }
    Expr::ExpressionBase* build() const{
      return new TabPropsEvaluator<FieldT>( *interp_, ivarNames_, derVarTag_ );
    }
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
                   const Expr::TagList& ivarNames,
                   const Expr::Tag derVarName )
  : Expr::Expression<FieldT>(),
    evaluator_( interp ),
    doDerivative_( derVarName != Expr::Tag() ),
    derIndex_( doDerivative_
               ? std::find( ivarNames.begin(), ivarNames.end(), derVarName ) - ivarNames.begin()
                   : ivarNames.size() )
{
  if( doDerivative_ && derIndex_ == ivarNames.size() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__
        << "Invalid usage of TabPropsEvaluator.\n"
        << "If you want to obtain the partial derivative, you must supply a valid Tag\n";
    throw std::runtime_error( msg.str() );
  }
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
      if( doDerivative_ ){
        DerFunctor1D::set_evaluator( &evaluator_, derIndex_ );
        result <<= SpatialOps::apply_pointwise<DerFunctor1D>( indepVars_[0]->field_ref() );
      }
      else{
        Functor1D::set_evaluator( &evaluator_ );
        result <<= SpatialOps::apply_pointwise<Functor1D>( indepVars_[0]->field_ref() );
      }
      break;
    }
    case 2:{
      if( doDerivative_ ){
        DerFunctor2D::set_evaluator( &evaluator_, derIndex_ );
        result <<= SpatialOps::apply_pointwise<Functor2D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref() );
      }
      else{
        Functor2D::set_evaluator( &evaluator_ );
        result <<= SpatialOps::apply_pointwise<Functor2D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref() );
      }
      break;
    }
    case 3:{
      if( doDerivative_ ){
        DerFunctor3D::set_evaluator( &evaluator_, derIndex_ );
        result <<= SpatialOps::apply_pointwise<Functor3D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref() );
      }
      else{
        Functor3D::set_evaluator( &evaluator_ );
        result <<= SpatialOps::apply_pointwise<Functor3D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref() );
      }
      break;
    }
    case 4:{
      if( doDerivative_ ){
        DerFunctor4D::set_evaluator( &evaluator_, derIndex_ );
        result <<= SpatialOps::apply_pointwise<Functor4D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref() );
      }
      else{
        Functor4D::set_evaluator( &evaluator_ );
        result <<= SpatialOps::apply_pointwise<Functor4D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref() );
      }
      break;
    }
    case 5:{
      if( doDerivative_ ){
        DerFunctor5D::set_evaluator( &evaluator_, derIndex_ );
        result <<= SpatialOps::apply_pointwise<Functor5D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref(), indepVars_[4]->field_ref() ); 
      }
      else{
        Functor5D::set_evaluator( &evaluator_ );
        result <<= SpatialOps::apply_pointwise<Functor5D>( indepVars_[0]->field_ref(), indepVars_[1]->field_ref(), indepVars_[2]->field_ref(), indepVars_[3]->field_ref(), indepVars_[4]->field_ref() );
      }
      break;
    }
    default:
      throw std::invalid_argument( "Unsupported dimensionality for interpolant in TabPropsEvaluator" );
  }
}

//--------------------------------------------------------------------

#endif // TabPropsEvaluator_Expr_h
