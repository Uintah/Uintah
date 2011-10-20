#ifndef TabPropsEvaluator_Expr_h
#define TabPropsEvaluator_Expr_h

#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>
#include <tabprops/BSpline.h>

#include <expression/Expr_Expression.h>

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
  typedef std::vector<      FieldT*>  FieldVec;
  typedef std::vector<const FieldT*>  IndepVarVec;
  typedef std::vector<const BSpline*> Evaluators;
  typedef std::vector<Expr::Tag>      VarNames;

  const VarNames indepVarNames_;

  IndepVarVec indepVars_;
  Evaluators  evaluators_;

  TabPropsEvaluator( const BSpline* const spline,
                     const VarNames& ivarNames,
                     const Expr::ExpressionID& id,
                     const Expr::ExpressionRegistry& reg );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const BSpline* const spline_;
    const VarNames ivarNames_;
  public:
    Builder( const BSpline* spline,
             const VarNames& ivarNames );

    Expr::ExpressionBase*
    build( const Expr::ExpressionID& id,
           const Expr::ExpressionRegistry& reg ) const;
  };

  ~TabPropsEvaluator();

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



template< typename FieldT >
TabPropsEvaluator<FieldT>::
TabPropsEvaluator( const BSpline* const spline,
                   const VarNames& ivarNames,
                   const Expr::ExpressionID& id,
                   const Expr::ExpressionRegistry& reg )
  : Expr::Expression<FieldT>(id,reg),
    indepVarNames_( ivarNames   )
{
  evaluators_.push_back( spline );
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsEvaluator<FieldT>::
~TabPropsEvaluator()
{
  for( Evaluators::iterator ieval=evaluators_.begin(); ieval!=evaluators_.end(); ++ieval ){
    //jcs this was causing a segfault:
    //    delete *ieval;
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TabPropsEvaluator<FieldT>::
advertise_dependents( Expr::ExprDeps& exprDeps )
{
  // we depend on expressions for each of the independent variables.
  for( VarNames::const_iterator inam=indepVarNames_.begin(); inam!=indepVarNames_.end(); ++inam ){
    exprDeps.requires_expression( *inam );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TabPropsEvaluator<FieldT>::
bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldManager<FieldT>& fm = fml.template field_manager<FieldT>();

  indepVars_.clear();
  for( VarNames::const_iterator inam=indepVarNames_.begin(); inam!=indepVarNames_.end(); ++inam ){
    indepVars_.push_back( &fm.field_ref( *inam ) );
  }
}

//--------------------------------------------------------------------

template< typename FieldT >
void
TabPropsEvaluator<FieldT>::
evaluate()
{
  FieldVec& results = this->get_value_vec();

  typedef std::vector< typename FieldT::      iterator > DVarIter;
  typedef std::vector< typename FieldT::const_iterator > IVarIter;

  DVarIter dvarIters;
  IVarIter ivarIters;

  for( typename FieldVec::iterator i=results.begin(); i!=results.end(); ++i ){
    dvarIters.push_back( (*i)->begin() );
  }
  for( typename IndepVarVec::const_iterator i=indepVars_.begin(); i!=indepVars_.end(); ++i ){
    ivarIters.push_back( (*i)->begin() );
  }

  std::vector<double> ivarsPoint;

  // loop over grid points.  iii is a dummy variable.
  for( typename FieldT::const_iterator iii=results[0]->begin(); iii!=results[0]->end(); ++iii ){
  
    // loop over fields to be evaluated
    typename DVarIter::iterator iresult = dvarIters.begin();
    for( typename Evaluators::const_iterator ieval=evaluators_.begin(); ieval!=evaluators_.end(); ++ieval, ++iresult ){

      // extract indep vars at this grid point
      ivarsPoint.clear();
      for( typename IVarIter::const_iterator i=ivarIters.begin(); i!=ivarIters.end(); ++i ){
        ivarsPoint.push_back( **i );
      }

      // calculate the result
      **iresult = (*ieval)->value(ivarsPoint );

    }

    // increment all iterators to the next grid point
    for( typename IVarIter::iterator i=ivarIters.begin(); i!=ivarIters.end(); ++i )  ++(*i);
    for( typename DVarIter::iterator i=dvarIters.begin(); i!=dvarIters.end(); ++i )  ++(*i);

  } // grid loop
}

//--------------------------------------------------------------------

template< typename FieldT >
TabPropsEvaluator<FieldT>::
Builder::Builder( const BSpline* const spline,
                  const VarNames& ivarNames )
  : spline_   ( spline    ),
    ivarNames_( ivarNames )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TabPropsEvaluator<FieldT>::
Builder::build( const Expr::ExpressionID& id,
                const Expr::ExpressionRegistry& reg ) const
{
  return new TabPropsEvaluator<FieldT>( spline_, ivarNames_, id, reg );
}

#endif // TabPropsEvaluator_Expr_h
