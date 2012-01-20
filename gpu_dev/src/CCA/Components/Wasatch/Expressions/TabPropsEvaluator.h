#ifndef TabPropsEvaluator_Expr_h
#define TabPropsEvaluator_Expr_h

#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>

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
  typedef std::vector<      FieldT*>  FieldVec;
  typedef std::vector<const FieldT*>  IndepVarVec;
  typedef std::vector<const InterpT*> Evaluators;
  typedef std::vector<Expr::Tag>      VarNames;

  const VarNames indepVarNames_;

  IndepVarVec indepVars_;
  Evaluators  evaluators_;

  TabPropsEvaluator( const InterpT* const interp,
                     const VarNames& ivarNames );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const InterpT* const interp_;
    const VarNames ivarNames_;
  public:
    Builder( const Expr::Tag& result,
             const InterpT* interp,
             const VarNames& ivarNames );
    ~Builder(){}
    Expr::ExpressionBase* build() const;
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
TabPropsEvaluator( const InterpT* const interp,
                   const VarNames& ivarNames )
  : Expr::Expression<FieldT>(),
    indepVarNames_( ivarNames   )
{
  evaluators_.push_back( interp );
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
Builder::Builder( const Expr::Tag& result,
                  const InterpT* const interp,
                  const VarNames& ivarNames )
  : ExpressionBuilder(result),
    interp_   ( interp    ),
    ivarNames_( ivarNames )
{}

//--------------------------------------------------------------------

template< typename FieldT >
Expr::ExpressionBase*
TabPropsEvaluator<FieldT>::
Builder::build() const
{
  return new TabPropsEvaluator<FieldT>( interp_, ivarNames_ );
}

#endif // TabPropsEvaluator_Expr_h
