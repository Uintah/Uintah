#ifndef Dummy_Expr_h
#define Dummy_Expr_h

#include <expression/Expression.h>

/**
 *  \class Dummy
 */
class Dummy
 : public Expr::Expression<SpatialOps::SingleValueField>
{
  const Expr::TagList childTags_;
  std::vector<const SpatialOps::SingleValueField*> children_;
  Dummy( const Expr::TagList& );
public:
  class Builder : public Expr::ExpressionBuilder
  {
  public:
    Builder( const Expr::Tag& resultTag,
             const Expr::TagList& childTags );
    Builder( const Expr::Tag& resultTag,
             const Expr::Tag& childTag );

    Expr::ExpressionBase* build() const;

  private:
    Expr::TagList childTags_;
  };
  ~Dummy();
  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



Dummy::Dummy( const Expr::TagList& tags )
  : Expr::Expression<SpatialOps::SingleValueField>(),
    childTags_( tags )
{}

//--------------------------------------------------------------------

Dummy::~Dummy(){}

//--------------------------------------------------------------------

void
Dummy::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  for( Expr::TagList::const_iterator i=childTags_.begin(); i!=childTags_.end(); ++i ){
    exprDeps.requires_expression( *i );
  }
}

//--------------------------------------------------------------------

void
Dummy::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<SpatialOps::SingleValueField>::type& fm = fml.field_manager<SpatialOps::SingleValueField>();

  children_.clear();
  for( Expr::TagList::const_iterator i=childTags_.begin(); i!=childTags_.end(); ++i ){
    children_.push_back( &fm.field_ref(*i) );
  }
}

//--------------------------------------------------------------------

void
Dummy::evaluate()
{
  using namespace SpatialOps;
  SpatialOps::SingleValueField& result = this->value();
  result <<= 0.0;
  for( size_t i=0; i<children_.size(); ++i ){
    result <<= result + *children_[i];
  }
}

//--------------------------------------------------------------------

Dummy::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::TagList& tags )
  : ExpressionBuilder( resultTag ),
    childTags_( tags )
{}

Dummy::
Builder::Builder( const Expr::Tag& resultTag,
                  const Expr::Tag& tag )
  : ExpressionBuilder( resultTag )
{
  childTags_.push_back(tag);
}

//--------------------------------------------------------------------

Expr::ExpressionBase*
Dummy::
Builder::build() const
{
  return new Dummy( childTags_ );
}


#endif // Dummy_Expr_h
