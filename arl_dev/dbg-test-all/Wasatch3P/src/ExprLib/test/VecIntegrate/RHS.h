#ifndef RHS_Expr_h
#define RHS_Expr_h

#include <expression/Expression.h>

typedef SpatialOps::SingleValueField FieldT;

/**
 *  \class RHS
 */
class RHS
 : public Expr::Expression<FieldT>
{
  const std::vector<double> freq_;
  const Expr::Tag timeTag_;
  const FieldT* time_;

  RHS( const std::vector<double>& freq,
       const Expr::Tag time );

public:
  class Builder : public Expr::ExpressionBuilder
  {
    const std::vector<double> f_;
    const Expr::Tag t_;
  public:
    Builder( const Expr::TagList& names,
             const std::vector<double>& f,
             const Expr::Tag t )
    : ExpressionBuilder(names),
      f_( f ), t_(t)
    {}
    ~Builder(){}
    Expr::ExpressionBase* build() const{ return new RHS(f_,t_); }
  };

  ~RHS(){};

  void advertise_dependents( Expr::ExprDeps& exprDeps );
  void bind_fields( const Expr::FieldManagerList& fml );
  void evaluate();
};



// ###################################################################
//
//                          Implementation
//
// ###################################################################



RHS::RHS( const std::vector<double>& f,
          const Expr::Tag time )
  : Expr::Expression<FieldT>(),
    freq_( f ),
    timeTag_( time )
{
# ifdef ENABLE_CUDA
  this->set_gpu_runnable( true );
# endif
}

//--------------------------------------------------------------------

void
RHS::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( timeTag_ );
}

//--------------------------------------------------------------------

void
RHS::bind_fields( const Expr::FieldManagerList& fml )
{
  time_ = &fml.field_ref<FieldT>( timeTag_ );
}

//--------------------------------------------------------------------

void
RHS::evaluate()
{
  using namespace SpatialOps;
  typedef std::vector<FieldT*> FieldVec;
  FieldVec& rhsvec = this->get_value_vec();

  assert( rhsvec.size() == freq_.size() );
  std::vector<double>::const_iterator ifreq = freq_.begin();
  for( FieldVec::iterator irhs=rhsvec.begin(); irhs!=rhsvec.end(); ++irhs, ++ifreq ){
    FieldT& rhs = **irhs;
    rhs <<= sin( *ifreq * *time_ );
  }
}

//--------------------------------------------------------------------

#endif // RHS_Expr_h
