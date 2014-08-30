#include <stdexcept>
#include <sstream>

#include "RHS.h"

//--------------------------------------------------------------------

RHS::RHS( const Expr::TagList& ctags,
          const double k1,
          const double k2 )
  : Expr::Expression<FieldT>(),
    ctags_( ctags ),
    k1_( k1 ), k2_( k2 )
{
  if( ctags.size() != 3 ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << "Expected 3 species in TagList, found " << ctags.size() << std::endl;
    throw( msg.str() );
  }
  this->set_gpu_runnable(true);
}

//--------------------------------------------------------------------

void
RHS::advertise_dependents( Expr::ExprDeps& exprDeps )
{
  exprDeps.requires_expression( ctags_ );
}

//--------------------------------------------------------------------

void
RHS::bind_fields( const Expr::FieldManagerList& fml )
{
  const Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();

  // bind all of the species
  c_.clear();
  for( Expr::TagList::const_iterator ict=ctags_.begin(); ict!=ctags_.end(); ++ict ){
    c_.push_back( &fm.field_ref( *ict ) );
  }
}

//--------------------------------------------------------------------

void
RHS::evaluate()
{
  using namespace SpatialOps;

  FieldVec& result = this->get_value_vec();

  const SingleValueField& cA = *(c_[0]);
  const SingleValueField& cB = *(c_[1]);
  SpatFldPtr<SingleValueField> r1 = SpatialFieldStore::get<SingleValueField>(cA);
  SpatFldPtr<SingleValueField> r2 = SpatialFieldStore::get<SingleValueField>(cA);
//  const double& cC = *(c_[2]);

  // calculate the rates.
  *r1 <<= k1_ * cA;
  *r2 <<= k2_ * cB;

  SingleValueField& rhsA = *(result[0]);
  SingleValueField& rhsB = *(result[1]);
  SingleValueField& rhsC = *(result[2]);

  rhsA <<= -*r1;
  rhsB <<= *r1 - *r2;
  rhsC <<= *r2;
}

//--------------------------------------------------------------------
