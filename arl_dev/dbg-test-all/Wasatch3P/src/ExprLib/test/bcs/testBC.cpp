#include <spatialops/structured/FVStaggered.h>
#include <spatialops/pointfield/PointFieldTypes.h>
#include <spatialops/pointfield/PointOperators.h>

#include <expression/Functions.h>
#include <expression/ExprLib.h>

#include <test/TestHelper.h>

#include <iostream>

namespace so = SpatialOps;

typedef SpatialOps::SVolField FieldT;
typedef SpatialOps::Point::PointField PointFieldT;


//====================================================================

class FieldBCHelper
{
  const PointFieldT& pf_;
  const SpatialOps::Point::PointToField<FieldT>& p2f_;
public:
  FieldBCHelper( const PointFieldT& pf,
                 const SpatialOps::Point::PointToField<FieldT>& p2f )
    : pf_( pf ), p2f_( p2f )
  {}
  void operator()( FieldT& f )
  {
    p2f_.apply_to_field( pf_, f );
  }
  static inline bool is_gpu_runnable(){ return false; }
};

//====================================================================

int main()
{
  TestHelper status( true );

  Expr::ExprPatch patch(10);

  Expr::ExpressionFactory factory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fm = fml.field_manager<FieldT>();

  //____________________________________________________________
  // select the points that we will be modifying and create an associated field
  const so::BoundaryCellInfo bcInfo = so::BoundaryCellInfo::build<PointFieldT>(true,true,true);
  const so::GhostData pointGhost(0);

  PointFieldT pf( so::IntVec(2,1,1), bcInfo, pointGhost, NULL );

  pf[0] = 1.234;
  pf[1] = 3.456;

  std::vector<size_t> ix;
  ix.push_back(4);
  ix.push_back(2);
  SpatialOps::Point::PointToField<FieldT> p2f( ix );

  FieldBCHelper bc( pf, p2f );

  //____________________________________________________________
  // build the main expression tree and execute it
  Expr::Tag tag("x",Expr::STATE_NONE);
  const Expr::ExpressionID xid = factory.register_expression( new Expr::ConstantExpr<FieldT>::Builder( tag, 1.0 ) );
  Expr::ExpressionTree tree( xid, factory, patch.id() );
  {
    std::ofstream fout("maintree.dot");
    tree.write_tree(fout);
  }

  //NOTE : Placeholder expr is GPU runnable by default but GPU is currently
  // not suppoted for the point fields. So GPU runnable flag is set to false.
# ifdef ENABLE_CUDA
  Expr::ExpressionBase& expr = factory.retrieve_expression( tag, patch.id() );
  expr.set_gpu_runnable( false );
# endif

  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  Expr::Expression<FieldT>& xexpr = dynamic_cast<Expr::Expression<FieldT>& >( factory.retrieve_expression(tag,patch.id(),true) );
  xexpr.process_after_evaluate( bc, bc.is_gpu_runnable() );

  tree.execute_tree();

  //_________________________________________________
  // did the BC get set?
  {
    const FieldT& f = fm.field_ref(tag);
    status( f[4] == 1.234, "point 1" );
    status( f[2] == 3.456, "point 2" );
  }

  if( status.ok() ) return 0;
  return -1;
}
