#include <iostream>

#include <expression/ExprLib.h>

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/OperatorDatabase.h>

#include "HeatFluxExpr.h"
#include "GradTExpr.h"


int main()
{
  typedef SpatialOps::SVolField    CellField;
  typedef SpatialOps::SSurfXField  XSideField;

  typedef SpatialOps::BasicOpTypes<CellField>::GradX  GradXC2F;

  typedef Expr::ExprPatch  Patch;   // currently implements 1-D meshes

  typedef HeatFluxExpr      <GradXC2F  >  HeatFlux_X;
  typedef Expr::ConstantExpr<XSideField>  TCond;
  typedef Expr::PlaceHolder <CellField >  Temp;
  typedef GradTExpr         <GradXC2F  >  GradT_X;

  bool error = false;

  //
  // build a patch
  //
  const int npts = 10;
  Patch patch(npts);

  // set mesh spacing
  std::vector<double> spacing(3,1.0);

  //
  // register required operators
  //
  patch.operator_database().register_new_operator( new GradXC2F( SpatialOps::build_two_point_coef_collection( -1.0/spacing[0], -1.0/spacing[0] ) ) );

  const Expr::Tag thermCondTag( "ThermalConductivity",Expr::STATE_NONE );
  const Expr::Tag tempTag     ( "Temperature",        Expr::STATE_NONE );
  const Expr::Tag gradTTag    ( "dT/dx",              Expr::STATE_NONE );
  const Expr::Tag xHeatFluxTag("HeatFluxX",           Expr::STATE_NONE );
  //
  // register all expressions
  //
  Expr::ExpressionFactory exprFactory;
  const Expr::ExpressionID heatFluxX_id = exprFactory.register_expression( new HeatFlux_X::Builder(xHeatFluxTag,thermCondTag,gradTTag) );
  const Expr::ExpressionID tcondID_id   = exprFactory.register_expression( new TCond     ::Builder(thermCondTag,2.0) );
  const Expr::ExpressionID temp_id      = exprFactory.register_expression( new Temp      ::Builder(tempTag) );
  const Expr::ExpressionID gradt_id     = exprFactory.register_expression( new GradT_X   ::Builder(gradTTag,tempTag) );

# ifdef ENABLE_CUDA
  Expr::ExpressionBase& exprtempTag = exprFactory.retrieve_expression( tempTag, patch.id() );
  exprtempTag.set_gpu_runnable( false );
# endif

  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( heatFluxX_id, exprFactory, patch.id() );

  Expr::FieldManagerList& fml = patch.field_manager_list();
  {
    std::ofstream fout( "heatflux.dot" );
    tree.write_tree(fout);
  }

  tree.register_fields( fml );

  //
  // allocate all fields on the patch for this tree
  //
  fml.allocate_fields( patch.field_info() );
  fml.dump_fields(std::cout);

  //
  // bind fields to expressions
  //
  tree.bind_fields( fml );

  //
  // bind operators to expressions
  //
  tree.bind_operators( patch.operator_database() );

  //
  // initialize the fields
  //
  Expr::FieldMgrSelector<CellField>::type& cellFM = fml.field_manager< CellField>();
  CellField& temp = fml.field_manager<CellField>().field_ref(tempTag);
  int i=0;
  for( CellField::iterator it=temp.begin(); it!=temp.end(); ++it, ++i ) *it = i*i;

  //
  // execute the expression tree.
  //
  tree.lock_fields(fml);  // prevent fields from being allocated temporarily so that we can get them after graph execution.
  tree.execute_tree();

  const XSideField& heatFlux = fml.field_manager<XSideField>().field_ref(xHeatFluxTag);
  const XSideField& gradT    = fml.field_manager<XSideField>().field_ref(gradTTag    );

# ifdef ENABLE_CUDA
  const_cast<XSideField&>(heatFlux).add_device( CPU_INDEX );
  const_cast<XSideField&>(gradT).add_device(    CPU_INDEX );
# endif

  const XSideField::const_iterator iend=heatFlux.end();
  XSideField::const_iterator iflux=heatFlux.begin();
  const XSideField& cgradT = const_cast<XSideField&>(gradT);
  for( size_t i=0; iflux!=iend; ++iflux, ++i ){
    const double err = std::abs( *iflux + 2*cgradT[i] );
    if( err > 1.0e-10 ){
      std::cout << "ERROR: incorrect result at point " << i
                << "  (err=" << err << ")" << std::endl;
      error = true;
    }
  }

  std::cout << "Testing expression library on heat flux calculation...";
  if( error ){
    std::cout << "FAIL" << std::endl  << std::endl;
    return -1;
  }
  else{
    std::cout << "PASS" << std::endl << std::endl;
  }

  return 0;
}
