#include <iostream>
#include <sstream>

#include <expression/ExprLib.h>

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/OperatorDatabase.h>
#include <spatialops/structured/FieldHelper.h>
#include <spatialops/structured/Grid.h>

#include "TemperatureTransEqn.h"

#include <test/TestHelper.h>

//====================================================================

typedef SpatialOps::SVolField    CellField;
typedef SpatialOps::SSurfXField  XSideField;

typedef SpatialOps::BasicOpTypes<CellField>  OpTypes;
typedef OpTypes::GradX      GradXC2F;
typedef OpTypes::DivX       DivX;
typedef OpTypes::InterpC2FX InterpC2F;

//====================================================================

bool
test_integrator( const Expr::TSMethod method,
                 Expr::ExpressionFactory& factory,
		 Expr::ExprPatch& patch,
		 const SpatialOps::DoubleVec& spacing,
		 const int npts )
{
  const Expr::Tag tempTag     ( "Temperature",        Expr::STATE_N    );
  const Expr::Tag thermCondTag( "ThermalConductivity",Expr::STATE_NONE );
  const Expr::Tag gradTTag    ( "dT/dx",              Expr::STATE_NONE );
  const Expr::Tag heatFluxTag ("HeatFluxX",           Expr::STATE_NONE );

  TemperatureTransEqn tempEqn( tempTag, thermCondTag, gradTTag, heatFluxTag, factory, patch );

  Expr::FieldManagerList& fml = patch.field_manager_list();

  //
  // build the time-integrator and add the temperature equation to it.
  //
  Expr::TimeStepper timeIntegrator( factory, method, patch.id() );

  timeIntegrator.add_equation<TemperatureTransEqn::FieldT>
    ( tempEqn.solution_variable_name(), tempEqn.get_rhs_id() );

  timeIntegrator.finalize( fml, patch.operator_database(), patch.field_info() );
  {
    std::ofstream fout("tree.dot");
    timeIntegrator.get_tree()->write_tree(fout);
  }

  //  fml.dump_fields(cout);

  //
  // initial conditions
  //
  const Expr::Tag xtag("x",Expr::STATE_NONE);
  {
    Expr::ExpressionFactory factoryIC;

    factoryIC.register_expression( new Expr::GaussianFunction<CellField>::Builder( tempTag, xtag, 30, 0.2, 0.4, 1.0 ) );
    factoryIC.register_expression( new Expr::PlaceHolder<CellField>::Builder(xtag) );

    const Expr::ExpressionID icid = tempEqn.initial_condition( factoryIC );
    Expr::ExpressionTree icTree( icid, factoryIC, patch.id() );
    icTree.register_fields( fml );
    fml.allocate_fields( patch.field_info() );
    icTree.bind_fields( fml );
    icTree.bind_operators( patch.operator_database() );

    {
      std::ofstream fout("icTree.dot");
      icTree.write_tree(fout);
    }

    // set the mesh - we need this for the initial condition function....
    const SpatialOps::Grid grid( SpatialOps::IntVec(npts,1,1), spacing );
    CellField& x = fml.field_ref<CellField>( xtag );
    grid.set_coord<SpatialOps::XDIR>( x );

    icTree.execute_tree();

#   ifdef ENABLE_CUDA
    fml.field_ref<CellField>(tempTag).add_device( CPU_INDEX );
#   endif
    SpatialOps::write_matlab( fml.field_ref<CellField>(tempTag), "T0", true );
  }

  //
  // setup boundary conditions
  //
  tempEqn.setup_boundary_conditions( factory );

  // integrate for a while.
  double time=0;
  const double endTime=10.0;
  const double dt = 0.01;

  // lock fields that we want to write out so that their memory isn't reused
  fml.field_manager<XSideField>().lock_field(heatFluxTag);
  fml.field_manager<CellField >().lock_field(xtag);
  while( time<endTime ){
    timeIntegrator.step( dt );
    time += dt;
  }

  CellField& x    = fml.field_ref<CellField>( xtag    );
  CellField& temp = fml.field_ref<CellField>( tempTag );

# ifdef ENABLE_CUDA
  x   .add_device( CPU_INDEX );
  temp.add_device( CPU_INDEX );
  temp.set_device_as_active( CPU_INDEX ); // so that we can iterate it below
# endif

  SpatialOps::write_matlab(fml.field_ref<CellField >(xtag   ),"x",true);
  SpatialOps::write_matlab(fml.field_ref<CellField >(tempTag),"T",true);

  double Texpected;
  if( method==Expr::ForwardEuler ){
    Texpected=21.1853;
  }
  else{
    Texpected=21.1871;
  }

  for( CellField::const_iterator it=temp.begin(); it!=temp.end(); ++it ){
    //if(std::isnan(*it) ){  <- nvcc will not compile isnan... No idea why not.  The following is equivalent and compiles:
    if( *it != *it ){
      std::cout << std::endl << "***FAIL*** - detected NaN!" << std::endl;
      break;
    }
  }

  TestHelper status(false);
  status( std::abs(temp[npts/2] - Texpected ) < 1e-5 );
  if( std::abs(temp[npts/2] - Texpected ) >= 1e-5 ){
    std::cout << "***FAIL***" << std::endl
              << " Expected T[" << npts/2 << "] = " << Texpected << std::endl
              << " Found    T[" << npts/2 << "] = " << temp[npts/2] << std::endl;
  }

  return status.ok();
}

//====================================================================

int main()
{
  //
  // build a patch
  //
  const int npts = 100;
  Expr::ExprPatch patch(npts);

  SpatialOps::OperatorDatabase& opDB = patch.operator_database();

  // set mesh spacing
  const SpatialOps::DoubleVec spacing(1.0,1.0,1.0);

  //
  // register required operators
  //
  const bool bcx = patch.has_physical_bc_xplus();
  const bool bcy = patch.has_physical_bc_yplus();
  const bool bcz = patch.has_physical_bc_zplus();
  const SpatialOps::IntVec dim( patch.dim() );
  opDB.register_new_operator( new InterpC2F( SpatialOps::build_two_point_coef_collection( 0.5, 0.5 ) ) );
  opDB.register_new_operator( new GradXC2F( SpatialOps::build_two_point_coef_collection( -1.0/spacing[0], 1.0/spacing[0] ) ) );
  opDB.register_new_operator( new DivX( SpatialOps::build_two_point_coef_collection( -1.0/spacing[0], 1.0/spacing[0] ) ) );

  TestHelper status(true);
  try{
    Expr::ExpressionFactory exprFactory;
    status( test_integrator( Expr::ForwardEuler, exprFactory, patch, spacing, npts ), "Forward Euler" );
  }
  catch( std::exception& e ){
    std::cout << e.what() << std::endl << std::endl;
  }

  try{
    Expr::ExpressionFactory exprFactory;
    status( test_integrator( Expr::SSPRK3, exprFactory, patch, spacing, npts ), "SSPRK3" );
  }
  catch( std::exception& e ){
    std::cout << e.what() << std::endl << std::endl;
  }

  if( status.ok() ){
    std::cout << "\nPASS\n";
    return 0;
  }

  std::cout << "\nFAIL\n";
  return -1;
}

//====================================================================
