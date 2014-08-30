#include <expression/ExprLib.h>
#include <expression/Functions.h>

#include <spatialops/structured/FVStaggered.h>
#include <spatialops/structured/FieldHelper.h>

#include <iomanip>
#include <iostream>

#include <test/TestHelper.h>

typedef Expr::ExprPatch  PatchT;
typedef SpatialOps::SVolField  FieldT;

using namespace std;

//====================================================================

bool test_constant_function()
{
  const int npts = 10;
  PatchT patch(npts);

  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fieldMgr = fml.field_manager<FieldT>();

  Expr::Tag ytag ( "y" , Expr::STATE_NONE );
  const double a = 1.234;
  const Expr::ExpressionID yid = exprFactory.register_expression( new Expr::ConstantExpr<FieldT>::Builder(ytag,a) );

  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( yid, exprFactory, patch.id() );
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  tree.execute_tree();

  const FieldT& y = fieldMgr.field_ref(ytag );

# ifdef ENABLE_CUDA
  const_cast<FieldT&>(y).add_device( CPU_INDEX );
 #endif

  bool isokay = true;
  for( FieldT::const_iterator iy=y.begin(); iy!=y.end(); ++iy ){
    if( fabs(a-*iy) > 1e-15 ){
      isokay = false;
      cout << a << ", " << *iy << endl;
    }
  }

  return isokay;
}

//====================================================================

bool test_linear_function()
{
  const int npts = 10;
  PatchT patch(npts);

  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fieldMgr = fml.field_manager<FieldT>();

  Expr::Tag ytag ( "y" , Expr::STATE_NONE );
  Expr::Tag y2tag( "y2", Expr::STATE_NONE );
  Expr::Tag xtag ( "x" , Expr::STATE_NONE );

  fieldMgr.register_field( xtag );

  const double a = 2.0;
  const double b = 3.0;

  /*
   * Here we build two expressions:
   *    y  = ax+b
   *    y2 = y/a -b/a = x
   * This facilitates an easy validity check for the linear expression.
   */
  const Expr::ExpressionID yid = exprFactory.register_expression( new Expr::LinearFunction<FieldT>::Builder( ytag,  xtag, a, b ) );
  const Expr::ExpressionID y2id= exprFactory.register_expression( new Expr::LinearFunction<FieldT>::Builder( y2tag, ytag, 1.0/a, -b/a, true ) );

  // set x as a placeholder expression.  This just masks this field as
  // an expression so that it plays nice with other expressions.
  const Expr::ExpressionID xid = exprFactory.register_expression( new Expr::PlaceHolder<FieldT>::Builder(xtag) );

  //NOTE - xtag uses non-const iterator, so has to turn off GPU runnable
  // tag. consumers couldn't be added.
# ifdef ENABLE_CUDA
  Expr::ExpressionBase& expr = exprFactory.retrieve_expression( xtag, patch.id() );
  expr.set_gpu_runnable( false );
# endif
  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( y2id, exprFactory, patch.id() );
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  FieldT& x = fieldMgr.field_ref( xtag );
  double xpt = 0.0;
  for( FieldT::iterator ix=x.begin(); ix!=x.end(); ++ix ){
    *ix = xpt;  xpt+=1.1;
  }

  tree.execute_tree();

  const FieldT& y2 = fieldMgr.field_ref(y2tag);
  const FieldT& x2  = const_cast<FieldT&>(x);

# ifdef ENABLE_CUDA
  const_cast<FieldT&>(y2).add_device( CPU_INDEX );
# endif

  bool isokay = true;
  for( FieldT::const_iterator ix=x2.begin(), iy2=y2.begin(); ix!=x2.end(); ++ix, ++iy2 ){
    if( fabs(*ix - *iy2) > 1e-15 ){
      isokay = false;
      cout << *ix << ", " << *iy2 << ", " << *ix - *iy2 << endl;
    }
  }

  return isokay;
}

//====================================================================

bool test_gaussian_function()
{
  const int npts = 100;
  PatchT patch(npts);

  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fieldMgr = fml.field_manager<FieldT>();

  Expr::Tag ytag ( "y" , Expr::STATE_NONE );
  Expr::Tag xtag ( "x" , Expr::STATE_NONE );

  const double y0  = 1.1;
  const double x0  = 35.0;
  const double sig = 10.0;
  const double amp = 2.25;

  const Expr::ExpressionID yid = exprFactory.register_expression( new Expr::GaussianFunction<FieldT>::Builder( ytag, xtag, amp, sig, x0, y0 ) );

  // set x as a placeholder expression.  This just masks this field as
  // an expression so that it plays nice with other expressions.
  const Expr::ExpressionID xid = exprFactory.register_expression( new Expr::PlaceHolder<FieldT>::Builder(xtag) );

  //NOTE - xtag expression uses non-const iterator so has to turn off GPU runnable
  // tag. consumers couldn't be added.
# ifdef ENABLE_CUDA
  Expr::ExpressionBase& expr = exprFactory.retrieve_expression( xtag, patch.id() );
  expr.set_gpu_runnable( false );
# endif

  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( yid, exprFactory, patch.id() );
  {
    std::ofstream fout("gaussian.dot");
    tree.write_tree(fout);
  }
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  FieldT& x = fieldMgr.field_ref( xtag );
  double xpt = 0.0;
  for( FieldT::iterator ix=x.begin(); ix!=x.end(); ++ix ){
    *ix = xpt;  xpt+=1.0;
  }
# ifdef ENABLE_CUDA
  const_cast<FieldT&>(x).add_device( GPU_INDEX );
# endif
  tree.execute_tree();

  const FieldT& y = fieldMgr.field_ref(ytag );
  const FieldT& x2  = const_cast<FieldT&>(x);

# ifdef ENABLE_CUDA
  const_cast<FieldT&>(y).add_device( CPU_INDEX );
# endif

  bool isokay = true;
  for( FieldT::const_iterator ix=x2.begin(), iy=y.begin(); ix!=x2.end(); ++ix, ++iy ){
    const double tmp = *ix-x0;
    const double fx = y0+amp*exp(-tmp*tmp/(2.0*sig*sig));
    const double abserr = fabs( fx - *iy);
    if( abserr  > 1e-15 ){
      isokay = false;
      cout << setw(10) << *ix << ", " << setw(10) << *iy << ", " << setw(10) << fx << ", " << setw(10) << abserr << endl;
    }
  }

  if( !isokay ){
    SpatialOps::write_matlab(y,"y",true);
    SpatialOps::write_matlab(x,"x",true);
  }

  return isokay;
}

//====================================================================

bool test_sin_function()
{
  const int npts = 10;
  PatchT patch(npts);

  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fieldMgr = fml.field_manager<FieldT>();

  Expr::Tag ytag ( "y" , Expr::STATE_NONE );
  Expr::Tag xtag ( "x" , Expr::STATE_NONE );

  const double a = 1.1;
  const double b = 3.141593;
  const double c = 10.0;

  const Expr::ExpressionID yid = exprFactory.register_expression( new Expr::SinFunction<FieldT>::Builder( ytag, xtag, a, b, c ) );

  // set x as a placeholder expression.  This just masks this field as
  // an expression so that it plays nice with other expressions.
  const Expr::ExpressionID xid = exprFactory.register_expression( new Expr::PlaceHolder<FieldT>::Builder(xtag) );

  //NOTE - xtag expression uses non-const iterator so has to turn off GPU runnable
  // tag. consumers couldn't be added.
# ifdef ENABLE_CUDA
  Expr::ExpressionBase& expr = exprFactory.retrieve_expression( xtag, patch.id() );
  expr.set_gpu_runnable( false );
# endif

  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( yid, exprFactory, patch.id() );
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  FieldT& x = fieldMgr.field_ref( xtag );
  double xpt = 0.0;
  for( FieldT::iterator ix=x.begin(); ix!=x.end(); ++ix ){
    *ix = xpt;  xpt+=0.1;
  }
# ifdef ENABLE_CUDA
  const_cast<FieldT&>(x).add_device( GPU_INDEX );
# endif
  tree.execute_tree();

  const FieldT& y = fieldMgr.field_ref(ytag );
  const FieldT& x2  = const_cast<FieldT&>(x);

# ifdef ENABLE_CUDA
  const_cast<FieldT&>(y).add_device( CPU_INDEX );
# endif

  bool isokay = true;
  for( FieldT::const_iterator ix=x2.begin(), iy=y.begin(); ix!=x2.end(); ++ix, ++iy ){
    const double fx = a*std::sin(*ix*b)+c;
    const double abserr = fabs( fx - *iy);
    if( abserr  > 2e-15 ){
      isokay = false;
      cout << setw(10) << *ix << ", " << setw(10) << *iy << ", " << setw(10) << fx << ", " << setw(10) << abserr << endl;
    }
  }

  if( !isokay ){
    SpatialOps::write_matlab(y,"y",true);
    SpatialOps::write_matlab(x,"x",true);
  }

  return isokay;
}

//====================================================================


bool test_tanh_function()
{
  const int npts = 100;
  PatchT patch(npts);

  Expr::ExpressionFactory exprFactory;
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::FieldMgrSelector<FieldT>::type& fieldMgr = fml.field_manager<FieldT>();

  Expr::Tag ytag ( "y" , Expr::STATE_NONE );
  Expr::Tag xtag ( "x" , Expr::STATE_NONE );

  const double L1 = 2.1;
  const double L2 = 5.2;
  const double w  = 1.0;
  const double A = 2.0;

  const Expr::ExpressionID yid = exprFactory.register_expression( new Expr::DoubleTanhFunction<FieldT>::Builder( ytag, xtag, L1, L2, w, A ) );

  // set x as a placeholder expression.  This just masks this field as
  // an expression so that it plays nice with other expressions.
  const Expr::ExpressionID xid = exprFactory.register_expression( new Expr::PlaceHolder<FieldT>::Builder(xtag) );

  //NOTE - xtag expression uses non-const iterator so has to turn off GPU runnable
  // tag. consumers couldn't be added.
# ifdef ENABLE_CUDA
  Expr::ExpressionBase& expr = exprFactory.retrieve_expression( xtag, patch.id() );
  expr.set_gpu_runnable( false );
# endif

  //
  // build the expression tree
  //
  Expr::ExpressionTree tree( yid, exprFactory, patch.id() );
  tree.register_fields( fml );
  fml.allocate_fields( patch.field_info() );
  tree.bind_fields( fml );
  tree.bind_operators( patch.operator_database() );

  FieldT& x = fieldMgr.field_ref( xtag );
  double xpt = 0.0;
  for( FieldT::iterator ix=x.begin(); ix!=x.end(); ++ix ){
    *ix = xpt;  xpt+=0.1;
  }
# ifdef ENABLE_CUDA
  const_cast<FieldT&>(x).add_device( GPU_INDEX );
# endif
  tree.execute_tree();

  const FieldT& y = fieldMgr.field_ref(ytag );
  const FieldT& x2  = const_cast<FieldT&>(x);

# ifdef ENABLE_CUDA
  const_cast<FieldT&>(y).add_device( CPU_INDEX );
# endif

  bool isokay = true;
  for( FieldT::const_iterator ix=x2.begin(), iy=y.begin(); ix!=x2.end(); ++ix, ++iy ){
    const double fx =  (A/2.0) * (1.0+tanh((*ix-L1)/w))* (1.0-0.5*(1+tanh((*ix-L2)/w)));
    const double abserr = fabs( fx - *iy);
    if( abserr  > 1e-15 ){
      isokay = false;
      cout << setw(10) << *ix << ", " << setw(10) << *iy << ", " << setw(10) << fx << ", " << setw(10) << abserr << endl;
    }
  }

  if( !isokay ){
    SpatialOps::write_matlab(y,"y",true);
    SpatialOps::write_matlab(x,"x",true);
  }

  return isokay;
}


//====================================================================

int main()
{
  try{
    TestHelper status( true );
    status( test_constant_function(), "constant" );
    status( test_linear_function()  , "linear" );
    status( test_gaussian_function(), "gaussian" );
    status( test_sin_function()     , "sin" );
    status( test_tanh_function()    , "tanh" );
    if( status.ok() ) return 0;
  }
  catch( std::exception& e ){
    cout << e.what() << endl;
  }
  return -1;
}

//====================================================================
