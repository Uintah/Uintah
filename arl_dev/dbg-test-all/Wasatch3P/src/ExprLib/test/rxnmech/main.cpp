#include <expression/ExprLib.h>

#include "RHS.h"

using namespace std;
using namespace SpatialOps;

double check_results( const vector<SingleValueField*>& vars,
                      const vector<double> ic,
                      const double k1,
                      const double k2,
                      const double time )
{
  BOOST_FOREACH( SingleValueField* fld, vars ){
    fld->set_device_as_active( CPU_INDEX );
  }
  const double cA0 = ic[0];
//  const double cB0 = ic[1];
//  const double cC0 = ic[2];

  double exact[3];
  exact[0] = cA0 * exp( -k1 * time );
  exact[1] = -cA0 * k1/(k1-k2) * ( exp(-k1*time) - exp(-k2*time) );
  exact[2] = cA0 / (k1-k2) * (k1-k2-k1*exp(-k2*time) + k2*exp(-k1*time) );

  double l2=0;
  for( size_t i=0; i<3; ++i ){
    const double err = (*vars[i])[0] - exact[i];
    l2 += err*err;
  }
  return sqrt(l2);
}

//--------------------------------------------------------------------

int main()
{
  vector<string> solnVarNames;
  solnVarNames.push_back("A");
  solnVarNames.push_back("B");
  solnVarNames.push_back("C");

  Expr::TagList ctags, rhstags;
  for( size_t i=0; i<3; ++i ){
    ctags.push_back( Expr::Tag( solnVarNames[i], Expr::STATE_N ) );
    rhstags.push_back( Expr::Tag( solnVarNames[i] + "_rhs", Expr::STATE_NONE ) );
  }

  Expr::ExpressionFactory exprFactory;
  Expr::ExprPatch patch(1);
  Expr::FieldManagerList& fml = patch.field_manager_list();
  Expr::TimeStepper ts( exprFactory, Expr::SSPRK3 );

  Expr::FieldMgrSelector<SingleValueField>::type& fmdbl = fml.field_manager<SingleValueField>();

  const double k1 = 2.0;
  const double k2 = 1.0;
  const Expr::ExpressionID rhsID
    = exprFactory.register_expression( new RHS::Builder( rhstags, ctags, k1, k2 ) );

  ts.add_equations<SingleValueField>( solnVarNames, rhsID );

  const Expr::Tag timeTag( "time", Expr::STATE_NONE );
  exprFactory.register_expression( new Expr::PlaceHolder<SingleValueField>::Builder(timeTag) );

  try{
    ts.finalize( fml, OperatorDatabase(), patch.field_info() );
  }
  catch( std::runtime_error& err ){
    std::cout << err.what() << std::endl;
    return -1;
  }

  {
    ofstream fout( "rhs.dot" );
    ts.get_tree()->write_tree( fout );
    fout.close();
  }

  // grab the fields to set Initial conditions on them.
  vector<SingleValueField*> vars;
  BOOST_FOREACH( const Expr::Tag& tag, ctags ){
    vars.push_back( &fmdbl.field_ref( tag ) );
  }

  vector<double> ics;  // initial conditions
  ics.push_back( 2.0 );
  ics.push_back( 0.0 );
  ics.push_back( 0.0 );

  std::vector<double> l2err, dtlist;
  double dt = 0.5;
  const size_t nt=10;
  const double tend = 2.0;
  SingleValueField& t = fmdbl.field_ref( timeTag );

  for( size_t idt=0; idt<nt; ++idt ){
    t <<= 0;

    // set initial conditions
    for( size_t i=0; i<3; ++i )  *(vars[i]) <<= ics[i];


# ifdef ENABLE_CUDA
  t.validate_device( CPU_INDEX );
  t.set_device_as_active( CPU_INDEX );
# endif
      // integrate in time
    while( t[0]<tend ){
#     ifdef ENABLE_CUDA
      t.set_device_as_active( GPU_INDEX );
#     endif
      ts.step(dt);
#     ifdef ENABLE_CUDA
      t.validate_device( CPU_INDEX );
      t.set_device_as_active( CPU_INDEX );
#     endif
    }

    l2err.push_back( check_results( vars, ics, k1, k2, t[0] ) );
    dtlist.push_back( dt );

    // cut the timestep in half and run again to look at convergence
    dt *= 0.5;
    t <<= 0.0;
  }

  ofstream fout( "conv.csv" );
  fout << "dt, L2 error, first, second, third, fourth" << endl;
  for( size_t i=0; i<nt; ++i ){
    fout << dtlist[i] << ", " << l2err[i];
    for( size_t iord=1; iord<5; ++iord ){
      const double err = l2err[0] * std::pow( dtlist[i]/dtlist[0], double(iord) );
      fout << ", " << err;
    }
    fout << endl;
  }

  // verify that the solution is at lest third order.  This is easily
  // done by plotting the data in the .csv file, but here we just get
  // a ballbark estimate for testing.
  const double err3ord = l2err[0] * std::pow( dtlist[nt-1]/dtlist[0], 3.0 );
  if( 10*err3ord > l2err[nt-1] ){
    cout << "PASS" << endl;
    return 0;
  }

  cout << "FAIL" << endl;
  return -1;

}
