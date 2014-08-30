//--------------------------------------------------------------------
#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <stdexcept>
//--------------------------------------------------------------------
// Cantera includes for thermochemistry
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>    // defines class IdealGasMix
//--------------------------------------------------------------------
#include <tabprops/prepro/rxnmdl/MixtureFraction.h>
#include <tabprops/prepro/rxnmdl/EquilRxnMdl.h>
#include <tabprops/prepro/rxnmdl/FastChemRxnMdl.h>
#include <tabprops/StateTable.h>
//--------------------------------------------------------------------
#include "TestHelper.h"


using namespace std;

bool test_fast_chem( const int order )
{
  // test with gri 3.0 mech

  // set up the thermo object
  Cantera_CXX::IdealGasMix gas("gri30.cti","gri30");
  int ns = gas.nSpecies();

  std::vector<double> y_fuel, y_oxid;
  y_fuel.assign(ns,0.0);
  y_oxid.assign(ns,0.0);

  // set oxidizer composition
  y_oxid[gas.speciesIndex("O2")] = 0.21;
  y_oxid[gas.speciesIndex("N2")] = 0.79;

  // set fuel composition
  y_fuel[gas.speciesIndex("CH4")] = 1.0;

  cout << "Testing fast chemistry reaction model..." << flush;
  {
    FastChemRxnMdl adiabaticFastChem( gas, y_oxid, y_fuel, false, order, 51 );

    adiabaticFastChem.select_for_output( StateVarEvaluator::VISCOSITY   );
    adiabaticFastChem.select_for_output( StateVarEvaluator::DENSITY     );
    adiabaticFastChem.select_for_output( StateVarEvaluator::TEMPERATURE );
    adiabaticFastChem.select_species_for_output( "CO2", StateVarEvaluator::SPECIES );
    adiabaticFastChem.select_species_for_output( "CO2", StateVarEvaluator::MOLEFRAC);
    adiabaticFastChem.select_species_for_output( "H2O", StateVarEvaluator::SPECIES );
    adiabaticFastChem.select_species_for_output( "O2" , StateVarEvaluator::SPECIES );
    adiabaticFastChem.select_species_for_output( "CH4", StateVarEvaluator::SPECIES );
    adiabaticFastChem.select_species_for_output( "H2" , StateVarEvaluator::SPECIES );

    FastChemRxnMdl fastChem( gas, y_oxid, y_fuel, false, order, 51, 21 );

    fastChem.select_for_output( StateVarEvaluator::VISCOSITY   );
    fastChem.select_for_output( StateVarEvaluator::DENSITY     );
    fastChem.select_for_output( StateVarEvaluator::TEMPERATURE );
    fastChem.select_species_for_output( "CO2", StateVarEvaluator::SPECIES );
    fastChem.select_species_for_output( "CO2", StateVarEvaluator::MOLEFRAC);
    fastChem.select_species_for_output( "H2O", StateVarEvaluator::SPECIES );
    fastChem.select_species_for_output( "O2" , StateVarEvaluator::SPECIES );
    fastChem.select_species_for_output( "CH4", StateVarEvaluator::SPECIES );
    fastChem.select_species_for_output( "H2" , StateVarEvaluator::SPECIES );

    // calculate the equilibrium solution at all mixture fractions.
    try{
      adiabaticFastChem.implement();
      fastChem.implement();
    }
    catch ( const std::runtime_error & e ){
      cout << "FAIL!" << endl << e.what() << endl;
      return false;
    }

  }
  cout << "PASS" << endl;


  cout << endl << "Testing ability to load and query model..." << endl;
  //
  //  Read the results from disk and create the map object. We could
  //  create it directly, but this way we also exercise the reading.
  //
  StateTable mdl;
  StateTable amdl;
  try{
    mdl .read_table( "FastChem.tbl" );
    amdl.read_table( "AdiabaticFastChem.tbl" );
  }
  catch( const std::runtime_error &e ){
    cout << "FAIL!" << endl << e.what() << endl;
    return false;
  }

  //
  // query the FastChem model and dump temperature and a few key species out
  //
  ofstream fout( "fastChem.dat", ios::out );
  ofstream afout( "adiabaticFastChem.dat", ios::out );

  const int nf   = 55;
  const int ngam = 11;

  fout  << "#    f         gamma          T (K)            CO2             O2            CH4" << endl;
  afout << "#    f          T (K)         Viscosity       Density         CO2             O2            CH4" << endl;

  double indepVars[2];

  for( int i=0; i<nf; i++ ){
    indepVars[0] = double(i) / double(nf-1);

    afout << setiosflags(ios::scientific)
          << setprecision(6)
          << setw(15) << indepVars[0]
          << setw(15) << amdl.query( "Temperature", indepVars )
          << setw(15) << amdl.query( "Viscosity",   indepVars )
          << setw(15) << amdl.query( "Density",     indepVars )
          << setw(15) << amdl.query( "CO2", indepVars )
          << setw(15) << amdl.query( "H2O", indepVars )
          << setw(15) << amdl.query( "O2",  indepVars )
          << setw(15) << amdl.query( "CH4", indepVars )
          << setw(15) << amdl.query( "H2",  indepVars )
          << endl;

    for( int j=0; j<ngam; j++ ){

      indepVars[1] = double(2*j)/double(ngam-1) - 1.0;

      fout << setiosflags(ios::scientific)
           << setprecision(6)
           << setw(15) << indepVars[0]
           << setw(15) << indepVars[1]
           << setw(15) << mdl.query( "Temperature", indepVars )
           << setw(15) << mdl.query( "Viscosity",   indepVars )
           << setw(15) << mdl.query( "Density",     indepVars )
           << setw(15) << mdl.query( "CO2", indepVars )
           << setw(15) << mdl.query( "H2O", indepVars )
           << setw(15) << mdl.query( "O2",  indepVars )
           << setw(15) << mdl.query( "CH4", indepVars )
           << setw(15) << mdl.query( "H2",  indepVars )
           << endl;
    }
    fout << endl;
  }
  afout.close();
  fout.close();

  // if we got here, then we didn't discover any errors...
  cout << "PASS!" << endl
       << "  See interpolated data files: 'fastChem.dat' " << endl
       << "                          and: 'adiabaticFastChem.dat'" << endl
       << "  for interpolated results." << endl;
  return true;

}
//--------------------------------------------------------------------
bool test_adiab_eq( const int order )
{
  // test with gri 3.0 mech

  // set up the thermo object
  Cantera_CXX::IdealGasMix gas("gri30.cti","gri30");
  int ns = gas.nSpecies();

  std::vector<double> y_fuel, y_oxid;
  y_fuel.assign(ns,0.0);   y_oxid.assign(ns,0.0);

  // set oxidizer and fuel composition
  y_oxid[gas.speciesIndex("O2") ] = 0.21;    y_oxid[gas.speciesIndex("N2")] = 0.79;
  y_fuel[gas.speciesIndex("CH4")] = 0.221;   y_fuel[gas.speciesIndex("H2")] = 0.447;
  y_fuel[gas.speciesIndex("N2") ] = 0.332;

  cout << "Testing equilibrium reaction model..." << flush;

  AdiabaticEquilRxnMdl adiabaticEquil( gas, y_oxid, y_fuel, false, order, 101 );
  adiabaticEquil.implement();                 // calculate the full model

  // if we got here, then we didn't discover any errors...
  cout << "PASS!" << endl << endl;
  return true;
}
//--------------------------------------------------------------------
bool test_eq_hl( const int order )
{
  // test with gri 3.0 mech

  // set up the thermo object
  Cantera_CXX::IdealGasMix gas("gri30.cti","gri30");
  int ns = gas.nSpecies();

  std::vector<double> y_fuel, y_oxid;
  y_fuel.assign(ns,0.0);   y_oxid.assign(ns,0.0);

  // set oxidizer and fuel composition
  y_oxid[gas.speciesIndex("O2") ] = 0.21;    y_oxid[gas.speciesIndex("N2")] = 0.79;
  y_fuel[gas.speciesIndex("CH4")] = 0.221;   y_fuel[gas.speciesIndex("H2")] = 0.447;
  y_fuel[gas.speciesIndex("N2") ] = 0.332;

  cout << "Testing equilibrium reaction model with heat loss..." << endl;

  EquilRxnMdl eq( gas, y_oxid, y_fuel, false, order, 51, 21 );
  eq.set_fuel_temp( 300 ); eq.set_oxid_temp( 300 );
  eq.select_for_output( StateVarEvaluator::TEMPERATURE );
  eq.implement();                 // calculate the full model

  // if we got here, then we didn't discover any errors...
  cout << "PASS!" << endl << endl;
  return true;
}
//--------------------------------------------------------------------
bool
test_tecplot_output( )
{
  StateTable eqtbl;

  eqtbl.read_table("Equilibrium.tbl");

  vector<int> npts(2);  npts[0]=201; npts[1]=201;
  vector<double> hi(2); hi[0]=1.0;  hi[1]=1.0;
  vector<double> lo(2); lo[0]=0.0;  lo[1]=-1.0;

/*
  eqtbl.read_hdf5("AdiabaticFastChem");
  vector<int> npts(1);  npts[0]=21;
  vector<double> hi(1); hi[0]=1.0;
  vector<double> lo(1); lo[0]=0.0;
*/
  eqtbl.write_tecplot( npts,hi,lo );

  return true;
}
//--------------------------------------------------------------------
int main()
{
  TestHelper status(true);
  try{
    for( int iorder=1; iorder<4; ++iorder ){
      cout << "Interpolant order: " << iorder << endl;

      status( perform_mixfrac_tests(), "mixfrac" );
      status( test_fast_chem(iorder),  "fast chem" );
      status( test_adiab_eq(iorder),   "adiabatic equilibrium" );
      status( test_eq_hl(iorder),      "equilibrium" );
      status( test_tecplot_output(),   "tecplot" );
    }
    cout << "--------------------------------------------------" << endl;
    if( status.ok() ){
      cout << "PASS! All tests passed!" << endl;
      cout << "--------------------------------------------------" << endl;
      return 0;
    }
    else{
      cout << "FAIL!  At least one test failed" << endl;
      cout << "--------------------------------------------------" << endl;
    }
  }
  catch(Cantera::CanteraError&){
    Cantera::showErrors(cout);
  }
  catch( std::exception & e ){
    cout << e.what() << std::endl
         << "ABORTING!" << std::endl;
  }
  return -1;
}
