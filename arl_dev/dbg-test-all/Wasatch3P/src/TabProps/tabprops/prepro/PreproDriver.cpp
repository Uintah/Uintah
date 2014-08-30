/*
 * Copyright (c) 2014 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <tabprops/prepro/PreproDriver.h>
#include <tabprops/prepro/ParseGroup.h>

#include <sstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <boost/algorithm/string.hpp>
#include <boost/foreach.hpp>

#include <tabprops/StateTable.h>
#include <tabprops/prepro/TableBuilder.h>

// mixing model includes
#include <tabprops/prepro/mixmdl/PresumedPDFMixMdl.h>
#include <tabprops/prepro/mixmdl/BetaMixMdl.h>
#include <tabprops/prepro/mixmdl/ClippedGaussMixMdl.h>
#include <tabprops/prepro/mixmdl/Integrator.h>
#include <tabprops/prepro/mixmdl/GaussKronrod.h>
#include <tabprops/prepro/mixmdl/MixMdlHelper.h>

// reaction model includes
#include <tabprops/prepro/rxnmdl/ReactionModel.h>
#include <tabprops/prepro/rxnmdl/StreamMixing.h>
#include <tabprops/prepro/rxnmdl/FastChemRxnMdl.h>
#include <tabprops/prepro/rxnmdl/EquilRxnMdl.h>
#include <tabprops/prepro/rxnmdl/MixtureFraction.h>
#include <tabprops/prepro/rxnmdl/JCSFlamelets.h>

#include <tabprops/prepro/rxnmdl/philCoalTable/PhilCoalTableImporter.h>

using namespace std;

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------
MixMdlParser::MixMdlParser( const ParseGroup & pg )
  : m_parseGroup( pg ),
    m_mixMdl( NULL )
{
  // resolve the mixing model
  const string nam = pg.get_attribute<string>("type");
  if( boost::iequals( nam, "Beta" ) ){
    m_mixMdl = new BetaMixMdl();
  }
  else if( boost::iequals( nam, "ClipGauss" ) ){
    m_mixMdl = new ClipGauss();
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: '" << nam << "' is not a supported Mixing Model." << endl
           << "       Supported models: " << endl
           << "         -> ClipGauss  (recommended)" << endl
           << "         -> Beta" << endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }


  // resolve information common to beta and clipped gaussian
  StateTable rxnMdl;
  rxnMdl.read_table( pg.get_value<string>( "ReactionModelFileName" )+".tbl" );

  m_convVarName = pg.get_value<string>("ConvolutionVariable","MixtureFraction");

  // build the mixing model helper
  MixMdlHelper mixMdlBuilder( *m_mixMdl, rxnMdl, m_convVarName,
                              pg.get_value<int>("order",1) );

  set_mesh( pg, mixMdlBuilder );

  // implement the mixing model
  mixMdlBuilder.implement();
}

//--------------------------------------------------------------------

MixMdlParser::~MixMdlParser()
{
  delete m_mixMdl;
}

//--------------------------------------------------------------------

void
MixMdlParser::set_mesh( const ParseGroup & pg,
                        MixMdlHelper & mm )
{
  const vector<string> indepVarNames = mm.reaction_model().get_indepvar_names();
  int npts;

  for( vector<string>::const_iterator inam = indepVarNames.begin();
       inam != indepVarNames.end();
       ++inam )
  {
    npts = 101;
    double minVal = 0.0;
    double maxVal = 1.0;

    vector<double> mesh;

    // see if we have a mesh set for this variable
    if( pg.has_child(*inam) ){
      const ParseGroup pgRxnVar = pg.get_child( *inam );
      npts = pgRxnVar.get_value<int>("npts",npts);

      // min/max values?
      minVal = pgRxnVar.get_value<double>( "min", minVal );
      maxVal = pgRxnVar.get_value<double>( "max", maxVal );

      cout << *inam << " : using " << npts
          << " points on interval [ " << minVal << ", " << maxVal << " ]" << endl;
      if( pgRxnVar.has_child("scale") && boost::iequals( pgRxnVar.get_value<string>("scale"), "log" ) ){
        cout << "    LOG SCALE" << endl;
        minVal = std::log10( minVal );
        maxVal = std::log10( maxVal );
        const double dx = (maxVal-minVal)/double(npts-1);
        mesh.resize(npts,0.0);
        for( size_t i=0; i<npts; ++i )  mesh[i] = pow( 10.0, double(i)*dx+minVal );
      }
    }
    else{
      cout << endl << "NOTE: using default mesh for ";
    }
    if( mesh.empty() ){
      mesh.resize(npts,0.0);
      const double dx = (maxVal-minVal)/double(npts-1);
      for( int i=0; i<npts; ++i )  mesh[i] = double(i)*dx + minVal;
    }
    mm.set_mesh( *inam, mesh );
  }

  // now, the convolution variable:
  npts = 51;
  double minVal=0.0, maxVal=1.0;
  if( pg.has_child( m_convVarName+"Variance" ) ){
    const ParseGroup pgConvVar = pg.get_child( m_convVarName+"Variance" );
    npts = pgConvVar.get_value("npts",npts);
    minVal = pgConvVar.get_value("min",minVal);
    maxVal = pgConvVar.get_value("max",maxVal);
  }

  cout << m_convVarName+"Variance" << " : using " << npts
       << " points on interval [ " << minVal << ", " << maxVal << " ]" << endl;

  const double dx = (maxVal-minVal)/double(npts-1);
  vector<double> gmesh;
  gmesh.assign(npts,0.0);
  for( int i=0; i<npts; ++i ) gmesh[i] = double(i)*dx;
  mm.set_mesh( m_convVarName+"Variance", gmesh );
}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------
RxnMdlParser::RxnMdlParser( const ParseGroup & pg )
  : m_parseGroup( pg ),
    m_rxnMdl( NULL ),
    m_gas( NULL )
{
  // resolve the model
  const string nam = pg.get_attribute<string>("type");
  if( boost::iequals(nam,"AdiabaticFastChem") || boost::iequals(nam,"FastChem") ){
    fastchem( pg );
  }
  else if( boost::iequals(nam,"AdiabaticEquilibrium") || boost::iequals(nam,"Equilibrium") ){
    equil( pg );
  }
  else if( boost::iequals(nam,"NonReacting") ){
    nonreact( pg );
  }
  else if( boost::iequals(nam, "SLFM") ){
    slfm_jcs( pg );
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: '" << nam << "' is not a supported reaction model." << endl
           << "  Supported models: " << endl
           << "   -> NonReacting: nonreacting mixture of two streams" << endl
           << "   -> FastChem: Burke-Schumann (infinitely fast) chemistry with heat loss" << endl
           << "   -> AdiabaticFastChem: adiabatic Burke-Schumann chemistry" << endl
           << "   -> Equilibrium: Thermochemical equilibrium with heat loss" << endl
           << "   -> AdiabaticEquilibrium: Adiabatic thermochemical equilibrium" << endl
           << "   -> SLFM: Steady laminar flamelet importer from JCS Flamelets" << endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  if( m_rxnMdl == NULL ){
    std::ostringstream errmsg;
    errmsg << "ERROR: No reaction model has been defined!" << endl
         << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  // set up output variables as necessary
  setup_output_vars( pg );

  try{
    cout << endl << "Implementing reaction model: '" << nam << "'" << endl;
    m_rxnMdl->implement();
  }
  catch(Cantera::CanteraError&){
    Cantera::showErrors(cout);
    throw std::exception();
  }
  catch( std::runtime_error & e ){
    cout << e.what() << std::endl;
    throw std::exception();
  }
}

//--------------------------------------------------------------------

RxnMdlParser::~RxnMdlParser()
{
  delete m_rxnMdl;
  delete m_gas;
}

//--------------------------------------------------------------------

void
RxnMdlParser::nonreact( const ParseGroup & pg )
{
  setup_cantera( pg );

  // set fuel and oxidizer composition
  const ParseGroup yfuel = pg.get_child( "FuelComposition"     );
  const ParseGroup yoxid = pg.get_child( "OxidizerComposition" );

  vector<double> yf, yo;
  bool haveMassFrac_o, haveMassFrac_f;
  get_comp( yfuel, yf, haveMassFrac_f, *m_gas );
  get_comp( yoxid, yo, haveMassFrac_o, *m_gas );

  if( haveMassFrac_o != haveMassFrac_f ){
    std::ostringstream errmsg;
    errmsg << "ERROR:  Invalid composition specified for group '" << pg.name()
           << "'  You must specify only mass fractions or only mole fractions for stream compositions" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw runtime_error( errmsg.str() );
  }

  const int nfpts = pg.get_value<int>("nfpts",101);
  const int order = pg.get_value<int>("order",1  );

  // set up the nonreacting case
  StreamMixing * const rxnmdl = new StreamMixing( *m_gas, yo, yf, haveMassFrac_f, order, nfpts );

    // set stream temperatures
  rxnmdl->set_oxid_temperature( get_oxid_temp(pg) );
  rxnmdl->set_fuel_temperature( get_fuel_temp(pg) );

  // set pressure
  rxnmdl->set_pressure( get_pressure(pg) );

  m_rxnMdl = rxnmdl;
}

//--------------------------------------------------------------------

void
RxnMdlParser::fastchem( const ParseGroup & pg )
{
  setup_cantera( pg );

  // set fuel and oxidizer composition
  const ParseGroup yfuel = pg.get_child( "FuelComposition"     );
  const ParseGroup yoxid = pg.get_child( "OxidizerComposition" );

  vector<double> yf, yo;
  bool haveMassFrac_o, haveMassFrac_f;
  get_comp( yfuel, yf, haveMassFrac_f, *m_gas );
  get_comp( yoxid, yo, haveMassFrac_o, *m_gas );

  if( haveMassFrac_o != haveMassFrac_f ){
    std::ostringstream errmsg;
    errmsg << "ERROR:  Invalid composition specified for group '" << pg.name()
           << "'  You must specify only mass fractions or only mole fractions for stream compositions"
           << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw runtime_error( errmsg.str() );
  }

  // do we have a mesh specified for mixture fraction?
  const int nfpts = pg.get_value<int>("nfpts",101);
  const int order = pg.get_value<int>("order",1  );
  FastChemRxnMdl* rxnmdl = NULL;
  if( boost::iequals( pg.get_attribute<string>("type"), "FastChem" ) ){
    rxnmdl = new FastChemRxnMdl( *m_gas,
                                 yo, yf,
                                 haveMassFrac_f,
                                 order,
                                 nfpts,
                                 pg.get_value<int>("NHeatLossPts",51),
                                 set_stretch_fac(pg,3.0) );
  }
  else if( boost::iequals( pg.get_attribute<string>("type"), "AdiabaticFastChem" ) ){
    rxnmdl = new FastChemRxnMdl( *m_gas,
                                 yo, yf,
                                 haveMassFrac_f,
                                 order,
                                 nfpts,
                                 set_stretch_fac(pg,3.0) );
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: '" << pg.get_attribute<string>("type") << "' is not a recognized reaction model." << endl
         << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  cout << " fst=" << rxnmdl->get_mixture_fraction()->stoich_mixfrac() << endl;

  rxnmdl->set_oxid_temperature( get_oxid_temp(pg) );
  rxnmdl->set_fuel_temperature( get_fuel_temp(pg) );
  rxnmdl->set_pressure( get_pressure(pg) );

  m_rxnMdl = rxnmdl;
}

//--------------------------------------------------------------------

void
RxnMdlParser::equil( const ParseGroup & pg )
{
  setup_cantera( pg );

  // set fuel and oxidizer composition
  const ParseGroup yfuel = pg.get_child( "FuelComposition"     );
  const ParseGroup yoxid = pg.get_child( "OxidizerComposition" );
  vector<double> yf, yo;
  bool haveMassFrac_o, haveMassFrac_f;
  get_comp( yfuel, yf, haveMassFrac_f, *m_gas );
  get_comp( yoxid, yo, haveMassFrac_o, *m_gas );

  if( haveMassFrac_o != haveMassFrac_f ){
    std::ostringstream errmsg;
    errmsg << "ERROR:  Invalid composition specified for group '" << pg.name()
           << "'  You must specify only mass fractions or only mole fractions for stream compositions"
           << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw runtime_error( errmsg.str() );
  }

  // do we have a mesh specified for mixture fraction?
  const int nfpts = pg.get_value<int>("nfpts",101);
  const int order = pg.get_value<int>("order",1  );

  if( boost::iequals( pg.get_attribute<string>("type"), "Equilibrium" ) ){
    EquilRxnMdl* rxnmdl = new EquilRxnMdl( *m_gas,
                                           yo, yf,
                                           haveMassFrac_f,
                                           order,
                                           nfpts,
                                           pg.get_value<int>("NHeatLossPts",51),
                                           set_stretch_fac(pg,3.0) );
    cout << " fst=" << rxnmdl->get_mixture_fraction()->stoich_mixfrac() << endl;

    rxnmdl->set_oxid_temp( get_oxid_temp(pg) );
    rxnmdl->set_fuel_temp( get_fuel_temp(pg) );
    rxnmdl->set_pressure( get_pressure(pg) );

    m_rxnMdl = rxnmdl;
  }
  else if( boost::iequals( pg.get_attribute<string>("type"), "AdiabaticEquilibrium" ) ){
      AdiabaticEquilRxnMdl* rxnmdl =
        new AdiabaticEquilRxnMdl( *m_gas,
                                  yo, yf,
                                  haveMassFrac_f,
                                  order,
                                  nfpts,
                                  set_stretch_fac(pg,3.0) );
    cout << " fst=" << rxnmdl->get_mixture_fraction()->stoich_mixfrac() << endl;

    rxnmdl->set_oxid_temp( get_oxid_temp(pg) );
    rxnmdl->set_fuel_temp( get_fuel_temp(pg) );
    rxnmdl->set_pressure( get_pressure(pg) );

    m_rxnMdl = rxnmdl;
  }
  else{
    std::ostringstream errmsg;
    errmsg << "ERROR: '" << pg.get_attribute<string>("type") << "' is not a recognized reaction model." << endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
}

	//--------------------------------------------------------------------

void
RxnMdlParser::slfm_jcs( const ParseGroup & pg )
{
  setup_cantera( pg );

  SLFMImporter * myLib = new SLFMImporter( *m_gas, pg.get_value<int>("order",3) );
  m_rxnMdl = myLib;

  // any other output???
  if( pg.has_child("MatlabFilePrefix") ){
    myLib->get_lib()->request_matlab_output( pg.get_value<string>("MatlabFilePrefix") );
  }

  if( pg.has_child("TextFilePrefix") ){
    myLib->get_lib()->request_text_output( pg.get_value<string>("TextFilePrefix") );
  }
}

//--------------------------------------------------------------------

void
RxnMdlParser::setup_cantera( const ParseGroup & pg )
{
  if( NULL != m_gas ){
    std::ostringstream errmsg;
    errmsg << "ERROR: Cantera has already been initialized.  Cannot initialize more than once!"
           << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }

  try{
    m_gas = new Cantera_CXX::IdealGasMix( pg.get_value<string>( "CanteraInputFile" ),
                                          pg.get_value<string>("CanteraGroupName","") );
  }
  catch( Cantera::CanteraError& e ){
    Cantera::showErrors();
    std::ostringstream errmsg;
    errmsg << e.what() << std::endl
           << "Error initializing cantera." << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
}

//--------------------------------------------------------------------

void
RxnMdlParser::get_comp( const ParseGroup & pg,
                        vector<double> & y,
                        bool & haveMassFrac,
                        Cantera_CXX::IdealGasMix & gas )
{
  // mole fractions or mass fractions?
  const string type = pg.get_attribute<string>("type");

  haveMassFrac = boost::iequals(type,"MassFraction");

  if( !boost::iequals(type,"MoleFraction" ) ){
    ostringstream errmsg;
    errmsg << "ERROR: The stream composition has not been specified within scope: " << endl
        << "       '" << pg.name() << "'" << endl
        << "       Fuel and oxidizer compositions are required!" << endl
        << __FILE__ << " : " << __LINE__ << std::endl;
    throw runtime_error( errmsg.str() );
  }

  typedef std::map<string,double> CompMap;
  CompMap comp;
  for( ParseGroupIterator ii=pg.begin("Species"); ii!=pg.end("Species"); ++ii ){
    comp[ ii->get_attribute<string>("name") ] = ii->get_value<double>();
//    cout << ii->get_attribute<string>("name") << " = " << ii->get_value<double>() << endl;
  }

  // trap possible incorrect species name specification.
  try{
    y.assign( gas.nSpecies(), 0.0 );
    for( CompMap::const_iterator ii=comp.begin(); ii!=comp.end(); ++ii ){
      const int ix = gas.speciesIndex( ii->first );
      if( ix < 0 || ix > gas.nSpecies()-1 ){
        std::ostringstream errmsg;
        errmsg << "ERROR: Invalid species name (" << ii->first << ")" << std::endl
               << __FILE__ << " : " << __LINE__ << std::endl;
        throw std::runtime_error( errmsg.str() );
      }
      y[ix] = ii->second ;
    }
  }
  catch( std::exception & e ){
    std::ostringstream errmsg;
    errmsg << e.what() << endl;
    errmsg << "ERROR: likely mismatch in species names!  Available species names follow:" << endl;
    const vector<string> & specnames = gas.speciesNames();
    for( vector<string>::const_iterator isrt=specnames.begin(); isrt!=specnames.end(); ++isrt )
      errmsg << "       " << *isrt << endl;

    errmsg << __FILE__ << " : " << __LINE__ << std::endl;

    throw std::runtime_error( errmsg.str() );
  }
}

//--------------------------------------------------------------------

double
RxnMdlParser::get_fuel_temp( const ParseGroup & pg )
{
  double t = 300;
  if( !pg.has_child("FuelTemperature") ){
    cout << "WARNING: fuel temperature will default to " << t << " K for model '"
         << pg.name() << "'" << endl;
  }
  else{
    t = pg.get_value<double>("FuelTemperature");
  }
  return t;
}

//--------------------------------------------------------------------

double
RxnMdlParser::get_oxid_temp( const ParseGroup & pg )
{
  double t=300;
  if( !pg.has_child("OxidizerTemperature") ){
    cout << "WARNING: oxidizer temperature will default to " << t << " K for model '"
         << pg.name() << "'" << endl;
  }
  else{
    t = pg.get_value<double>("OxidizerTemperature");
  }
  return t;
}

//--------------------------------------------------------------------

double
RxnMdlParser::get_pressure( const ParseGroup & pg )
{
  double pressure = 101325.0;
  if( !pg.has_child("Pressure") ){
    cout << "WARNING: system pressure will default to " << pressure << " Pa for model '"
         << pg.name() << "'" << endl;
  }
  else{
    pressure = pg.get_value<double>("Pressure");
  }
  return pressure;
}

//--------------------------------------------------------------------

void
RxnMdlParser::setup_output_vars( const ParseGroup & pg )
{
  // allow for multiple lines (so long as they are unique)
  for( ParseGroupIterator ii=pg.begin("SelectForOutput"); ii!=pg.end("SelectForOutput"); ++ii ){

    typedef vector<string> OutVars;
    const OutVars v = ii->get_value_vec<string>();

    // allow for multiple entries on a line
    BOOST_FOREACH( const std::string& nam, v ){
      std::cout << " -> outputting '" << nam << "'\n";
      m_rxnMdl->select_for_output( get_state_var(nam) );
    }
  }

  // Species Mass Fractions
  // allow for multiple lines (so long as they are unique)
  for( ParseGroupIterator ii=pg.begin("SelectSpeciesForOutput"); ii!=pg.end("SelectSpeciesForOutput"); ++ii ){
    const vector<string> sp = ii->get_value_vec<string>();
    // allow for multiple entries on a line
    for( vector<string>::const_iterator isp = sp.begin(); isp!=sp.end(); ++isp ){
      m_rxnMdl->select_species_for_output( *isp, StateVarEvaluator::SPECIES );
    }
  }

  // Species Mole Fractions
  // allow for multiple lines (so long as they are unique)
  for( ParseGroupIterator ii=pg.begin("SelectMoleFracForOutput"); ii!=pg.end("SelectMoleFracForOutput"); ++ii ){
    const vector<string> sp = ii->get_value_vec<string>();
    // allow for multiple entries on a line
    for( vector<string>::const_iterator isp = sp.begin(); isp!=sp.end(); ++isp ){
      m_rxnMdl->select_species_for_output( *isp, StateVarEvaluator::MOLEFRAC );
    }
  }
}

//--------------------------------------------------------------------

double
RxnMdlParser::set_stretch_fac( const ParseGroup & pg,
                               const double defaultFac )
{
  const double fac = pg.get_value<double>("GridStretchFactor",defaultFac);
  if( fac < 0.0 ){
    ostringstream errmsg;
    errmsg << "ERROR: Invalid grid stretching factor specified for: '"
           << pg.name() << "'" << std::endl
           << __FILE__ << " : " << __LINE__ << std::endl;
    throw std::runtime_error( errmsg.str() );
  }
  return fac;
}

//--------------------------------------------------------------------

//====================================================================


void parse_input( const string & file )
{
  ParseGroup pg( file );

  cout << endl
       << "Echo of the parsed input file follows:" << endl
       << "--------------------------------------" << endl;
  pg.display(cout);
  cout << "--------------------------------------" << endl << endl;

  // check for reaction models
  for( ParseGroupIterator irxn=pg.begin("ReactionModel"); irxn!=pg.end("ReactionModel"); ++irxn ){
    RxnMdlParser rxnMdl( *irxn );
//    // verify that we parsed all information specified for this reaction model
//    irxn.ensure_complete_parse_inquiry();
  }

  // check for mixing models
  for( ParseGroupIterator imix=pg.begin("MixingModel"); imix!=pg.end("MixingModel"); ++imix ){
      MixMdlParser mixMdl( *imix );
//    // verify that we parsed all information specified for this mixing model
//    mixMdlPg.ensure_complete_parse_inquiry();
  }

  if( pg.has_child("Coal") ){
    const ParseGroup coalpg = pg.get_child("Coal");
    coalpg.display(cout);
    PhilCoalTableReader ct( coalpg.get_value<string>("filename"),
                            coalpg.get_value<string>("OutputFileName","CoalModel"),
                            coalpg.has_child("SubSampleFirstDim") );
  }
}
//--------------------------------------------------------------------
string getp(int& i, int argc, char** args) {
    string a="--";
    if (i < argc-1) {
        a = string(args[i+1]);
    }
    if (a[0] == '-') {
        a = "<missing>";
    }
    else {
        i += 1;
    }
    return a;
}
//--------------------------------------------------------------------
void show_help()
{
  cout << endl
       << "mixrxn: Program to generate mixing and reaction model databases." << endl << endl
       << " options:" << endl
       << "   -i <file name>  specify the name of the input file (required)" << endl
       << endl;
}
//--------------------------------------------------------------------
int main( int argc, char** argv )
{
  cout << endl
       << "--------------------------------------------------------------------" << endl
       << "TabProps: a library for generating mixing and reaction models" << endl
       << "Author:   James C. Sutherland (James.Sutherland@utah.edu)" << endl
       << endl
       << "Version (date): " << TabPropsVersionDate << endl
       << "Version (hash): " << TabPropsVersionHash << endl
       << endl
       << "--------------------------------------------------------------------" << endl
       << endl;

  string inputFileName = "";
  if( argc == 1 ){
    show_help();
    return -1;
  }

  int i=1;
  while (i < argc) {
    const string arg = string(argv[i]);
    if (arg == "-i") {
      inputFileName = getp(i,argc,argv);
    }
    else if( arg == "-h" || arg == "-H" ){
      show_help();
    }
    else {
      cout << endl << endl << "unknown option:" << arg << endl;
      show_help();
      exit(-1);
    }
    ++i;
  }

  if( inputFileName == "" ){
    cout << "ERROR: No input file was specified!" << endl;
    show_help();
    return -1;
  }

  try{
    parse_input( inputFileName );
  }
  catch( exception & e ){
    cout << e.what() << endl << "ABORTING WITH FAILURE\n";
    return -1;
  }

  return 0;
}
