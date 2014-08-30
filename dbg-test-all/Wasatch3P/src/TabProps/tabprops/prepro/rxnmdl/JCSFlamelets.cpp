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

#include <cmath>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include <algorithm>
//--------------------------------------------------------------------
#include <boost/foreach.hpp>
//--------------------------------------------------------------------
#include <cantera/Cantera.h>
#include <cantera/IdealGasMix.h>
//--------------------------------------------------------------------
#include <tabprops/prepro/rxnmdl/JCSFlamelets.h>
#include <tabprops/StateTable.h>    // for table verification
//--------------------------------------------------------------------

using std::cout; using std::endl;
using std::string;
using std::vector;

//====================================================================

JCSFlamelet::JCSFlamelet( const int ns, const std::string file, const int order )
: nspec_( ns ),
  order_( order )
{
  chiSt_ = -1.0;
  npts_ = -1;
  if(nspec_ > nsmax){ // ERROR condition
    std::ostringstream errmsg;
    errmsg << "ERROR: requested number of species is too large!  Increase 'nsmax'" << endl;
    throw std::runtime_error( errmsg.str() );
  }
  load_flamelet(file);
}
//--------------------------------------------------------------------
JCSFlamelet::~JCSFlamelet()
{
  delete Tinterp_;
  delete Hinterp_;
  vector<InterpT*>::iterator iys;
  for( iys=Yinterp_.begin(); iys!=Yinterp_.end(); ++iys )
    delete *iys;
}
//--------------------------------------------------------------------
double
JCSFlamelet::get_chi_max() const
{
  std::vector<double>::const_iterator ii = std::max_element( chi_.begin(), chi_.end() );
  double chimax = *ii;
  return chimax;
}
//--------------------------------------------------------------------
StateEntry &
JCSFlamelet::query( const double f )
{
  stateTmp_.ns = nspec_;
  stateTmp_.temp = Tinterp_->value( &f );
  stateTmp_.enthalpy = Hinterp_->value( &f );
  vector<InterpT*>::const_iterator iys;
  int i=0;
  for( iys=Yinterp_.begin(); iys!=Yinterp_.end(); ++iys, ++i )
    stateTmp_.species[i] = (*iys)->value( &f );

  return stateTmp_;
}
//--------------------------------------------------------------------
void
JCSFlamelet::load_flamelet( const string& fnam )
{
  cout << "Loading flamelet file: " << fnam << endl;

  //-- Step 1: Determine number of entries and chew up header lines...
  int nskip = extract_header( fnam );

# ifdef DEBUG
  cout << endl << "skipping " << nskip << " lines in file " << fnam << endl;
# endif
  std::ifstream inFile(fnam.c_str(),std::ios::in);
  for(int i=1; i<=nskip; i++){
    string line;
    getline(inFile,line);
  }

# ifdef DEBUG
  cout << npts_ << " points in flamelet file..." << endl
       << nspec_ << " species in flamelet file..." << endl;
# endif

  allocate_memory();

  //-- read a line from the file.
  for(int i=0; i<npts_; i++){
    if(!inFile) break;  // error condition...

    double f;
    StateEntry state;
    inFile >> f >> state.temp;
    mixfr_.push_back(f);

    double ysum=0.0;
    int j=0;
    for (j=0; j<nspec_; j++){
      inFile >> state.species[j];
      ysum += state.species[j];
    }

    if( ysum > 1.01 | ysum < 0.01 ){
      cout << "***WARNING***" << endl
           << "   species do not sum to unity!.  Could be error in ns."
           << "   sum=" << ysum << endl
           << "   Species will be normalized." << endl;
    }

    //-- normalize species mass fractions to ensure that they sum to unity.
    for(j=0; j<nspec_; j++)
      state.species[j] /= ysum;
    //-- zero out remaining mass fractions...
    for(j=nspec_; j<nsmax; j++)
      state.species[j] = 0;
    double dissipRate;
    inFile >> dissipRate >> state.enthalpy;

    chi_.push_back( dissipRate );
    flmlt_.push_back( state );
    flmlt_.back().ns = nspec_;
#   ifdef DEBUG
    cout << i+1 << ": " << mixfr_[i] << endl; // << "  " << flmlt_[i] << endl;
#   endif
  }
  inFile.close();

  // interp the entries
  generate_interps();
}
//--------------------------------------------------------------------
void
JCSFlamelet::generate_interps()
{
  vector<double> depvar(npts_);

  // interp the temperature
  vector<double>::iterator idep = depvar.begin();
  vector<StateEntry>::const_iterator iflm;
  for( iflm =flmlt_.begin(); iflm!=flmlt_.end(); ++iflm, ++idep ){
    *idep = iflm->temp;
  }

  const bool clip = true;
  Tinterp_ = new Interp1D( order_, mixfr_, depvar, clip );

  // interp the enthalpy
  idep = depvar.begin();
  for( iflm =flmlt_.begin(); iflm!=flmlt_.end(); ++iflm, ++idep ){
    *idep = iflm->enthalpy;
  }
  Hinterp_ = new Interp1D( order_, mixfr_, depvar, clip );

  // interp the species
  for( int i=0; i<nspec_; ++i ){
    idep = depvar.begin();
    for( iflm=flmlt_.begin(); iflm!=flmlt_.end(); ++iflm, ++idep ){
      *idep = iflm->species[i];
    }
    Yinterp_.push_back( new Interp1D( order_, mixfr_, depvar, clip ) );
  }
}
//--------------------------------------------------------------------
void
JCSFlamelet::allocate_memory()
{
  flmlt_.reserve(npts_);
  mixfr_.reserve(npts_);
  chi_.reserve(npts_);
}
//--------------------------------------------------------------------
int
JCSFlamelet::extract_header( const string& filename )
{
  std::ifstream inFile(filename.c_str(),std::ios::in);

  char ctmp;
  string line;
  inFile >> ctmp >> chiSt_ >> npts_;

# ifdef DEBUG
  cout << chiSt_ << " " << npts_ << endl;
# endif

  getline(inFile,line);  // chew up first line
  getline(inFile,line);  // get second line to prepare for search below...

  // examine each line of the file.  look for species name list.
  // cound how many header lines there are and return this.
  int nskip=1;
  while(line.find("#",0) != line.npos){
    // look for the line containing the species information...?
    const size_t i_T = line.find( "T(K)", 0 );
    if( i_T != line.npos ){
#     ifdef DEBUG
      cout << line << endl << endl;
#     endif
      const string delim = " ";
      const size_t i_chi = line.find("chi",0);
      size_t istart, iend;

      // determine the starting index for the species name
      istart = line.find_first_of( delim, i_T );
      istart = line.find_first_not_of( delim, istart );

      int count=0;
      while(istart != line.npos && istart < i_chi ){

        // determine the ending index for the species name
        iend = line.find_first_of( delim, istart );

        // deal with case where we run out the end of the line.
        // this should never happen becaues "chi" and "enthalpy"
        // are after species.
        if( iend == line.npos )  iend = line.length();

        // add this species name to the list and increment counter.
        specName_.push_back( line.substr(istart,iend-istart) );
        count++;

        // setup for the next species
        istart = line.find_first_not_of( delim, iend );
        if(count>nsmax) break;
      }
      if(count != nspec_){  // ERROR condition
        std::ostringstream errmsg;
        errmsg << "ERROR: nspec inconsistent with number of species " << endl
               << "       in flamelet file: " << filename << endl
               << "       found " << count << " and expecting " << nspec_ << endl;
        throw std::runtime_error( errmsg.str() );
      }

#     ifdef DEBUG
      cout << "------------------------------------" << endl;
      cout << "here come the species names: " << endl;
      for(int i=1; i<nspec_; i++)
        cout << specName_[i] << endl;
      cout << "------------------------------------" << endl << endl;
#     endif
    }
    getline(inFile,line);
    nskip++;
    if(nskip>10) break;
  }
  inFile.close();

  return nskip;
}

//==============================================================================

std::ostream& operator << (std::ostream& os, const StateEntry& st)
{
  using namespace std;
  int hi = st.ns;
  if( hi <= 0 | hi > nsmax ) hi=nsmax;
  os << setw(15) << fixed << setprecision(6)
     << st.temp;
  for (int i=0; i<st.ns; i++)
       os << setw(15) << scientific << setprecision(6)
          << st.species[i];
  os << setw(15) << scientific
     << st.enthalpy;
  return os;
}

//==============================================================================

JCSFlameLib::JCSFlameLib( const int ns,
                          const int order )
: order_( order )
{
  dumpMatlab_ = false;
  dumpText_   = false;

  Tinterp_ = NULL;
  Hinterp_ = NULL;

  nspec     = ns;
  n_entries = 0;

  read_flamelib();
}
//--------------------------------------------------------------------
JCSFlameLib::~JCSFlameLib()
{
  vector<JCSFlamelet*>::iterator ii;
  for( ii = flmlts.begin(); ii != flmlts.end(); ii++ ){
    delete *ii;
  }

  delete Tinterp_;
  delete Hinterp_;
  vector<const InterpT*>::const_iterator iys;
  for( iys=Yinterp_.begin(); iys!=Yinterp_.end(); ++iys )
    delete *iys;

}
//--------------------------------------------------------------------
void
JCSFlameLib::query( const double f,
                 const double chi,
                 StateEntry& state )
{
  const double xx[] = { f, chi };

  state.ns = nspec;
  state.temp = Tinterp_->value( xx );
  state.enthalpy = Hinterp_->value( xx );
  for( int i=0; i<nspec; ++i )
    state.species[i] = Yinterp_[i]->value( xx );
  return;
}
//--------------------------------------------------------------------
void
JCSFlameLib::read_flamelib()
{
  //-- generate a list of files which will comprise the library.

# ifdef PC
  // file list command for PCs running any flavor of DOS
  system("dir *.flm");
  system("dir /B *.flm > filelist.dat");
# else
  // file list command for LINUX/UNIX
  system("ls *.flm > filelist.dat");
# endif

  //-- Assemble the flamelet library.

  std::ifstream lsfil;
  string fnam;
  n_entries=0;
  lsfil.open("filelist.dat");  // if open fails, throw an exception!

  while( lsfil ){
    lsfil >> fnam;

  if( fnam == "" ) break;
    n_entries++;

#   ifdef DEBUG
    cout << "Initializing for: " << fnam << "  with " << nspec_ << " species";
#   endif
    //-- load this flamelet from disk and put the new flamelet in the data structure.
    flmlts.push_back( new JCSFlamelet(nspec,fnam,order_) );

#   ifdef DEBUG
    cout << " ...done." << endl << " " << n_entries << ":  chi_o="
         << flmlts[n_entries-1]->get_chi_max() << endl;
#   endif

    fnam = "";
  }
  lsfil.close();

  //-- delete the temporary file from disk
# ifdef PC
  system("del filelist.dat");
# else
  system("rm filelist.dat");
# endif

  //-- sort the library
  std::sort( flmlts.begin(), flmlts.end(), sort_ascending() );

  //-- store the maximum dissipation rate as the index for the library
  vector<JCSFlamelet*>::const_iterator ii;
  for( ii = flmlts.begin(); ii != flmlts.end(); ii++ ){
    chi_o.push_back( (*ii)->get_chi_max() );
  }
}
//--------------------------------------------------------------------
void
JCSFlameLib::generate_table( TableBuilder & table,
                             const string & filePrefix,
                             Cantera_CXX::IdealGasMix & gas )
{
  // set up the mesh
  const vector<double> fmesh = flmlts[0]->get_mixfr();

  table.set_mesh(0,fmesh);
  table.set_mesh(1,chi_o);

  vector<double> xx( 2,              0.0 );
  vector<double> ys( gas.nSpecies(), 0.0 );
  int ichi = 0;
  for( vector<JCSFlamelet*>::const_iterator iflm = flmlts.begin(); iflm!=flmlts.end(); ++iflm, ++ichi ){
    const JCSFlamelet & flamelet = **iflm;
    xx[1] = flamelet.get_chi_max();

    if( xx[1] != chi_o[ichi] ){
      std::ostringstream errmsg;
      errmsg << "ERROR: Inconsistent dissipation rate encountered."
             << "       See JCSFlameLib::generate_table()" << endl;
      throw std::runtime_error( errmsg.str() );
    }

    for( vector<double>::const_iterator imf=fmesh.begin(); imf!=fmesh.end(); ++imf ){
      xx[0] = *imf;

      const double T = flamelet.get_tinterp()->value( &xx[0] );
      const double P = 101325.0;  // Pressure (Pa)
      for( int i=0; i<nspec; ++i ){
        ys[i] = flamelet.get_yinterp(i)->value( &xx[0] );
      }
      table.insert_entry( xx, T, P, ys );
    }
  }

  table.set_filename( filePrefix );
  table.generate_table();

  verify_table( filePrefix + ".tbl" );
}
//--------------------------------------------------------------------
void
JCSFlameLib::verify_table( const string filename ) const
{
  // load the table from disk
  std::cout << "Loading " << filename << std::endl;
  StateTable table;
  table.read_table( filename );

  const InterpT* const tblTsp = table.find_entry( "Temperature" );

  bool pointsInterpolatedCorrectly = true;
  if( tblTsp == NULL ){
    cout << "Could not verify table since temperature was not found." << endl;
    pointsInterpolatedCorrectly = false;
  }
  else{
    double indepVar[2];
    for( vector<JCSFlamelet*>::const_iterator iflmlt = flmlts.begin(); iflmlt!=flmlts.end(); ++iflmlt )
    {
      const JCSFlamelet * const flamelet = *iflmlt;
      indepVar[1] = flamelet->get_chi_max();

      const vector<double> & mixfr = flamelet->get_mixfr();
      const vector<StateEntry> & state = flamelet->get_state();
      vector<StateEntry>::const_iterator istate = state.begin();
      for( vector<double>::const_iterator imf=mixfr.begin(); imf!=mixfr.end(); ++imf, ++istate )
      {
        indepVar[0] = *imf;

        try{
          // get percent difference between the temperature for this flamelet at this mixture
          // fraction and the interpolated value for the temperature.
          double Tperr = 100.0*(istate->temp - tblTsp->value(indepVar) )/istate->temp;
          if( std::abs(Tperr) > 1.0e-10 ){
            cout << "PROBLEMS at [f,chi] = [" << indepVar[0] << "," << indepVar[1]
                 << "]!  Temperature: " << istate->temp << ", " << tblTsp->value(indepVar) << ", %error=" << Tperr << endl;
            pointsInterpolatedCorrectly = false;
          }
        }
        catch( std::runtime_error& err ){
          cout << err.what() << endl
               << "ERROR interpolating flamelet table for f="
               << *imf << ", chi=" << indepVar[1] << "." << endl;
        }
      }
    }
    if( pointsInterpolatedCorrectly ){
      cout << "The table correctly interpolates the flamelet solutions :)" << endl;
    }
    else{
      cout << "********" << endl
           << "WARNING: the table does not interpolate the flamelet solutions!  This likely indicates a problem with the table!"
           << "********" << endl;
    }
    /*
    // interactive diagnostics:
    bool more = true;
    while( true ){
    cout << "Enter query f: ";   cin >> indepVar[0];
    if( indepVar[0] < 0.0 || indepVar[0] > 1.0 ) break;
    cout << "  Enter query chi: "; cin >> indepVar[1];
    if( indepVar[1] < 0.0 ) break;
    cout << "  T=" << tblTsp->value(indepVar) << endl;
    }
    */
  }
  if( dumpText_   ) dump( table );
  if( dumpMatlab_ ) dump_matlab( table );
}
//--------------------------------------------------------------------
void
JCSFlameLib::request_matlab_output( const string filename )
{
  dumpMatlab_ = true;
  matlabPrefixName_ = filename;
}
//--------------------------------------------------------------------
void
JCSFlameLib::request_text_output( const string filename )
{
  dumpText_ = true;
  textPrefixName_ = filename;
}
//--------------------------------------------------------------------
void
JCSFlameLib::dump( const StateTable& table ) const
{
  using namespace std;

  ofstream fout( (textPrefixName_ + ".txt").c_str(), std::ios::out );

  const vector<string> & specNames = (*flmlts.begin())->get_spec_names();
  fout << "#"
       << setw(10) << "f"
       << setw(15) << "chi";
  BOOST_FOREACH( const std::string& depVarName, table.get_depvar_names() ){
    fout << setw(15) << depVarName;
  }
  fout << endl;

  vector<JCSFlamelet*>::const_iterator iflm;
  for( iflm=flmlts.begin(); iflm!=flmlts.end(); iflm++ ){
    const JCSFlamelet * myFlamelet = *iflm;
    const vector<double> & f = myFlamelet->get_mixfr();
    const double chi = myFlamelet->get_chi_max();
    for( int i=0; i<myFlamelet->get_n_pts(); ++i ){
      double indepVars[2] = { f[i], chi };
      fout << setw(10) << setprecision(6) << fixed
           << f[i]
           << setw(15) << setprecision(6) << scientific
           << myFlamelet->get_chi_max();
      BOOST_FOREACH( const std::string& depVarName, table.get_depvar_names() ){
        fout << setw(15) << setprecision(6) << scientific
             << table.query( depVarName, indepVars );
      }
      fout << endl;
    }
  }

  fout.close();
}
//--------------------------------------------------------------------
void
JCSFlameLib::dump_matlab( const StateTable& table ) const
{
  using namespace std;
  ofstream fout( string(matlabPrefixName_+".m").c_str(), ios::out );

  const vector<string> & specNames = (*flmlts.begin())->get_spec_names();
  fout << "names = {" << endl
       << " 'mixfr'" << endl
       << " 'chi_o'"   << endl;
  BOOST_FOREACH( const std::string& depVarName, table.get_depvar_names() ){
    fout << " '" << depVarName << "'\n";
  }
  fout << "};" << endl;

  fout << "flamelet = cell(" << flmlts.size() << ",1);" << endl << endl;

  int i=0;
  vector<JCSFlamelet*>::const_iterator iflm;
  for( iflm=flmlts.begin(); iflm!=flmlts.end(); iflm++, i++ ){
    const JCSFlamelet * myFlamelet = *iflm;
    const vector<double> & f = myFlamelet->get_mixfr();
    const double chi = myFlamelet->get_chi_max();
    const vector<StateEntry> & state = myFlamelet->get_state();
    fout << "flamelet{" << i+1 << "} = [ ..." << endl;
    for( int i=0; i<myFlamelet->get_n_pts(); i++ ){
      double indepVars[2] = { f[i], chi };
      fout << setw(9)  << setprecision(6) << fixed      << f[i]
           << setw(15) << setprecision(6) << scientific << chi;
      BOOST_FOREACH( const std::string& depVarName, table.get_depvar_names() ){
        fout << setw(15) << setprecision(6) << scientific
             << table.query( depVarName, indepVars );
      }
      fout << "; ..." << endl;
    }
    fout << " ];" << endl << endl;
  }

  fout.close();
}
//==============================================================================

SLFMImporter::SLFMImporter( Cantera_CXX::IdealGasMix & gas, const int order )
  : ReactionModel( gas, indep_var_names(), order, "SLFM" ),
    m_flmLib( new JCSFlameLib( gas.nSpecies(), order ) )
{
  tableBuilder_.set_filename( "AdiabaticFlamelets" );
}
//--------------------------------------------------------------------
SLFMImporter::~SLFMImporter()
{
  delete m_flmLib;
}
//--------------------------------------------------------------------
void
SLFMImporter::implement()
{
  m_flmLib->generate_table( tableBuilder_, "SLFM", gasProps_ );
}
//--------------------------------------------------------------------
vector<string> &
SLFMImporter::indep_var_names()
{
  static vector<string> names(2);
  names[0] = "MixtureFraction";
  names[1] = "DissipationRate";
  return names;
}

//==============================================================================


/*
// Test utility for the Flamelet class
bool test_flamelet()
{
  string fnam = "stdy_010.000.flm";
  int ns=12;
  JCSFlamelet flmlt(ns,fnam);

  double f = 0.1;
  cout << "Enter the mixture fraction: ";
  cin >> f;
  while( f >= 0.0 && f <= 1.0){
    const StateEntry & state = flmlt.query(f);
    cout << endl << state << endl;
    cout << endl << "Chi max: " << flmlt.get_chi_max() << endl;
    cout << endl << "Enter the mixture fraction: ";
    cin >> f;
  }
  cout << endl << "done!" << endl;
  return true;
}



// Test utility for the JCSFlameLib and JCSFlamelet classes...
bool test_flamelib()
{
  int ns=12;
  double f, chi;
  JCSFlameLib myflamelib(ns);
  myflamelib.dump("flameLib.dat");
  myflamelib.dump_matlab("SLFM");
//  return true;
  cout << "dissipation rate range: [" << myflamelib.get_chi_min()
       << "," << myflamelib.get_chi_max() << "]" << endl;
  for( int i=0; i<=20; i++){
    cout << "enter f: ";
    cin >> f;

    if(f<0.0 | f>1.0) break;

    cout << "enter chi: ";
    cin >> chi;
    if( chi <= 0.0 | chi > myflamelib.get_chi_max() ) break;

    StateEntry mystate;
    myflamelib.query(f, chi, mystate);
    cout << endl << mystate << endl;
  }
  cout << endl << "done." << endl;
  return true;
}


int main()
{
  //  test_flamelet();
  test_flamelib();
  cout << "DONE" << endl;
}
*/
