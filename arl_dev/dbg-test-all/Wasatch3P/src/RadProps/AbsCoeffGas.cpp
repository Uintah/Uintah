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

/**
 * \file AbsCoeffGas.cpp
 */

#define PI 3.14159265

#include "AbsCoeffGas.h"
#include "RadiativeSpecies.h"
#include <iostream>
#include <fstream>   // file I/O
#include <iomanip>   // format manipulation
#include <string>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <cassert>
#include <limits>
#include <algorithm>

#ifdef RadProps_ENABLE_PREPROCESSOR
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>
#endif

using namespace std;

/**
 * @brief Implements the composite trapezoid rule for uniformly spaced data.
 * @fn double trapz( const double dx, const vector<double>& fx )
 * @param dx the spacing (assumed constant)
 * @param fx the function values
 * @return the integral
 */
double trapz( const double dx, const vector<double>& fx )
{
  const size_t n = fx.size();  assert( n>2 );
  // composite trapezoid rule
  double integral = 0.5*(fx[0] + fx[n-1]);
  for( size_t i=1; i<n-1; ++i ){
    integral += fx[i];
  }
  integral *= dx;
  return integral;
}


/**
 * @fn double planck_blackbody_intensity( const double eta, const double tref )
 * @param eta  wavenumber (1/cm)
 * @param tref temperature (K)
 * @return the Planck blackbody intensity (J/cm^2)
 *  calculates
 *  \f[
 *  I(\nu,T) = 2 h c^2 \nu^3 \frac{1}{\exp(\frac{h c \nu}{k T})-1}
 *  \f]
 *  where c is the speed of sound, k is the Boltzmann constant, h is the Planck constant.
 *  See <a href="http://en.wikipedia.org/wiki/Planck%27s_law#Different_forms"> this link</a> for more information.
 */
inline double planck_blackbody_intensity( const double eta, const double tref )
{
  const double h = 6.626070e-34;   // Planck constant (J s);
  const double c = 2.997925e10;    // Speed of light in vacuum (cm/s);
  const double k = 1.380658e-23;   // Boltzmann constant (J/K);
  return 2*h*c*c*pow(eta,3.0) / ( (exp(h*c*eta/(k*tref)) - 1.0) );
}


namespace detail{

template< typename IndexT, typename ValT >
IndexT
index_finder( const ValT& x,
              const std::vector<ValT>& xgrid,
              const bool allowClipping=false )
{
  const size_t nx = xgrid.size();
  IndexT ilo = 0;
  IndexT ihi = nx-1;

  if( allowClipping && x<xgrid.front() ) return ilo;
  if( allowClipping && x>xgrid.back()  ) return ihi;

  // sanity check
  if( x<xgrid.front() || x>xgrid.back() ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl
        << "root is not bracketed!" << endl;
    throw std::runtime_error(msg.str());
  }

  // regula falsi method to find lower index

  while( ihi-ilo > 1 ){
    const ValT m = ( xgrid[ihi]-xgrid[ilo] ) / ValT( ihi-ilo );
    const IndexT c = std::max( ilo+1, IndexT( ihi - (xgrid[ihi]-x)/m ) );
    assert( c>0 && c<xgrid.size() );
    if( x >= xgrid[c] )
      ilo = std::min(ihi-1,c);
    else
      ihi = std::max(ilo+1,c);
  }
  // error checking:
  if( !allowClipping && (x > xgrid[ihi] || x < xgrid[ilo]) ){
    std::ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << std::endl
        << " Regula falsi failed to converge properly!" << std::endl
        << " Target x=" << x << std::endl
        << " xlo=" << xgrid[ilo] << ", xhi=" << xgrid[ihi]
        << std::endl << std::endl;
    throw std::runtime_error( msg.str() );
  }

  return ilo;
}

}

struct RadiationData{
  double waveNumber;    ///< wavenumber, cm
  double absCoeff;      ///< absorption coefficient, 1/cm
};

std::ostream& operator<<(std::ostream& out, const RadiationData& data ){
  out << data.waveNumber << "," << data.absCoeff;
  return out;
}

std::ifstream& operator>>( std::ifstream& in, RadiationData& data ){
  in >> data.waveNumber >> data.absCoeff;
  return in;
}


SpeciesAbsCoeff::SpeciesAbsCoeff( const string& filename, const double temperature )
 : temperature_( temperature )
{

  std::ifstream file;
  file.open(filename.c_str(),std::ios::in); // opens as ASCII!
  if( file.bad() ){
    std::ostringstream msg;
    msg << "ERROR! Could not open file '" << filename << "'";
    throw std::runtime_error( msg.str() );
  }

  RadiationData rdata;
  file >> rdata;
  npts_ = 1;
  loWaveNum_ = rdata.waveNumber;        // the lowest wavenumber in the dataset
  double wvn = rdata.waveNumber;        // the previous wave number we read
  double wvnSpc = 0;                    // the wavenumber spacing between
  while( !file.eof() ){
    ++npts_;
    file >> rdata;                      // read the next entry
    absCoeff_.push_back( rdata.absCoeff );
    wvnSpc = rdata.waveNumber - wvn;
    if( std::abs((rdata.waveNumber-wvn)-wvnSpc) > 1e-10 ){
      ostringstream msg;
      msg << __FILE__ << " : " << __LINE__
          << "\nInconsistency in wavenumber spacing! " << wvnSpc << "\n";
      throw runtime_error( msg.str() );
    }
    wvn = rdata.waveNumber;             // update the previous wavenumber
  }
  hiWaveNum_  = rdata.waveNumber;       // the highest wavenumber in the dataset
  waveNumInc_ = (hiWaveNum_-loWaveNum_)/double(npts_-1);
  file.close();
}

double
SpeciesAbsCoeff::planck_abs_coeff() const
{
  std::vector<double> plnckfunc1(npts_,0.0);
//  std::vector<double> plnckfunc2(npts_,0.0);  // use this for consistency checking
  const double sigma = 5.670373e-12;    // Stefanâ Boltzmann constant (J/(cm^2 s K^4))

  for( size_t i=0; i<npts_; ++i ){
    // compute the integrand, which is I * kappa -- see equation 2.21 in Lyubima's thesis
    plnckfunc1[i] = planck_blackbody_intensity( loWaveNum_ + i*waveNumInc_, temperature_ ) * absCoeff_[i];
    //    plnckfunc2[i] = planck_blackbody_intensity(wvnmB_+i*wvnmst_,myTref);
  }

  const double integral = trapz( waveNumInc_, plnckfunc1 );
  const double tt = integral*PI/( sigma * pow(temperature_,4.0) );
//  const double tt2 = trapz(wvnmst_,plnckfunc1)/trapz(wvnmst_,plnckfunc2);
////  std::cout << " original: "<<  tt
////            << " , ratio: " << tt2 << std::endl;
//  if( std::abs(tt-tt2)/tt2 > 0.1 )
//    std::cout << "\tT=" << myTref << ", " << std::abs(tt-tt2)/tt2 << std::endl;

  return tt;
}

double
SpeciesAbsCoeff::rosseland_abs_coeff() const
{
  const double sigma = 5.67e-12;   // Stefanâ Boltzmann constant (J/(cm^2 s K^4))
  const double h = 6.626070e-34;   // Planck constant (J s);
  const double c = 2.997925e10;    // Speed of light in vacuum (cm/s);
  const double k = 1.380658e-23;   // Boltzmann constant (J/K);
  const double tmp = 2.0*c*c*c*h*h/(k*temperature_*temperature_);

  std::vector<double> rossfunc1(npts_,0.0);
//  std::vector<double> rossfunc2(N_,0.0);  // use this for consistency checking
  for( int i=0; i<npts_; ++i ){
    if( absCoeff_[i] > 1e-16 ){
      /* Matlab commands to get dI/dT:
          syms h c k T v;
          I = 2*h*c^2*v^3/(exp(h*c*v/k/T)-1);
          pretty(I)
          pretty(diff(I,T))
       */
      const double wvn = (loWaveNum_+i*waveNumInc_);
      const double expTerm = exp(h*c*wvn/(k*temperature_));
      rossfunc1[i] = tmp * pow(wvn,4.0) * expTerm/pow(expTerm-1.0,2.0) / absCoeff_[i];
//      rossfunc2[i] = tmp * pow(wvn,4.0) * expTerm/pow(expTerm-1.0,2.0);
    }
  }

  const double integral = trapz( waveNumInc_,rossfunc1 ) * PI / (4.0*sigma*pow(temperature_,3.0) );
  return 1.0/integral;
//  const double t1 = trapz(wvnmst_,rossfunc1);
//  const double t2 = trapz(wvnmst_,rossfunc2);
//  std::cout << t1 <<" , " << t2 << " , " << t2/t1 << ", " << 4.0*sigma*pow(myTref,3.0)/(PI*t1) << "\n";
////  return trapz(wvnmst_,rossfunc2)/trapz(wvnmst_,rossfunc1);
//  return 4.0*sigma*pow(myTref,3.0)/(PI*t1);
}

double
SpeciesAbsCoeff::effective_abs_coeff( const double opl ) const
{
  std::vector<double> plnckfunc1(npts_,0.0);
  std::vector<double> plnckfunc2(npts_,0.0);
  for( int i=0; i<npts_; i++ ){
    const double tmp = planck_blackbody_intensity(loWaveNum_+i*waveNumInc_,temperature_)*exp(-absCoeff_[i]*opl);
    plnckfunc1[i] = tmp * absCoeff_[i];
    plnckfunc2[i] = tmp;
  }
  return trapz(waveNumInc_,plnckfunc1) / trapz(waveNumInc_,plnckfunc2);
}

//==============================================================================

void
SpeciesGData::read( std::istream& file )
{
  data.clear();
  size_t ntemp;
  file >> speciesName >> ntemp;
  for( size_t j=0; j<ntemp; ++j){
    SpeciesGDataSingleTemperature d;
    d.read(file);
    data.push_back(d);
  }
}

void
SpeciesGData::write( std::ostream& file ) const
{
  file << speciesName << std::endl;
  file << data.size() << std::endl;
  for( size_t j=0; j<data.size(); ++j ){
    data[j].write(file);
  }
}

void
SpeciesGDataSingleTemperature::read( std::istream& file )
{
  size_t ng;
  file >> temperature >> ng;
  gvalues.resize(ng,0.0);
  kvalues.resize(ng,0.0);
  for( size_t i=0; i<ng; ++i ){
    file >> gvalues[i] >> kvalues[i];
  }
}

void
SpeciesGDataSingleTemperature::write( std::ostream& file ) const
{
  file << std::setprecision( std::numeric_limits<double>::digits10 );
  file << temperature << std::endl
       << gvalues.size() << std::endl;
  for( int i=0; i<gvalues.size(); ++i ){
    file << gvalues[i] << " " << kvalues[i]<<endl;
  }
}

SpeciesGDataSingleTemperature
calculate_plank( const SpeciesAbsCoeff& absCoeffMix,
                 const double myTref )
{
  SpeciesGDataSingleTemperature gandk;

  const std::vector<double>& absCoef = absCoeffMix.get_coeffs();
  const size_t nwvnm = absCoef.size();

  const double kmin = *std::min_element( absCoef.begin(), absCoef.end() );
  const double kmax = *std::max_element( absCoef.begin(), absCoef.end() );

  const double pwr = 0.1;
  const double pwrk_min = std::pow(kmin,pwr);
  const double pwrk_max = std::pow(kmax,pwr);

  const int n_pwrk=5;  //Number of g-values

  std::vector<double> pwrk( n_pwrk,   0.0 );
  std::vector<double> ff  ( n_pwrk+1, 0.0 );
  std::vector<double> gg  ( n_pwrk+1, 0.0 );
  std::vector<double>  k  ( n_pwrk+1, 0.0 );

  double sum00 = 0;
  for( size_t j=0; j<n_pwrk; ++j ){
    const double pwrk_step = (pwrk_max - pwrk_min)*(log(j+2)-log(j+1)) / log(n_pwrk+2);
    sum00 += pwrk_step;
    pwrk[j] = sum00+kmin-pwrk_step/2;
    k[j] = std::pow( pwrk[j], 1.0/pwr );
  }

  const double hhh = 6.626076e-34;  // Planck constant (Js);
  const double ccc = 2.997925e10;   // Speed of light in vacuum (cm/s)
  const double kkk = 1.380658e-23;  // Boltzmann constant (J/K);
  const double hck = hhh*ccc/kkk;

  const double c1      = 3.7419e-12; // First radiation constant (W cm^2)
  const double sigma   = 5.67e-12;   // Stefan Boltzmann constant (J/(cm^2 s K^4))

  for( size_t i=0; i<nwvnm; ++i ){
    const double wvnm_b = absCoeffMix.min_wavenumber();    // Min wavenumber (1/cm)
    const double wvnm_e = absCoeffMix.max_wavenumber();    // Max wavenumber (1/cm)
    const double wvnmst = absCoeffMix.wavenumber_step();   // wavenumber step (1/cm)
    const double c1sigt4 = c1/(sigma*std::pow(myTref,4))*wvnmst;
    const double eb = c1sigt4 * std::pow((wvnm_b+wvnmst*i),3) / ( exp(hck/myTref*(wvnm_b+wvnmst*i)) - 1.0 );
    const double kpwri = pow( absCoef[i], pwr );
    vector<double>::const_iterator up = std::upper_bound( pwrk.begin(), pwrk.end(), kpwri );
    const size_t iadd = up - pwrk.begin();
    ff[iadd] += eb;
  }
  double fh=0;
  for( size_t l=0; l<n_pwrk+1; ++l ){
    fh += ff[l];
    gg[l] = fh;
  }

  // jcs why are the temporary arrays of length n_pwrk+1 but the resultant array is length n_pwrk ???
  gandk.temperature=myTref;
  for( size_t i=0; i<n_pwrk-1; ++i ){
    gandk.gvalues.push_back(gg[i]);
    gandk.kvalues.push_back( k[i]);
  }

  gandk.gvalues.push_back(1);
  gandk.kvalues.push_back(kmax);

  return gandk;
}

#ifdef RadProps_ENABLE_PREPROCESSOR

// sorts "data" and populates the vector of indices that indicate the sort pattern
template<typename T>
void
paired_sort( vector<size_t> & index,
             vector<T> & data )
{
  // A vector of a pair which will contain the sorted value and its index in the original array
  vector<pair<T,size_t> > indexedPair;
  indexedPair.resize(data.size());
  for( size_t i=0; i<indexedPair.size(); ++i ){
    indexedPair[i].first = data[i];
    indexedPair[i].second = i;
  }
  sort( indexedPair.begin(), indexedPair.end() );
  index.resize( data.size() );
  for( size_t i = 0; i < index.size(); ++i ){
    index[i] = indexedPair[i].second;
    data [i] = indexedPair[i].first;
  }
}

/**
 *
 * @param specNam the name of the species to load files for
 * @param path the path to search
 * @param tempVec the sorted vector of temperatures (output)
 * @param sortFname the sorted vector of file names (output)
 */
void
get_files_sorted_by_temperature( const string specNam, const string path,
                                 vector<double>& tempVec,
                                 vector<string>& sortFname )
{
  const string firstPart( "AbsCoeff" + specNam + "T" );
  const string  lastPart( "txt" );
  const boost::regex filter( firstPart + ".*" + lastPart );  // the ".*" is a wildcard matching operation.

  vector<string> fnames;
  try{
    boost::filesystem::directory_iterator end;
    for( boost::filesystem::directory_iterator i(path); i!=end; ++i ){

      if( !boost::filesystem::is_regular_file( i->status() ) ) continue; // only look at regular files (not dirs)

      const std::string fname = i->path().filename().string();

      // Skip if no match
      boost::smatch what;
      if( !boost::regex_match( fname, what, filter ) ) continue;
      // grab off the temperature for this file
      const string tmp = fname.substr( firstPart.size(), fname.size()-firstPart.size()-lastPart.size() );
      const double temp = boost::lexical_cast<double>(tmp);

      // File matches. parse it
      tempVec.push_back(temp);
      const string p( path+"/"+fname );
      fnames.push_back( p );
      assert( boost::filesystem::is_regular_file(p) );
    }
  }
  catch( std::exception& err ){
    std::ostringstream msg;
    msg << "ERROR loading files.  Details follow:\n" << err.what();
    throw std::runtime_error( msg.str() );
  }

  // sort by temperature
  vector<size_t> myIndex;
  paired_sort( myIndex, tempVec );

  sortFname.clear();
  for( size_t i=0; i<myIndex.size(); ++i ){
    sortFname.push_back( fnames[ myIndex[i] ] );
  }
}


SpeciesAbsData
load_species_abs_coefs( const string& mySp,
                        const string path="." )
{
  SpeciesAbsData specData;
  specData.speciesName = mySp;
  std::vector<SpeciesAbsCoeff> speciesAbsCoeffTempVector;
  std::vector<double>& tempVec = specData.temperatures;
  std::vector<SpeciesAbsCoeff>& specAbsCoefs = specData.absCoeff;

  vector<string> fnames;
  get_files_sorted_by_temperature( mySp, path, tempVec, fnames );

  for( size_t i=0; i<fnames.size(); ++i ){
    cout << "\t-> loading " << fnames[i] << endl;
    specAbsCoefs.push_back( SpeciesAbsCoeff(fnames[i],tempVec[i]) );
  }
  return specData;
}


FSK::FSK( const std::vector<RadiativeSpecies>& mySpecies,
          const string outputFileName,
          const string path )
  : allowClipping_( true )
{
  speciesOrder_ = mySpecies;
  std::vector<std::string> spnam;
  for( size_t j=0; j<mySpecies.size(); ++j ){
    spnam.push_back( species_name(mySpecies[j]) );
    cout << "Species = " << spnam[j] << endl;
  }

  const size_t nspecies=spnam.size();
  for( size_t isp=0; isp<nspecies; ++isp ){

    cout << "loading data for species " << spnam[isp] << endl;
    const SpeciesAbsData specAbsData = load_species_abs_coefs( spnam[isp], path );

    SpeciesGData gSpIV;
    gSpIV.speciesName = spnam[isp];
    const size_t ntemp = specAbsData.temperatures.size();
    std::vector<SpeciesGDataSingleTemperature>& calFSK = gSpIV.data;
    for( size_t itemp=0; itemp<ntemp; ++itemp ){
      const SpeciesAbsCoeff& sac = specAbsData.absCoeff[itemp];
      calFSK.push_back( calculate_plank( specAbsData.absCoeff[itemp],
                                         specAbsData.temperatures[itemp] ) );
    }
//    cout << endl << "G data for " << spnam[isp] << endl;
//    gSpIV.write(cout);
//    cout << endl << endl;
    spCalFSK_.push_back(gSpIV);
  }

  ofstream gfile( outputFileName.c_str() );
  gfile << nspecies << endl;
  for( size_t ll=0; ll<nspecies; ++ll ){
    spCalFSK_[ll].write(gfile);
  }
  gfile.close();
  cout << endl << "Processed FSK data has been written to " << outputFileName << endl << endl;
}
#endif // RadProps_ENABLE_PREPROCESSOR

FSK::FSK( const string fileN )
  : allowClipping_( true )
{
  cout << "Loading FSK Radiation data file: " << fileN << endl;

  std::ifstream fileM2( fileN.c_str(), std::ios::in ); // opens as ASCII!
  if( !fileM2.good() ){
    std::ostringstream msg;
    msg << "ERROR! Could not open file '" << fileN << "'" << std::endl
        << __FILE__ << " : " << __LINE__;
    throw std::runtime_error( msg.str() );
  }

  int nspecies;
  fileM2 >> nspecies;
  for( size_t isp=0; isp<nspecies; ++isp ){
    cout << "loading " << isp+1 << " of " << nspecies << endl;
    SpeciesGData spGData;
    spGData.read( fileM2 );
    spCalFSK_.push_back( spGData );
    speciesOrder_.push_back(species_enum( spCalFSK_[isp].speciesName ));
  }
  fileM2.close();

  // echo file information back out to disk
//  ofstream gfile("TestGs2.txt");
//  gfile << nspecies << endl;
//  for (int ll=0; ll<nspecies; ll++) {
//    spCalFSK_[ll].write(gfile);
//  }
//  gfile.close();

}

double
FSK::mixture_abs_coeff( const std::vector<double>& mixMoleFrac,
                        const double mixT,
                        const double gp ) const
{
  std::vector<double> mixG;
  std::vector<double> mixK;
  mixG.clear();
  mixK.clear();

  mixture_coeffs( mixG, mixK, mixMoleFrac, mixT );

  // clip if we exceed bounds.
  if( gp <= mixG.front() ) return mixK.front();
  if( gp >= mixG.back()  ) return mixK.back();

  const size_t indexg = detail::index_finder<size_t,double>( gp, mixG, allowClipping_ );
  return mixK[indexg]+(mixK[indexg+1]-mixK[indexg])*(gp-mixG[indexg])/(mixG[indexg+1]-mixG[indexg]);
}

void
FSK::mixture_coeffs( std::vector<double>& gmix,
                     std::vector<double>& kmix,
                     const std::vector<double>& mixMoleFrac,
                     const double mixT ) const
{
  const size_t nspecies = mixMoleFrac.size();
  const size_t ng = spCalFSK_[0].data[0].gvalues.size();
  gmix.resize(ng,1.0);
  kmix.resize(ng,0.0);

  for( size_t isp=0; isp<nspecies; ++isp ){

    const SpeciesGData& spData = spCalFSK_[isp];

    const size_t ntemp = spData.data.size();
    std::vector<double> tvec(ntemp,0.0);  // jcs this is slow - need to fix it...
    for( size_t j=0; j<ntemp; ++j ){
      tvec[j] = spData.data[j].temperature;
    }

    const size_t tindex = detail::index_finder<size_t,double>( mixT, tvec, allowClipping_ );

    const vector<double>& g     = spData.data[tindex  ].gvalues;
    const vector<double>& gPlus = spData.data[tindex+1].gvalues;
    const vector<double>& k     = spData.data[tindex  ].kvalues;
    const vector<double>& kPlus = spData.data[tindex+1].kvalues;
    const double T              = spData.data[tindex  ].temperature;
    const double Tplus          = spData.data[tindex+1].temperature;

    for( size_t ig=0; ig<ng; ++ig ){
      gmix[ig] *= std::abs( g[ig] + ( gPlus[ig]-g[ig] )*( mixT-T )/( Tplus-T ) );
      kmix[ig] += mixMoleFrac[isp]/nspecies*sqrt(k[ig]*kPlus[ig]);
    }
  }
}

void
FSK::a_function( std::vector<double>& a,
                 const std::vector<double>& mixMoleFrac,
                 const double Tmed,
                 const double Twall ) const
{
  std::vector<double> mediumG, mediumK, wallG, wallK;

  mixture_coeffs(mediumG,mediumK,mixMoleFrac, Tmed);
  mixture_coeffs(wallG,wallK,mixMoleFrac, Twall);
  const size_t L = mediumG.size();

  a.clear();
  a.resize(L,0.0);

  a[0]=((wallG[1]-wallG[0])/(mediumG[1]-mediumG[0]));
  for( size_t l=1; l<mediumG.size()-1; l++ ){
    a[l]=(wallG[l+1]-wallG[l-1])/(mediumG[l+1]-mediumG[l-1]);
  }
  a[L-1]=(wallG[L-1]-wallG[L-2])/(mediumG[L-1]-mediumG[L-2]);
}


//==============================================================================

#ifdef RadProps_ENABLE_PREPROCESSOR
GreyGas::GreyGas( const std::vector<RadiativeSpecies>& mySpecies,
                  const double opl,
                  const std::string outputFileName,
                  const string path )
 : allowClipping_( true ),
   opl_( opl ),
   nspecies_( mySpecies.size() )
{
  speciesOrder_=mySpecies;
  std::vector<std::string> speciesNames;
  for( int j=0; j<mySpecies.size(); j++ ){
    speciesNames.push_back( species_name(mySpecies[j]) );
  }

  for( size_t isp=0; isp<nspecies_; ++isp ){
    cout << "Species = " << speciesNames[isp] << endl;
    const SpeciesAbsData specData = load_species_abs_coefs( speciesNames[isp], path );

    // process this species information at each temperature to obtain
    // the mean absorption coefficients
    GreyGasData spData;
    spData.speciesName = speciesNames[isp];
    assert( speciesNames[isp] == specData.speciesName );
    spData.ntemp = specData.temperatures.size();

    for( size_t itemp=0; itemp<spData.ntemp; ++itemp ){
      const SpeciesAbsCoeff& spAbsCoeff = specData.absCoeff[itemp];
      spData.temperatures.push_back( specData.temperatures[itemp] );
      spData.planckCoeff.push_back( spAbsCoeff.planck_abs_coeff()        );
      spData.rossCoeff  .push_back( spAbsCoeff.rosseland_abs_coeff()     );
      spData.effAbsCoeff.push_back( spAbsCoeff.effective_abs_coeff(opl_) );
    }
    data_.push_back(spData);
  }

  std::cout << "Writing preprocessed grey-gas properties to file: " << outputFileName << std::endl;
  ofstream out( outputFileName.c_str() );
  out << nspecies_ << endl << opl_ << endl;
  for( size_t isp=0; isp<nspecies_; ++isp ){
    data_[isp].write(out);
  }
}
#endif // RadProps_ENABLE_PREPROCESSOR

GreyGas::GreyGas( const std::string fileName )
: allowClipping_( true )
{
//  cout << "Loading Radiation data file: " << fileG << endl;

  std::ifstream fileMG( fileName.c_str(), std::ios::in ); // opens as ASCII!
  if( !fileMG.good() ){
    std::ostringstream msg;
    msg << "ERROR! Could not open file '" << fileName << "'" << std::endl
        << __FILE__ << " : " << __LINE__;
    throw std::runtime_error( msg.str() );
  }

  fileMG >> nspecies_ >> opl_;
  for( size_t isp=0; isp<nspecies_; ++isp ){
//    cout << "Loading " << isp+1 << " of " << nspecies << " species." << endl;
    GreyGasData data;
    data.read( fileMG );
    data_.push_back( data );
    speciesOrder_.push_back(species_enum( data_[isp].speciesName ));
  }
  fileMG.close();
}

void
GreyGas::mixture_coeffs( double& planckCff,
                         double& rossCff,
                         double& effCff,
                         const std::vector<double>& mixMoleFrac,
                         const double mixT ) const
{
  if( nspecies_ != mixMoleFrac.size() ){
    ostringstream msg;
    msg << __FILE__ << " : " << __LINE__ << endl << endl
        << "The number of species supplied to 'GreyGas::mixture_coeffs()' is not consistent with the number in the table\n"
        << "  Number in table: " << nspecies_ << endl
        << "  Number supplied: " << mixMoleFrac.size() << endl << endl;
    throw invalid_argument( msg.str() );
  }
  planckCff = 0;
  rossCff   = 0;
  effCff    = 0;

  for( size_t i=0; i<nspecies_; i++ ){

    const GreyGasData& grData = data_[i];

    const size_t tindex = detail::index_finder<size_t,double>( mixT, grData.temperatures, allowClipping_ );

    const double planck     = grData.planckCoeff[tindex  ];
    const double planckPlus = grData.planckCoeff[tindex+1];

    const double ross     = grData.rossCoeff[tindex  ];
    const double rossPlus = grData.rossCoeff[tindex+1];

    const double mean     = grData.effAbsCoeff[tindex  ];
    const double meanPlus = grData.effAbsCoeff[tindex+1];

    const double temp     = grData.temperatures[tindex  ];
    const double tempPlus = grData.temperatures[tindex+1];

    // linear interpolation - no clipping...
    // jcs could we pre-tabulate this?
    const double tFac = (mixT-temp)/(tempPlus-temp);
    planckCff += mixMoleFrac[i]*( planck+(planckPlus-planck)*tFac );
    rossCff   += mixMoleFrac[i]*( ross  +(rossPlus  -ross  )*tFac );
    effCff    += mixMoleFrac[i]*( mean  +(meanPlus  -mean  )*tFac );
  }
}

void
GreyGas::GreyGasData::read( std::ifstream& file )
{
  file >> speciesName;
  file >> ntemp;
  planckCoeff.resize(ntemp,0.0);
  rossCoeff.resize(ntemp,0.0);
  effAbsCoeff.resize(ntemp,0.0);
  temperatures.resize(ntemp,0.0);
  for( int i=0; i<ntemp; ++i ){
    file >> temperatures[i] >> planckCoeff[i] >> rossCoeff[i] >> effAbsCoeff[i];
  }
}

void
GreyGas::GreyGasData::write( std::ofstream& file ) const
{
  file << speciesName << std::endl;
  file << ntemp << std::endl;
  for( int ll=0; ll<ntemp; ++ll ){
    file << temperatures[ll] << "  "<< planckCoeff[ll] << "  " << rossCoeff[ll] << "  " << effAbsCoeff[ll] << endl;
  }
}

