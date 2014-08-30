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

#ifndef JCS_FlameLib_h
#define JCS_FlameLib_h

#include <vector>
#include <string>

#include <tabprops/TabProps.h>
#include <tabprops/prepro/TableBuilder.h>
#include <tabprops/prepro/rxnmdl/ReactionModel.h>

namespace Cantera_CXX{
class IdealGasMix;
}

class StateTable; // forward

//-- data structure to hold state variables
static const int nsmax=100;  // maximum number of species...

struct StateEntry
{
  int ns;
  double temp;
  double species[nsmax];   // statically allocated to avoid deep copies...
  double enthalpy;
};

//-- overloaded operator for state variable output
std::ostream& operator << (std::ostream&, const StateEntry&);


//==============================================================================


/**
 *  @class JCSFlamelet
 *  @author James Sutherland
 *  @date July, 2004
 *
 *  Class to hold a single flamelet.
 *  Provides tools to:
 *    \li read the flamelet
 *    \li query the state at a given mixture fraction
 *
 *  Each (steady) flamelet is uniquely parameterized by its maximum dissipation rate.
 *
 *  JCSFlamelet may have arbitrary spacing in f-space.
 */
class JCSFlamelet
{

 public:

  /**
   * @param ns number of species
   * @param file name of the file
   * @param order order of the interpolant
   */
  JCSFlamelet( const int ns,
               const std::string file,
               const int order );
  ~JCSFlamelet();

  //-- access function returning number of points in this flamelet
  int get_n_pts() const { return npts_; }

  //-- access function returning number of species in this flamelet
  int get_n_spec() const { return nspec_;}

  //-- access function returning MAX dissipation rate for this flamelet
  double get_chi_max() const;

  //-- access function returning stoichiometric dissipation rate for this flamelet
  double get_chi_st() const {return chiSt_;}

  const std::vector<double> & get_chi() const{ return chi_; };
  const std::vector<double> & get_mixfr() const{ return mixfr_; };
  const std::vector<StateEntry> & get_state() const{ return flmlt_; };

  //-- Query this flamelet for the state at a given mixture fraction.
  StateEntry & query( const double f );

  const std::vector<std::string> get_spec_names(){ return specName_; };

  bool operator< (const JCSFlamelet& flm) const{ return (chiSt_ <  flm.get_chi_st()); }
  bool operator> (const JCSFlamelet& flm) const{ return (chiSt_ >  flm.get_chi_st()); }
  bool operator<=(const JCSFlamelet& flm) const{ return (chiSt_ <= flm.get_chi_st()); }
  bool operator>=(const JCSFlamelet& flm) const{ return (chiSt_ >= flm.get_chi_st()); }
  bool operator==(const JCSFlamelet& flm) const{ return (chiSt_ == flm.get_chi_st()); }
  bool operator!=(const JCSFlamelet& flm) const{ return (chiSt_ != flm.get_chi_st()); }

  const InterpT * get_tinterp() const{ return Tinterp_; }
  const InterpT * get_hinterp() const{ return Hinterp_; }
  const InterpT * get_yinterp( const int i ) const{ return Yinterp_[i]; }


 private:

  const int nspec_;       ///< number of species
  int npts_;              ///< number of points in this flamelet solution
  const int order_;       ///< number of points in this flamelet solution

  double  chiSt_;            ///< stoichiometric dissipation rate
  std::vector<double> mixfr_; ///< mixture fraction points
  std::vector<double> chi_;   ///< dissipation rate at each point

  std::vector<StateEntry> flmlt_;  ///< the flamelet solution for this chi_o
  StateEntry stateTmp_;  ///< temporary space for internal work

  std::vector<std::string> specName_;  ///< vector of species names

  InterpT * Tinterp_;
  InterpT * Hinterp_;
  std::vector<InterpT*> Yinterp_;

  //-- loads this flamelet from disk
  void load_flamelet( const std::string& );

  //-- extracts file header from flamelet files,
  //   returning number of lines in the header
  int extract_header( const std::string& );

  //-- Allocates memory for this flamelet
  void allocate_memory();

  //-- generate a interp representation of the flamelet
  void generate_interps();

};


//==============================================================================

/**
 *  @author James Sutherland
 *  @date   July 2004
 *
 *  Class to hold a flamelet library.
 *
 *  Provides tools to:
 *    \li load the library
 *    \li query the state at a given mixture fraction and dissipation rate
 *
 *  There is no requirement that the mixture fraction grid be the same for each flamelet
 *  in the library.  However, each flamelet must have the same number of species.
 */

class JCSFlameLib
{
 public:
  /**
   * @param ns number of species
   * @param order interpolant order
   */
  JCSFlameLib( const int ns,
               const int order );

  // destructor
  ~JCSFlameLib();

  /** \brief return the maximum dissipation rate entry in the library */
  double get_chi_max() { double c=chi_o[n_entries-1];  return c;}

  /** \brief return the minimum dissipation rate entry in the library */
  double get_chi_min() { double c=chi_o[1];  return c; }

  /**
   * \brief Query the flamelet library for the state, given the mixture fraction and its dissipation rate
   * \param f the mixture fraction
   * \param chi the dissipation rate
   * \param state struct declared in "flamelets.h"
   */
  void query(const double f, const double chi, StateEntry& state);


  /** \brief generate a state table, writing it to the specified file */
  void generate_table( TableBuilder & table,
                       const std::string & filePrefix,
                       Cantera_CXX::IdealGasMix & gas );

  /**
   * @brief dump the library (with all additional TabProps requested variables)
   *        to a Matlab file. Should be called prior to generate_table.
   * @param prefix the prefix of the file name (suffix will be ".m")
   */
  void request_matlab_output( const std::string prefix = "SLFM_matlab" );

  /**
   * @brief dump the library (with all additional TabProps requested variables)
   *         to a text file. Should be called prior to generate_table.
   * @param prefix the prefix of the file name (suffix will be ".txt")
   */
  void request_text_output( const std::string prefix = "SLFM_text" );

//====================================================================

 private:
  int nspec;            ///< number of species
  int n_entries;        ///< number of entries in flamelet lib (nchi)
  const int order_;

  bool dumpMatlab_, dumpText_;
  std::string matlabPrefixName_, textPrefixName_;

  std::vector<double> chi_o;        ///< array of dissipation rates
  std::vector<JCSFlamelet*> flmlts; ///< flamelet library

  InterpT * Tinterp_;
  InterpT * Hinterp_;
  std::vector<const InterpT*> Yinterp_;
  InterpT * Pinterp_;

  /** \brief loads the flamelet library from disk */
  void read_flamelib();

  void verify_table( const std::string ) const;

  void dump( const StateTable& table ) const;
  void dump_matlab( const StateTable& table ) const;

  // for use in sorting flamelets with std::sort
  class sort_ascending{
  public:
    bool operator()(const JCSFlamelet* f1, const JCSFlamelet* f2) const{
      return ( *f1 < *f2 );
    }
  };

};


//====================================================================


/**
 *  @class  SLFMImporter
 *  @date   April, 2006
 *  @author James C. Sutherland
 *
 *  Facilitate importing flamelet files into the appropriate tabular structure.
 */
class SLFMImporter : public ReactionModel
{
public:

  SLFMImporter( Cantera_CXX::IdealGasMix & gas,
                const int order );
  ~SLFMImporter();

  void implement();

  JCSFlameLib * get_lib(){return m_flmLib;}

private:

  std::vector<std::string> & indep_var_names();
  JCSFlameLib * const m_flmLib;
};

#endif
