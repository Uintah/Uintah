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

#ifndef StateTable_h
#define StateTable_h

#include <vector>
#include <map>
#include <string>
#include <stdexcept>

#include <tabprops/TabProps.h>
#include <tabprops/PropertyStash.h>

/**
 *  @class  StateTable
 *  @author James C. Sutherland
 *  @date   December, 2005
 *
 *  @brief Support for tabulated functions
 *
 *  The table is structured, with arbitrary order interpolation provided by the
 *  underlying Lagrange polynomial representation of the data.
 *
 *  In the case where an unstructured dataset is available, it must be
 *  approximated (rather than interpolated) on a structured table.
 *
 *  Clustering the grid is acceptable, but the grid metrics cannot be functions
 *  of any other independent variable.
 *
 *  The reason for this restriction is primarily related to performance.
 *  Multidimensional unstructured interpolation is expensive and complicated.
 *  We want to avoid that mess altogether.
 */

class StateTable{

 public:

  typedef std::map<std::string,InterpT*> Table;

  /**
   *  Construct a StateTable with the specified dimensionality (number of
   *  independent variables).  All entries added to the table must have this
   *  same dimensionality.
   */
  StateTable( const int numDim );

  /**
   * Construct an empty StateTable.  The dimensionality will be determined when
   * the table is loaded from disk or when the first entry is added.  At that
   * point, the table's dimensionality is fixed.
   */
  StateTable();

  ~StateTable();

  /**
   *  Add the given entry to the table.
   *
   *  @param name : name of this interpolant.  This should be a unique name since it
   *  is used as an identifier in the table.  Duplicate names will generate
   *  errors.
   *
   *  @param interp : the interpolant object
   *
   *  @param indepVarNames : Names of the independent variables.  These names
   *  must match identically for each entry in the table.  Inconsistent names
   *  result in errors thrown.
   *
   *  @param createCopy : OPTIONAL argument.  If true (default), then a copy of
   *  interpolant will be created and added to the table.  If false, then the interpolant
   *  will be directly added.  If you use this option, be certain that interpolant is
   *  not destroyed!  Ownership of the interpolant is transfered to the table!
   */
  void add_entry( const std::string & name,
                  InterpT* const interp,
		  const std::vector<std::string> & indepVarNames,
		  const bool createCopy = true );

  /**
   * \brief add_metadata
   * \param name the name (key) to associate with this metadata
   * \param value the value for the metadata.
   *
   * Note that this method can only be used with POD types or with
   * derived types that support serialization (via boost::serialize).
   * Types that do not support serialization will result in errors
   * when the table is written.
   */
  template<typename T>
  inline void add_metadata( const std::string name, const T& value ){
    metaData_.set( name, value );
  }

  /**
   * \brief extract the requested metadata from the table
   * \param name the name (key) for the metadata
   * \result the value of the requested metadata
   *
   * This will throw a std::invalid_argument exception if
   * the requested metadata does not exist.
   */
  template<typename T>
  inline T get_metadata( const std::string name ) const{
    return metaData_.get<T>( name );
  }

  /**
   *  Find an interpolant with the given name.  If no interpolant corresponding to the
   *  given name is found in the table, a NULL pointer is returned.
   *
   *  @param name : name of the desired variable
   *  @return     : a pointer to the corresponding interpolant
   */
  const InterpT* find_entry( const std::string & name ) const;
  InterpT* find_entry( const std::string & name );

  /**
   *  Provides an interface to the interpolant evaluation.
   *
   *  @param interpolant The interpolant to query
   *  @param indepVars The independent variables vector
   *  @result the value of this interpolant at the point given by indepVars.
   */
  double query( const InterpT* const interpolant,
                const double * indepVars ) const;

  /**
   *  Obtain a dependent variable given the name of the interpolant.  This requires a
   *  search to obtain the appropriate interpolant given its name and is thus slower
   *  than a direct query.
   *
   *  @param name : The string name of the variable to query
   *  @param indepVars : the vector containing the independent variables.
   *                     Note that they must be ordered consistently with
   *                     the ordering used when the interpolant was built.  No
   *                     consistency checking is done.
   *  @return : The value of the variable of the given name at the point given
   *            by indepVars.
   */
  double query( const std::string & name,
                const double* indepVars ) const;

  /** Query if the given independent variable exists in the table */
  bool has_indepvar( const std::string & name ) const;

  /** Query if the given dependent variable exists in the table */
  bool has_depvar( const std::string & name ) const{
    return (NULL==find_entry(name)) ? false : true;
  }

  /** Obtain the list of independent variables in the table */
  const std::vector<std::string> & get_indepvar_names() const{
    return indepVarNames_;
  }

  /** Obtain the list of dependent variables in the table */
  const std::vector<std::string> & get_depvar_names() const{
    return depVarNames_;
  }

  /* @brief get the number of independent variables in the table */
  int get_ndim() const{ return numDim_; }

  // obtain iterators for the begin/end of the table
  Table::const_iterator begin() const{ return table_.begin(); }
  Table::const_iterator end() const{ return table_.end(); }

  std::string repo_date() const{ return gitRepoDate_; }
  std::string repo_hash() const{ return gitRepoHash_; }

  /** read the table from the supplied file name */
  void read_table( const std::string fileName );

  /** write the table to the specified file */
  void write_table( const std::string fileName ) const;

  /** write the table to a text file for plotting in tecplot */
  void write_tecplot( const std::vector<int> & npts,
		      const std::vector<double> & upbounds,
		      const std::vector<double> & lobounds,
		      const std::string & fileNamePrefix = "Table" );

  /**
   *  write the table to a matlab file.  This produces a .m file that
   *  can be executed in MATLAB to define the dependent and
   *  independent variables.
   */
  void write_matlab( const std::vector<int>& npts,
		     const std::vector<double>& upbounds,
		     const std::vector<double> & lobounds,
		     const std::vector<bool>& logScale,
		     const std::string & fileNamePrefix = "Table" );

  /**
   *  \brief write information about the table to the output stream.
   *         This includes the source build information, independent
   *         variables, and dependent variables.
   */
  void output_table_info( std::ostream& ) const;

  bool operator==( const StateTable& ) const;

  template<typename Archive> void serialize( Archive& ar, const unsigned int version );

//-------------------------------------------------------------------
//
//                    PRIVATE VARIABLES AND METHODS
//
//--------------------------------------------------------------------

private:

  StateTable( const StateTable & );         // no copying
  StateTable operator=(const StateTable&);  // no assignment

  int getIndex( const int i,
		const std::vector<int>& npts,
		const int ivar );

  const std::string gitRepoDate_, gitRepoHash_;
  std::string tableName_;

  // number of independent variables for all entries in the table
  int numDim_;

  Table table_;

  // names of the independent variables, indexed
  // consistently with the table entries.
  std::vector<std::string> indepVarNames_;
  std::vector<std::string> depVarNames_;

  PropertyStash metaData_;
};

#endif // StateTable_h
