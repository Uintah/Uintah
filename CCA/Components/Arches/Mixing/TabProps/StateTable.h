/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef StateTable_h
#define StateTable_h

#include <vector>
#include <map>
#include <string>
#include <stdexcept>

#include <CCA/Components/Arches/Mixing/TabProps/BSpline.h>

#include <sci_defs/hdf5_defs.h>

namespace Uintah {

/**
 *  @class  StateTable
 *  @author James C. Sutherland
 *  @date   December, 2005
 *
 *  @brief Support for B-Splined tables, with I/O in HDF5 format.
 *
 *  The table is structured, with arbitrary order interpolation provided by the
 *  underlying B-Spline representation of the data.  In the case where an
 *  unstructured dataset is available, it must be approximated (rather than
 *  interpolated) on a structured table.
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

  typedef const BSpline* TableEntry;
  typedef std::map<std::string,TableEntry> Table;

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
  StateTable(){};

  ~StateTable();

  /** Name the table.  This name is not written to HDF5 database. */
  void name( const std::string & nam ){ tableName_ = nam; }
  
  const std::string & name() const{ return tableName_; }

  /**
   *  Add the given entry to the table.
   *
   *  @param name : name of this spline.  This should be a unique name since it
   *  is used as an identifier in the table.  Duplicate names will generate
   *  errors.
   *
   *  @param spline : the BSpline object
   *
   *  @param indepVarNames : Names of the independent variables.  These names
   *  must match identically for each entry in the table.  Inconsistent names
   *  result in errors thrown.
   *
   *  @param createCopy : OPTIONAL argument.  If true (default), then a copy of
   *  spline will be created and added to the table.  If false, then the spline
   *  will be directly added.  If you use this option, be certain that spline is
   *  not destroyed!  Ownership of the spline is transfered to the table!
   */
  void add_entry( const std::string & name,
		  const BSpline * const spline,
		  const std::vector<std::string> & indepVarNames,
		  const bool createCopy = true );

  /**
   *  Find a spline with the given name.  If no spline corresponding to the
   *  given name is found in the table, a NULL pointer is returned.
   *
   *  @param name : name of the desired spline
   *  @return     : a pointer to the corresponding spline
   */
  const BSpline * find_entry( const std::string & name ) const;

  /**
   *  Provides an interface to the spline evaluation.
   *
   *  @param sp : The spline to query
   *  @param indepVars : The independent variables vector
   *  @result : the value of this spline at the point given by indepVars.
   */
  inline double query( const BSpline * const sp,
		       const double * indepVars ) const{
    if( sp == NULL )
      throw std::runtime_error("ERROR: NULL pointer to BSpline in StateTable::query()");
    return sp->value( indepVars );
  }

  /**
   *  Obtain a dependent variable given the name of the spline.  This requires a
   *  search to obtain the appropriate spline given its name and is thus slower
   *  than a direct query.
   *
   *  @param name : The string name of the spline to query
   *  @param indepVars : the vector containing the independent variables.
   *                     Note that they must be ordered consistently with
   *                     the ordering used when the spline was built.  No
   *                     consistency checking is done.
   *  @return : The value of the spline of the given name at the point given
   *            by indepVars.
   */
  inline double query( const std::string & name,
		       const double* indepVars ) const{
    return query( find_entry(name), indepVars );
  }

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

  /** @brief get the number of independent variables in the table */
  int get_ndim() const{ return numDim_; }

  // obtain iterators for the begin/end of the table
  Table::const_iterator begin(){ return splineTbl_.begin(); }
  Table::const_iterator end(){ return splineTbl_.end(); }

  /**
   *  Read table from a file with the given prefix name.  The file suffix is
   *  "h5" by convention.  The user only has control over the prefix, so that
   *  the file name is "<prefix>.h5"
   *
   *  @param fileNamePrefix : The hdf5 database file prefix.  The suffix is assumed to be
   *  "h5"
   *
   *  @param inputGroupName : OPTIONAL argument.  By default, a group with the same name
   *  as the file prefix will be opened.  If provided, a group by this name will be queried.
   */
  void read_hdf5( const std::string & fileNamePrefix,
		  std::string inputGroupName = "" );

  /** read table from a particular group within an already-open file */
  void read_hdf5( const hid_t & group );

  /**
   *  write table to a file with the given prefix name.  The file suffix is
   *  automatically concatenated as ".h5" so that the file name is given as
   *  "<prefix>.h5"
   */
  void write_hdf5( const std::string & fileNamePrefix = "stateTable" );

  /** write table to a particular group within an already-open file */
  void write_hdf5( const hid_t & group );

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

  std::string tableName_;
  
  // number of independent variables for all entries in the table
  int numDim_;

  Table splineTbl_;

  // names of the independent variables, indexed
  // consistently with the table entries.
  std::vector<std::string> indepVarNames_;
  std::vector<std::string> depVarNames_;

};

} // end namespace Uintah

#endif
