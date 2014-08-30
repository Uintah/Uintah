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
#ifndef TabProps_FlameMaster_h
#define TabProps_FlameMaster_h

#include <vector>
#include <string>
#include <map>

#include <tabprops/TabProps.h>
#include <tabprops/StateTable.h>


/**
 * \class FMFlamelet
 * \author James C. Sutherland
 * \date June, 2014
 *
 * \brief Imports a single FlameMaster flamelet data file from disk and
 *        provides methods to interrogate it.
 *
 * Note that this assumes that the independent variable is "Z" and will reorder
 * all fields to ensure that the "Z" variable is increasing (rather than
 * decreasing) as necessary.
 */
class FMFlamelet{

  int npts_; ///< the number of points for each field

  typedef std::map<std::string,std::string> MetaData;
  typedef std::map<std::string,std::vector<double> > VarEntry;

  const std::string fileName_;
  const int interpOrder_;
  MetaData metaData_;
  VarEntry varEntries_;

  VarEntry::const_iterator extract_entry( const std::string& varname ) const;

  void load_file( const std::string& );
  void reorder();

public:
  /**
   * \brief Construct a flamelet by loading a file from disk
   * \param fileName the name of the file to load
   * \param order the interpolant order
   */
  FMFlamelet( const std::string fileName,
              const int order=3 );

  /**
   * \brief Dump the information about this flamelet, including metadata entries and fields stored
   * \param os the output stream to dump information to
   */
  void dump_info( std::ostream& os ) const;

  /**
   * @brief Retrieve an interpolant for the desired quantity
   * @param entry the name of the flamelet field to interpolate (corresponding to the name in the FlameMaster file)
   * @param indepVarName the name of the independent variable for the interpolant (defaults to "Z")
   * @param allowClipping if true (default), then independent variable values outside the supported range will be clipped before interpolating.
   * @return the Interp1D object
   */
  Interp1D interpolant( const std::string& entry,
                        const std::string indepVarName="Z",
                        const bool allowClipping=true ) const;

  /**
   * @return the names of the dependent variables that were stored as fields in the FlameMaster file.
   */
  std::vector<std::string> table_entries() const;

  /**
   * @param varName the field of interest
   * @return the values of the field
   */
  const std::vector<double>& variable_values( const std::string varName ) const;

  /**
   * @param key the name of the metadata entry (header information in the FlameMaster file)
   * @return the value of the entry
   */
  template<typename T> T extract_metadata_value( const std::string key ) const;

};

/**
 * \class FlameMasterLibrary
 * \author James C. Sutherland
 * \date June, 2014
 *
 * \brief Imports a set of FlameMaster files from disk and generates a StateTable object.
 *
 * Currently, all flamelets in the library are interpolated onto the same mixture
 * fraction grid as the first flamelet.  The interpolation in \f$\chi\f$ space
 * is on \f$\chi\f$ itself - not \f$\log_{10}\chi\f$.
 */
class FlameMasterLibrary{

  const int interpOrder_;
  const bool allowClipping_;  // jcs may want to have this option per variable...
  const std::string mixfrac_, dissipRate_, heatLoss_;
  std::vector<FMFlamelet*> library_;
  StateTable table_;
  bool hasGeneratedTable_;

  void load_files( const std::string& path,
                   const std::string& pattern );

public:

  /**
   * \brief Create a FlameMasterLibrary.
   * \param path the directory to look for files in
   * \param filePattern The pattern of files to look for.  Should be in the form "stuff*morestuff*laststuff"
   * \param order the interpolant order (defaults to 3)
   * \param allowClipping flag to allow independent variables to be clipped to
   *  their supported bounds prior to interpolation being performed.
   */
  FlameMasterLibrary( const std::string path,
                      const std::string filePattern,
                      const int order=3,
                      const bool allowClipping=true );

  ~FlameMasterLibrary();

  // jcs note: here we are not using a TableBuilder because that requires Cantera.
  // For now, we will just build a table based on the information from the FlameMaster library itself.
  /**
   * @brief Generate a StateTable and return a handle to it.
   * @param filePrefix the name for the StateTable.
   * @return The StateTable containing the FlameMaster flamelet files.
   */
  StateTable& generate_table( const std::string& filePrefix );

};

#endif
