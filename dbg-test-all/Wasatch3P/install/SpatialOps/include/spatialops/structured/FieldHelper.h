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

#ifndef SpatialOps_FieldHelper_h
#define SpatialOps_FieldHelper_h

#include<spatialops/structured/MemoryWindow.h>

#include <ostream>
#include <fstream>
#include <string>
#include <cmath>
#include <iomanip>

/** \file FieldHelper.h */

namespace SpatialOps{

/**
 * \brief INTERNAL initialize a field with pseudorandom numbers
 *
 * NOTE: As this function is internally used by other functions, it
 * generally should not be called in user code.
 *
 * \param fi field iterator to use to assign values
 * \param mw memory window of fi (field iterator)
 * \param start double that represents minimum value to assign
 * \param print boolean value, if true prints the values it assigns to standard output
 * \param range maximum value to assign (negation will be minimum value)
 *
 * This function assigns non-integer values to the field described
 * by iterator fi and memory window mw.  The value assigned to a cell
 * with index (x,y,z) is given by:
 * \code
 *   range * sin(start + x + y * xExtent + z * xExtent * yExtent)
 * \endcode
 *
 * Since sine of anything returns values between [-1, 1], the range of values
 * assigned by this function are [-range, range].
 *
 * This function can be used to initialize multiple fields with different
 * pseudorandom numbers by setting start to different values. This function
 * turns each index triplet into a flat index.  For new values, set start to
 * be past the last used flat index. So in general, for the first field, set
 * start to 0.  Assuming all fields are the same size, set start to
 * xExtent * yExtent * zExtent for the second field. For the third field, set
 * start to be twice that value, and so on.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU set to active.)
 */
template<typename Field>
inline void internal_initialize_field( typename Field::iterator fi,
                                       const MemoryWindow mw,
                                       const double start,
                                       const bool print,
                                       const double range )
{
  const int xExtent = mw.extent(0);
  const int yExtent = mw.extent(1);
  const int zExtent = mw.extent(2);

  for( int z = 1; z <= zExtent; ++z ){
    for( int y = 1; y <= yExtent; ++y ){
      for( int x = 1; x <= xExtent; ++x, ++fi ){
        *fi = range * std::sin(start + x + y * xExtent + z * xExtent * yExtent);
        if( print ) std::cout << *fi << " ";
      }
      if( print ) std::cout << std::endl;
    }
    if( print ) std::cout << std::endl;
  }
}

/**
 * \ingroup fields
 * \brief initialize a field (and ghost cells) with pseudorandom numbers
 *
 * \param f field to initialize
 * \param start double that represents minimum value to assign (defaults to 0.0)
 * \param print boolean value, if true prints the values it assigns to standard output (defaults to false)
 * \param range maximum value to assign (negation will be minimum value) (defaults to 1.0)
 *
 * This function assigns non-integer values to the field f, including
 * ghost cells.  The value assigned to a cell with index (x,y,z) is given by:
 * \code
 *   range * sin(start + x + y * xExtent + z * xExtent * yExtent)
 * \endcode
 *
 * Since sine of anything returns values between [-1, 1], the range of values
 * assigned by this function are [-range, range].
 *
 * This function can be used to initialize multiple fields with different
 * pseudorandom numbers by setting start to different values. This function
 * turns each index triplet into a flat index.  For new values, set start to
 * be past the last used flat index. So in general, for the first field, set
 * start to 0.  Assuming all fields are the same size, set start to
 * xExtent * yExtent * zExtent for the second field. For the third field, set
 * start to be twice that value, and so on.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU set to active.)
 */
template<typename Field>
inline void initialize_field( Field & f,
                              const double start = 0.0,
                              const bool print = false,
                              const double range = 1.0 )
{
  internal_initialize_field<Field>( f.begin(), f.window_with_ghost(), start, print, range );
}

/**
 * \ingroup fields
 * \brief initialize a field (without ghost cells) with pseudorandom numbers
 *
 * \param f field to initialize
 * \param start double that represents minimum value to assign (defaults to 0.0)
 * \param print boolean value, if true prints the values it assigns to standard output (defaults to false)
 * \param range maximum value to assign (negation will be minimum value) (defaults to 1.0)
 *
 * This function assigns non-integer values to the field f, NOT including
 * ghost cells.  The value assigned to a cell with index (x,y,z) is given by:
 * \code
 *   range * sin(start + x + y * xExtent + z * xExtent * yExtent)
 * \endcode
 *
 * Since sine of anything returns values between [-1, 1], the range of values
 * assigned by this function are [-range, range].
 *
 * This function can be used to initialize multiple fields with different
 * pseudorandom numbers by setting start to different values. This function
 * turns each index triplet into a flat index.  For new values, set start to
 * be past the last used flat index. So in general, for the first field, set
 * start to 0.  Assuming all fields are the same size, set start to
 * xExtent * yExtent * zExtent for the second field. For the third field, set
 * start to be twice that value, and so on.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU set to active.)
 */
template<typename Field>
inline void interior_initialize_field( Field & f,
                                       const double start = 0.0,
                                       const bool print = false,
                                       const double range = 1.0 )
{
  internal_initialize_field<Field>( f.interior_begin(), f.window_without_ghost(), start, print, range );
}

/**
 * \brief INTERNAL print the values of a field to standard output
 *
 * NOTE: As this function is internally used by other functions, it
 * generally should not be called in user code.
 *
 * \param fi field iterator to use to read values to print
 * \param mw memory window of fi (field iterator)
 * \param os output stream to write to
 * \param addFormat boolean flag to either print with tight format or standard precision
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indicies
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline void internal_print_field( typename Field::const_iterator fi,
                                  const MemoryWindow& mw,
                                  std::ostream& os,
                                  bool addFormat )
{
  const int xExtent = mw.extent(0);
  const int yExtent = mw.extent(1);
  const int zExtent = mw.extent(2);

  for( int z = 1; z <= zExtent; ++z ){
    for( int y = 1; y <= yExtent; ++y ){
      for( int x = 1; x <= xExtent; ++x, ++fi ){
        if( !addFormat ) os << *fi << " ";
        else             os << std::setprecision(2) << *fi << "\t";
      }
      os << std::endl;
    }
    os << std::endl;
  }
}

/**
 * \ingroup fields
 * \brief print the values of a field (and ghost cells) to standard output
 *
 * \param f field to print
 * \param os output stream to write to
 * \param addFormat boolean flag to either print with tight format or standard precision
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indicies
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * For the use in this function, (0,0,0) is the lowest ghost cell, and (I,J,K)
 * is the highest ghost cell.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline void print_field( const Field& f, std::ostream& os, const bool addFormat = false )
{
  if( IS_GPU_INDEX(const_cast<Field&>(f).active_device_index()) ) const_cast<Field&>(f).add_device(CPU_INDEX);
  internal_print_field<Field>(f.begin(), f.window_with_ghost(), os, addFormat );
};

/**
 * \ingroup fields
 * \brief print the values of a field (without ghost cells) to standard output
 *
 * \param f field to print
 * \param os output stream to write to
 * \param addFormat boolean flag to either print with tight format or standard precision
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indicies
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * For the use in this function, (0,0,0) is the lowest interior cell, and
 * (I,J,K) is the highest interior cell. No ghost cells are printed with
 * this function.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline void interior_print_field( const Field& f, std::ostream& os, const bool addFormat = false )
{
  internal_print_field<Field>( f.interior_begin(), f.window_without_ghost(), os, addFormat );
};

/**
 * \brief INTERNAL compare two fields and possibly print values
 *
 * NOTE: As this function is internally used by other functions, it
 * generally should not be called in user code.
 *
 * \param fi1 first field iterator to use to read values
 * \param fi2 second field iterator to use to read values
 * \param mw memory window of fi1 and fi2 (field iterator)
 * \param display boolean value, if true prints comparison of each index
 * \param print boolean value, if true prints values of each index of each iterator
 * \return boolean value, if true fields are equal within given window
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indicies
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * If display is true, prints 0 or 1 for each index. 0 is printed if values
 * in fields are different at that index. 1 is printed if values in fields
 * are the same.
 *
 * If print is true, prints two tabs, then values for the first field in one
 * X-axis row, two more tabs, and then values for the second field in one
 * X-axis row.  Thus, the values of the two fields can be compared visually
 * side-by-side.  Note that the values printed with the print flag are
 * rounded and do NOT show machine-precision values.
 *
 * If both display and print are true, the output of both appear, side-by-side,
 * per X-axis row.
 *
 * If both display and print are false, nothing is printed to standard output.
 *
 * Printing values may be difficult to read if the fields' extents are too large,
 * or the screen used to view them is too small.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline bool internal_display_fields_compare( typename Field::const_iterator fi1,
                                             typename Field::const_iterator fi2,
                                             const MemoryWindow& mw,
                                             const bool display,
                                             const bool print )
{
  bool result = true;
  int xExtent = mw.extent(0);
  int yExtent = mw.extent(1);
  int zExtent = mw.extent(2);

  //copies of iterators for printing
  typename Field::const_iterator cfi1 = fi1;
  typename Field::const_iterator cif2 = fi2;

  // end condition for each test: index < axisExtent && (result || print || display)
  //  this ends the loops early if and only if the result has been found to be false in some cell
  //                                       AND print   == false
  //                                       AND display == false
  for(int z = 0; z < zExtent && (result || print || display); ++z ){
    for(int y = 0; y < yExtent && (result || print || display); ++y ){
      for(int x = 0; x < xExtent && (result || print || display); ++x, ++fi1, ++fi2 ){
        const bool compare = (*fi1 == *fi2);
        result = result && compare;
        if( display ) std::cout << compare << " ";
      }
      if( print ){
        std::cout << "\t\t";
        for( int x = 0; x < xExtent; ++x, ++cfi1 ){
          std::cout << *cfi1 << " ";
        }
        std::cout << "\t\t";
        for( int x = 0; x < xExtent; ++x, ++cif2 ){
          std::cout << *cif2 << " ";
        }
      }
      if( print || display ) std::cout << std::endl;
    }
    if( print || display ) std::cout << std::endl;
  }

  return result;
}

/**
 * \ingroup fields
 * \brief compare two fields and possibly print values (with ghost cells)
 *
 * \param field1 first field to use to read values
 * \param field2 second field to use to read values
 * \param display boolean value, if true prints comparison of each index
 * \param print boolean value, if true prints values of each index of each iterator
 * \return boolean value, if true fields are equal within given window
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indices
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * For the use in this function, (0,0,0) is the lowest ghost cell, and (I,J,K)
 * is the highest ghost cell.
 *
 * If display is true, prints 0 or 1 for each index. 0 is printed if values
 * in fields are different at that index. 1 is printed if values in fields
 * are the same.
 *
 * If print is true, prints two tabs, then values for the first field in one
 * X-axis row, two more tabs, and then values for the second field in one
 * X-axis row.  Thus, the values of the two fields can be compared visually
 * side-by-side.  Note that the values printed with the print flag are
 * rounded and do NOT show machine-precision values.
 *
 * If both display and print are true, the output of both appear, side-by-side,
 * per X-axis row.
 *
 * If both display and print are false, nothing is printed to standard output.
 *
 * Printing values may be difficult to read if the fields' extents are too large,
 * or the screen used to view them is too small.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline bool display_fields_compare(const Field& field1,
                                   const Field& field2,
                                   const bool display = false,
                                   const bool print = false)
{
  return internal_display_fields_compare<Field>( field1.begin(),
                                                 field2.begin(),
                                                 field1.window_with_ghost(),
                                                 display,
                                                 print );
}

/**
 * \ingroup fields
 * \brief compare two fields and possibly print values (without ghost cells)
 *
 * \param field1 first field to use to read values
 * \param field2 second field to use to read values
 * \param display boolean value, if true prints comparison of each index
 * \param print boolean value, if true prints values of each index of each iterator
 * \return boolean value, if true fields are equal within given window
 *
 * This function prints values starting with the lowest index first (0,0,0).
 * The first line contains the X-axis row of values (with Y and Z indicies
 * fixed at 0). The second line contains the X-axis row of values (with Y
 * index of 1, and Z index of 0). A blank line represents the end of one XY
 * plane and the start of the next.
 *
 * Graphically, with X-axis extent of I, Y-axis extent of J, and Z-axis extent
 * of K:
 * \verbatim
    (0,0,0) (1,0,0) ... (I,0,0)
    (0,1,0) (1,1,0) ... (I,1,0)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,0) (1,J,0) ... (I,J,0)

    (0,0,1) (1,0,1) ... (I,0,1)
    (0,1,1) (1,1,1) ... (I,1,1)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,1) (1,J,1) ... (I,J,1)

    (0,0,K) (1,0,K) ... (I,0,K)
    (0,1,K) (1,1,K) ... (I,1,K)
       .       .    .      .
       .       .     .     .
       .       .      .    .
    (0,J,K) (1,J,K) ... (I,J,K)
   \endverbatim
 * For the use in this function, (0,0,0) is the lowest interior cell, and
 * (I,J,K) is the highest interior cell. No ghost cells are printed with
 * this function.
 *
 * If display is true, prints 0 or 1 for each index. 0 is printed if values
 * in fields are different at that index. 1 is printed if values in fields
 * are the same.
 *
 * If print is true, prints two tabs, then values for the first field in one
 * X-axis row, two more tabs, and then values for the second field in one
 * X-axis row.  Thus, the values of the two fields can be compared visually
 * side-by-side.  Note that the values printed with the print flag are
 * rounded and do NOT show machine-precision values.
 *
 * If both display and print are true, the output of both appear, side-by-side,
 * per X-axis row.
 *
 * If both display and print are false, nothing is printed to standard output.
 *
 * Printing values may be difficult to read if the fields' extents are too large,
 * or the screen used to view them is too small.
 *
 * Currently this function only works on CPU-allocated memory and fields.
 * (CPU is at least valid, if not active.)
 */
template<typename Field>
inline bool interior_display_fields_compare( const Field& field1,
                                             const Field& field2,
                                             const bool display = false,
                                             const bool print = false )
{
  return internal_display_fields_compare<Field>( field1.interior_begin(),
                                                 field2.interior_begin(),
                                                 field1.window_without_ghost(),
                                                 display,
                                                 print );
}


/**
 *  \ingroup fields
 *  \fn void write_matlab( const FieldT&, const std::string, const bool)
 *  \brief writes a field to a matlab file
 *  \tparam FieldT the type of SpatialField
 *  \param field the field to write
 *  \param prefix the name of the field
 *  \param includeGhost [false] true to include ghost cells in the file
 */
template<typename FieldT>
void write_matlab( const FieldT& field,
                   const std::string prefix,
                   const bool includeGhost=false )
{
# ifdef ENABLE_CUDA
  // IO only works on CPU.  Ensure that we have a field there.
  const_cast<FieldT&>(field).add_device( CPU_INDEX );
# endif

  const std::string fname = "load_"+prefix+".m";

  std::ofstream fout( fname.c_str() );
  fout << "function [f,n] = load_" << prefix << "()" << std::endl;
  fout << std::scientific;

  const MemoryWindow& mw = includeGhost ? field.window_with_ghost() : field.window_without_ghost();

  typename FieldT::const_iterator fi = includeGhost ? field.begin() : field.interior_begin();
  const typename FieldT::const_iterator fie = includeGhost ? field.end() : field.interior_end();

  const size_t nx = mw.extent(0);
  const size_t ny = mw.extent(1);
  const size_t nz = mw.extent(2);

  fout << "f = [\n";

//    for( ; fi!=fie; ++fi ){
//      fout << *fi << "\n";
//    }
//    fout << "];\n";
//    fout << "f = reshape(f," << mw.extent(0) << "," << mw.extent(1) << "," << mw.extent(2) << ");\n";

  if( nx>1 && ny>1 && nz>1 ){ // 3D
    for( size_t k=1; k<=nz; ++k ){
      if( k>1 ) fout << "f(:,:," << k << ") = [ \n";
      for( size_t j=1; j<=ny; ++j ){
        for( size_t i=1; i<=nx; ++i, ++fi ){
          fout << *fi << " ";
        }
        fout << std::endl;
      }
      fout << "];\n";
    }
    fout << "n=[" << nx << "," << ny << "," <<  nz << "];\n";
  }
  else if( mw.extent(0) > 1 && mw.extent(1) > 1 ){ // 2D XY
    for( size_t j=1; j<=ny; ++j ){
      for( size_t i=1; i<=nx; ++i, ++fi ){
        fout << *fi << " ";
      }
      fout << std::endl << "];\n";
    }
    fout << "];\nn=[" << nx << "," << ny << "];\n";
  }
  else if( mw.extent(0) > 1 && mw.extent(2) > 1 ){ // 2D XZ
    for( size_t j=1; j<=nz; ++j ){
      for( size_t i=1; i<=nx; ++i, ++fi ){
        fout << *fi << " ";
      }
      fout << std::endl;
    }
    fout << "];\nn=[" << nx << "," << nz << "];\n";
  }
  else if( mw.extent(1) > 1 && mw.extent(2) > 1 ){ // 2D YZ
    for( size_t j=1; j<=nz; ++j ){
      for( size_t i=1; i<=ny; ++i, ++fi ){
        fout << *fi << " ";
      }
      fout << std::endl;
    }
    fout << "];\nn=[" << ny << "," << nz << "];\n";
  }
  else if( mw.extent(0) > 1 ){ // 1D X
    for( ; fi!=fie; ++fi ) fout << *fi << " ";
    fout << std::endl << "];\n";
    fout << "n=[" << nx << "];\n";
  }
  else if( mw.extent(1) > 1 ){ // 1D Y
    for( ; fi!=fie; ++fi ) fout << *fi << " ";
    fout << std::endl << "];\n";
    fout << "n=[" << ny << "];\n";
  }
  else if( mw.extent(2) > 1 ){ // 1D Z
    for( ; fi!=fie; ++fi ) fout << *fi << " ";
    fout << std::endl << "];\n";
    fout << "n=[" << nz << "];\n";
  }

  fout.close();
}


} // namespace SpatialOps

#endif
