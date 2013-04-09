/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <fstream>
#include <string>

/**
 *  \file 	UintahIFStreamUtil.h
 *  \author Tony Saad
 *  \date  	February, 2013
 *
 *  \brief Provides tools to parse input files using ifstream.
 */

namespace Uintah {
  
  /**
   *  \brief Function that parses the current location of an ifstream input file 
  for a value. The values are assumed to be separated by a space, comma (,), or
   semi colon (;). This function will skip all commented lines, i.e. those lines
   starting with #, %, //, and C-style comments. As an example, assume that your input file looks
   like the following:
   \verbatim
   # my input file
   # dimensions
   10 12 14
   # face direction
   x-
   # values
   1.34 2.55 6.42
   ...
   # end of my input file
   \endverbatim
   
   Then, to parse this input file, you can do the following:
   
   \code{.cpp}
   std::ifstream ifs( inputFileName.c_str() );
   int nx, ny, nz
   getValue(ifs, nx); // nx = 10
   getValue(ifs, ny); // ny = 12
   getValue(ifs, nz); // nz = 14
   
   std::string faceDir;
   getValue(ifs, faceDir); // faceDir = x-
   
   double u, v, w;
   for (int i=0; i<imax; i++) {
     getValue(ifs,u);
     getValue(ifs,v);
     getValue(ifs,w);
     // do something with u, v, and w...
   }
   \endcode
   
   This function is templated on the datatype of the parameter "value".
   *
   *  \param inputFStream A reference to the input file stream that is being parsed.
   *
   *  \param value A reference to the variable that is being filled.
   */
  template <typename Type>  void getValue( std::ifstream &inputFStream, Type& value );
  
  /**
   *  \brief Function that parses the current location of an ifstream input file
   for a string. This uses the templated function getValue and is provided for
   convenience and in case a user wishes to skip a value and not have to store it
   in a local variable.
   */
  const std::string getString( std::ifstream &inputFStream );

  /**
   *  \brief Function that parses the current location of an ifstream input file
   for an integer. This uses the templated function getValue and is provided for
   convenience and in case a user wishes to skip a value and not have to store it
   in a local variable.
   */
  int getInt( std::ifstream &inputFStream );
  
  /**
   *  \brief Function that parses the current location of an ifstream input file
   for a double. This uses the templated function getValue and is provided for
   convenience and in case a user wishes to skip a value and not have to store it
   in a local variable.
   */
  double getDouble( std::ifstream &inputFStream );
  
  /**
   *  \brief Function that returns the entire line at the current inputsteam
   praser location. By default, commented lines are skipped and the next uncommented
   line is returned. Optionally, if skipComments is set to false, the commented
   line is returned.
   */
  const std::string getLine( std::ifstream &inputFStream, bool skipComments=true );
  
} // end namespace Uintah
