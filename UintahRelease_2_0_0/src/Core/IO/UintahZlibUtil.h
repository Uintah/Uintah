/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
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


#include <zlib.h>
#include <sstream>
#include <string>

namespace Uintah {

// The following functions read from a gzipped file (or even a
// non-compressed file) one token of the requested type.  A token is
// defined as a contiguous group of characters separated by white
// space (tabs, spaces, new lines, etc.)

  std::string getString( gzFile & gzFp );
  std::string getString( std::stringstream &file_stream );

  double      getDouble( gzFile & gzFp );
  double      getDouble( std::stringstream &file_stream );

  int         getInt(    gzFile & gzFp );
  int         getInt( std::stringstream &file_stream );

  // Returns a full line... If the line is a comment it is skipped and
  // the next line is returned.  If the line is empty, an empty string
  // is returned.
  //
  std::string getLine(            gzFile & gzFp );
  std::string getLine( std::stringstream & file_stream );


  // Returns the number of bytes in the uncopmressed data.  If an error
  // occurs, then an exception is raised.
  //
  int gzipInflate( const std::string & filename,
                         std::string & uncomp_table_contents );

} // end namespace Uintah
