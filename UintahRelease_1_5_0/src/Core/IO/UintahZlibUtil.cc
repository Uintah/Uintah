/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <Core/IO/UintahZlibUtil.h>

#include <cstdio>

namespace Uintah {

  using std::string;

//
// Reads a token out of a gzipped file... Tokens are separated by
// white space (spaces, tabs, newlines...).
//
// NOTE: Lines that being with "#" (the very 1st character is a #) are
// considered comments and are skipped.
//
static
const string
getToken( gzFile gzFp )
{
  string token;

  while( true ) {

    char ch = gzgetc( gzFp );

    if( ch == '#' ) { // The input line is commented out.
      while( ch != '\n' ) { // Skip it.
        ch = gzgetc( gzFp );
      }
    }

    if( ch == -1 ) { // end of file reached
      break;
    }

    if( ch == '\n' || ch == '\t' || ch == '\r' || ch == ' ' ) { // done reading token
      if( token.size() > 0 ) {
        break;
      }
      else {
        continue; // until we get a non-empty token.
      }
    }
    token.push_back( ch );
  }
  return token;
}

const string
getString( gzFile gzFp )
{
  return getToken( gzFp );
}

double
getDouble( gzFile gzFp )
{
  double out;
  const string result = getToken( gzFp );
  sscanf( result.c_str(), "%lf", &out );
  return out;
}

int
getInt( gzFile gzFp )
{
  int out;
  const string result = getToken( gzFp );
  sscanf( result.c_str(), "%d", &out );
  return out;
}

const string
getLine( gzFile gzFp )
{
  string line;

  while( true ) {

    char ch = gzgetc( gzFp );

    if( ch == '#' ) { // The input line is commented out.
      while( ch != '\n' ) { // Skip it.
        ch = gzgetc( gzFp );
      }
    }

    if( ch == -1 ) { // end of file reached
      break;
    }

    if( ch == '\n' ) { // done reading line
      break;
    }
    line.push_back( ch );
  }
  return line;
}

} // end namespace Uintah
