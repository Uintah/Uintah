/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/IO/UintahZlibUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/ErrnoException.h>
#include <Core/Exceptions/InvalidState.h>

#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <cstring>

namespace Uintah {

  using std::string;
  using std::ifstream;
  using std::stringstream;

//
// Reads a token out of a gzipped file... Tokens are separated by
// white space (spaces, tabs, newlines...).
//
// NOTE: Lines that being with "#" (the very 1st character is a #) are
// considered comments and are skipped.
//
static
string
getToken( gzFile & gzFp )
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

static
string
getToken( stringstream & stream_file )
{
  string token;

  while( true ) {

    // char ch = gzgetc( gzFp );
    char ch = stream_file.get();

    if( ch == '#' ) { // The input line is commented out.
      while( ch != '\n' ) { // Skip it.
        // ch = gzgetc( gzFp );
        ch = stream_file.get();
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


string
getString( gzFile & gzFp )
{
  return getToken( gzFp );
}

string
getString( stringstream & stream_file )
{
  return getToken( stream_file );
}

double
getDouble( gzFile & gzFp )
{
  double out;
  string result = getToken( gzFp );
  sscanf( result.c_str(), "%lf", &out );
  return out;
}


double
getDouble( stringstream & stream_file )
{
  double out;
  string result = getToken(stream_file );
  sscanf( result.c_str(), "%lf", &out );
  return out;
}

int
getInt( gzFile & gzFp )
{
  int out;
  string result = getToken( gzFp );
  sscanf( result.c_str(), "%d", &out );
  return out;
}

int
getInt( stringstream & stream_file )
{
  int out;
  string result = getToken( stream_file );
  sscanf( result.c_str(), "%d", &out );
  return out;
}

string
getLine( gzFile & gzFp )
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

string
getLine( stringstream & stream_file )
{
  string line;

  while( true ) {

    // char ch = gzgetc( gzFp );
    char ch = stream_file.get();

    if( ch == '#' ) { // The input line is commented out.
      while( ch != '\n' ) { // Skip it.
        //  ch = gzgetc( gzFp );
        ch = stream_file.get();
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


int
gzipInflate( const string & filename,
                   string & uncomp_table_contents )
{

  ifstream table_file( filename.c_str(), std::ios::binary );

  if( !table_file.is_open() ) {
    throw ErrnoException( "Failed to open file: " + filename, errno, __FILE__, __LINE__ );
  }

  stringstream table_buffer;
  table_buffer << table_file.rdbuf();

  string compressedBytes = table_buffer.str();

  table_file.close();

  if ( compressedBytes.size() == 0 ) {  
    throw InvalidState( "File failed to inflate: " + filename, __FILE__, __LINE__ );
  }  
  
  uncomp_table_contents.clear() ;  
  
  unsigned full_length = compressedBytes.size() ;  
  unsigned half_length = compressedBytes.size() / 2;  
  
  unsigned uncompLength = full_length ;  
  char* uncomp = scinew char[uncompLength];
  
  z_stream strm;  
  strm.next_in = (Bytef *) compressedBytes.c_str();  
  strm.avail_in = compressedBytes.size() ;  
  strm.total_out = 0;  
  strm.zalloc = Z_NULL;  
  strm.zfree = Z_NULL;  
  bool done = false ;  
  
  if( inflateInit2(&strm, (16+MAX_WBITS)) != Z_OK) {
    throw InvalidState( "inflateInit2 failed for file: " + filename, __FILE__, __LINE__ );
  }  
  
  while (!done) {  
    // If our output buffer is too small  
    if (strm.total_out >= uncompLength ) {  
      // Increase size of output buffer  
      char* uncomp2 = scinew char[uncompLength + half_length];  
      memcpy( uncomp2, uncomp, uncompLength );
      uncompLength += half_length ;  
      delete[] uncomp ;  
      uncomp = uncomp2 ;  
    }  
  
    strm.next_out = (Bytef *) (uncomp + strm.total_out);  
    strm.avail_out = uncompLength - strm.total_out;  
  
    // Inflate another chunk.  
    int err = inflate (&strm, Z_SYNC_FLUSH);  
    if (err == Z_STREAM_END) done = true;  
    else if (err != Z_OK)  {  
      break;  
    }  
  }  
  
  if (inflateEnd( &strm ) != Z_OK) {  
    throw InvalidState( "inflateEnd failed for file: " + filename, __FILE__, __LINE__ );
  }  
  
  for ( size_t i=0; i<strm.total_out; ++i ) {  
    uncomp_table_contents += uncomp[ i ];  
  }  
  delete [] uncomp ;  

  return uncomp_table_contents.size();
}

} // end namespace Uintah
