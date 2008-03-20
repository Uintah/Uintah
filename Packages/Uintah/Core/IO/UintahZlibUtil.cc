#include <Packages/Uintah/Core/IO/UintahZlibUtil.h>

#include <stdio.h>

namespace Uintah {

using namespace std;

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
