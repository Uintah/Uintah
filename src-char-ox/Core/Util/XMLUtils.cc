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


#include <Core/Util/XMLUtils.h>

#include <Core/Util/StringUtil.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Exceptions/ProblemSetupException.h>

#include <sstream>

using namespace std;

void
Uintah::UintahXML::validateType( const string & stringValue, CheckType type )
{
  //__________________________________
  //  Make sure stringValue only contains valid characters
  switch( type ) {
  case UINT_TYPE :
    {
      string validChars(" 0123456789");
      string::size_type  pos = stringValue.find_first_not_of(validChars);
      if( pos != string::npos ){
        ostringstream warn;
        warn << "Bad Integer string: Found '"<< stringValue[pos]
             << "' in the string \""<< stringValue<< "\" at position " << pos << ".\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
    }
    break;
  case INT_TYPE :
    {
      string validChars(" -0123456789");
      string::size_type  pos = stringValue.find_first_not_of(validChars);
      if (pos != string::npos){
        ostringstream warn;
        warn << "Bad Integer string: Found '"<< stringValue[pos]
             << "' in the string \""<< stringValue<< "\" at position " << pos << ".\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
    }
    break;
  case FLOAT_TYPE :
    {
      string validChars(" -+.0123456789eE");
      string::size_type  pos = stringValue.find_first_not_of(validChars);
      if (pos != string::npos){
        ostringstream warn;
        warn << "Bad Float string: Found '"<< stringValue[pos]
             << "' inside of \""<< stringValue << "\" at position " << pos << "\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
      //__________________________________
      // check for two or more "."
      string::size_type p1 = stringValue.find_first_of(".");    
      string::size_type p2 = stringValue.find_last_of(".");     
      if (p1 != p2){
        ostringstream warn;
        warn << "Input file error: I found two (..) "
             << "inside of "<< stringValue << "\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__,true);
      }
    }
    break;
  } // end switch( type )
} 


/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Returns a (collapsed (no begin/ending spaces)) string that holds one line from the flie.
//
string
Uintah::UintahXML::getLine( FILE * fp )
{
  const int   LINE_LENGTH = 4096;

  char        line_buffer[ LINE_LENGTH ];


  while( true ) {

    char      * result = fgets( line_buffer, LINE_LENGTH, fp );

    if( result == nullptr ) {
      return "";
    }

    string line = line_buffer;

    if( line.length() >= (LINE_LENGTH-1) ) {
      // We would have to be smarter about doing this if lines are longer then LINE_LENGTH.
      // This should never happen so just throw an exception if it does.
      throw InternalError( "Grid::getLine() fail - read in line that is too long.", __FILE__, __LINE__ );
    }

    collapse( line );
    if( line != "" ) {
      return line;
    }
  }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////
//
// Used for breaking <tag> value </tag> into three separate strings.
vector<string>
Uintah::UintahXML::splitXMLtag( const string & line )
{
  vector< string > result;

  int numLeftTriBracket = count_substrs( line, "<" );
  int numRightTriBracket = count_substrs( line, ">" );
  if( numRightTriBracket != 2 || numLeftTriBracket != 2 ) {
    
    ostringstream msg;
    msg << "Grid::splitXMLtag(): Number of <>'s is wrong for " << line;

    throw InternalError( msg.str(), __FILE__, __LINE__ );
  }

  int beg_pos = line.find( "<" );
  int end_pos = line.find( ">" );
  
  result.push_back( line.substr( beg_pos, end_pos - beg_pos + 1 ) );

  beg_pos = end_pos+1;
  end_pos = line.find( "<", beg_pos );

  result.push_back( line.substr( beg_pos, end_pos - beg_pos ) );

  beg_pos = end_pos+1;

  result.push_back( line.substr( end_pos ) );

  return result;
}

