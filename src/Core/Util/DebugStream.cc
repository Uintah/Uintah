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
// DebugStream.cc - An ostream used for debug messages
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
// DebugStream is an ostream that is useful for outputing debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active, and if so where to send the output.  The syntax for the
// environment variable is:
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
// The + or - specifies whether the named object is on or off.  If a file is 
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cout.  The : and , characters are
// restricted to deliminators.
// Example:
// SCI_DEBUG = modules.meshgen.warning:+meshgen.out,util.debugstream.error:-
// Future Additions:
// o Possible additions to constructor:
//   - Default file to output to
//   - Mode that the file will be opened in (append, out, etc.) (usefulness??)
// o Allow DEFAULT specification in the env variable which would override
//   all default settings. (usefulness??)
// o Time stamp option

#include <Core/Util/DebugStream.h>

#include <algorithm>
#include <cstdlib> // for getenv()
#include <fstream>
#include <iostream>

using namespace std;

namespace Uintah {

static const char *ENV_VAR = "SCI_DEBUG";


DebugBuf::DebugBuf()
{}


DebugBuf::~DebugBuf()
{}

int
DebugBuf::overflow( int ch )
{
  if( m_owner == nullptr ) {
    cout << "DebugBuf: owner not initialized? Maybe static object init order error.\n";
  }
  else if( m_owner->active() ) {
    return(*(m_owner->m_outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}


DebugStream::DebugStream( const string& iname, bool defaulton ) :
    std::ostream( &m_dbgbuf ), m_outstream( 0 )
{
  m_name = iname;
  m_dbgbuf.m_owner = this;
  // set default values
  m_isactive = defaulton;
  m_outstream = &cout;
  m_filename = "cout";
  
  // check SCI_DEBUG to see if this instance is mentioned.
  checkenv( iname );
}


DebugStream::~DebugStream()
{
  if( m_outstream && m_outstream != &cerr && m_outstream != &cout ){
    delete( m_outstream );
  }
}

void
DebugStream::checkenv( const string & iname )
{
  // Set the input name (iname) to all lowercase as we are going to be doing case-insensitive checks.
  string temp = iname;
  transform( temp.begin(), temp.end(), temp.begin(), ::tolower );
  const string iname_lower = temp;

  char* vars = getenv( ENV_VAR );
  if( !vars ) {
     return;
  }
  string var( vars );

  // If SCI_DEBUG was defined, parse the string and store appropriate
  // values in onstreams and offstreams

  if( !var.empty() ){
    string name, file;
    
    unsigned long oldcomma = 0;
    string::size_type commapos = var.find(',', 0);
    string::size_type colonpos = var.find(':', 0);
    if(commapos == string::npos){
      commapos = var.size();
    }
    while( colonpos != string::npos ){
      name.assign( var, oldcomma, colonpos-oldcomma );

      // Doing case-insensitive test... so lower the name.
      transform( name.begin(), name.end(), name.begin(), ::tolower );

      if( name == iname_lower ){
	file.assign(var, colonpos+1, commapos-colonpos-1);
	if( file[0] == '-' ){
	  m_isactive = false;
	}
	else if( file[0] == '+' ){
	  m_isactive = true;
	}
	else{
	  // Houston, we have a problem: SCI_DEBUG was not correctly
	  // set.  Ignore and set all to default values...
	  return;
	}

	// if no output file was specified, set to cout
	if( file.length() == 1 ) {
	  m_filename = std::string( "std::cout" );
	  m_outstream = &cout;
	}
	else if( file.length() > 1 ) {
	  m_filename = file.substr(1, file.size()-1).c_str();
	  m_outstream = new ofstream( m_filename );
	}
	return;
      }
      oldcomma = commapos + 1;
      commapos = var.find( ',', oldcomma + 1 );
      colonpos = var.find( ':', oldcomma + 1 );
      if( commapos == string::npos ) {
	commapos = var.size();
      }
    }
  }
  // else {
  //    SCI_DEBUG was not defined,
  //    all objects will be set to default
  // }
}

} // End namespace Uintah
