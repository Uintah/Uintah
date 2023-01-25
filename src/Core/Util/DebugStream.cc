/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <Core/Exceptions/InternalError.h>

#include <sci_defs/compile_defs.h> // for STATIC_BUILD

#include <algorithm>
#include <cstdlib> // for getenv()
#include <fstream>
#include <iostream>

namespace Uintah {

static const char *ENV_VAR = "SCI_DEBUG";

std::map<std::string, DebugStream*> DebugStream::m_all_debug_streams{};
bool                                DebugStream::m_all_dbg_streams_initialized{false};

DebugBuf::DebugBuf()
{}


DebugBuf::~DebugBuf()
{}

int DebugBuf::overflow( int ch )
{
  if( m_owner == nullptr ) {
    std::cout << "DebugBuf: owner not initialized? Maybe static object init order error.\n";
  }
  else if( m_owner->active() ) {
    return(*(m_owner->m_outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}


DebugStream::DebugStream( const std::string& name
                          , bool defaulton )
  : std::ostream( &m_dbgbuf )
  , m_outstream( &std::cout )
  , m_name( name )
  , m_component( "Unknown" )
  , m_description( "No description" )
  , m_active( defaulton )
  , m_filename( "cout" )
{
#ifdef STATIC_BUILD
  instantiate_map();
#endif

  m_dbgbuf.m_owner = this;
    
  // Check to see if the name has been used before.
  checkName();
  
  // Check SCI_DEBUG to see if this instance is mentioned.
  checkEnv();
}


DebugStream::DebugStream( const std::string & name
                        , const std::string & component
                        , const std::string & description
                        ,       bool          defaulton
                        )
  : std::ostream( &m_dbgbuf )
  , m_outstream( &std::cout )
  , m_name( name )
  , m_component( component )
  , m_description( description )
  , m_active( defaulton )
  , m_filename( "cout" )
{
#ifdef STATIC_BUILD
  instantiate_map();
#endif

  m_dbgbuf.m_owner = this;
    
  // Check to see if the name has been used before.
  checkName();
  
  // Check SCI_DEBUG to see if this instance is mentioned.
  checkEnv();
}


DebugStream::~DebugStream()
{
  if (m_outstream && m_outstream != &std::cerr && m_outstream != &std::cout) {
    delete (m_outstream);
  }
}

void
DebugStream::checkName()
{
  // Comment out this if statement and the SCI_THROW to see all name conflicts.
  if (m_component != "" && m_component != "Unknown") {

    // See if the name has already been registered.
    auto iter = m_all_debug_streams.find(m_name);
    if (iter != m_all_debug_streams.end()) {

      printf("These two debugStreams are for the same component and have the same name. \n");
      (*iter).second->print();
      print();

      // Two debugStreams for the same compent with the same name.
      SCI_THROW(InternalError(std::string("Multiple DebugStreams for component " + m_component + " with name " + m_name), __FILE__, __LINE__));
    }
    else {
      m_all_debug_streams[m_name] = this;
    }
  }
}  


void
DebugStream::checkEnv()
{
  // Set the input name to all lowercase as we are going to be doing case-insensitive checks.
  std::string temp = m_name;
  std::transform(temp.begin(), temp.end(), temp.begin(), ::tolower);
  const std::string name_lower = temp;

  char* vars = getenv(ENV_VAR);
  if (!vars) {
    return;
  }
  std::string var(vars);

  // If SCI_DEBUG was defined, parse the string and store appropriate
  // values in onstreams and offstreams

  if (!var.empty()) {
    std::string name, file;

    unsigned long oldcomma = 0;
    std::string::size_type commapos = var.find(',', 0);
    std::string::size_type colonpos = var.find(':', 0);
    if (commapos == std::string::npos) {
      commapos = var.size();
    }
    while (colonpos != std::string::npos) {
      name.assign(var, oldcomma, colonpos - oldcomma);

      // Doing case-insensitive test... so lower the name.
      std::transform(name.begin(), name.end(), name.begin(), ::tolower);

      if (name == name_lower) {
        file.assign(var, colonpos + 1, commapos - colonpos - 1);
        if (file[0] == '-') {
          m_active = false;
        }
        else if (file[0] == '+') {
          m_active = true;
        }
        else {
          // Houston, we have a problem: SCI_DEBUG was not correctly set. Ignore and set all to default values...
          return;
        }

        // if no output file was specified, set to cout
        if (file.length() == 1) {
          m_filename = std::string("cout");
          m_outstream = &std::cout;
        }
        else if (file.length() > 1) {
          m_filename = file.substr(1, file.size() - 1).c_str();
          m_outstream = new std::ofstream(m_filename);
        }
        return;
      }
      oldcomma = commapos + 1;
      commapos = var.find(',', oldcomma + 1);
      colonpos = var.find(':', oldcomma + 1);
      if (commapos == std::string::npos) {
        commapos = var.size();
      }
    }
  }
  else {
    // SCI_DEBUG was not defined, all objects will be set to default
  }
}

void
DebugStream::instantiate_map()
{
  if (!m_all_dbg_streams_initialized) {
    m_all_dbg_streams_initialized = true;
    m_all_debug_streams = std::map<std::string, DebugStream*>();
  }
}

} // End namespace Uintah
