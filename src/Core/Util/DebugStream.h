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

// DebugStream.h - An ostream used for debug messages
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
//
// DebugStream is an ostream that is useful for outputting debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active(identified by its name), and if so where to send the output.
// The syntax for the environment variable is:
//
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
//
// The + or - specifies whether the named object is on or off.  If a file is
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cerr.  The : and , characters are
// restricted to deliminators.
//
// Note: The 'name' is not case-sensitive.  Eg: GeometryPiece matches GEOMETRYPIECE.
//
// Example:
//   SCI_DEBUG = modules.meshgen.warning:+meshgen.out,util.debugstream.error:-
//
// Future Additions:
// o Possible additions to constructor:
//   - Default file to output to
//   - Mode that the file will be opened in (append, out, etc.) (usefulness??)
// o Allow DEFAULT specification in the env variable which would override
//     all default settings. (usefulness??)
// o Time stamp option
//
// Annoyances:
// o Because the list of environment variables, "environ", is built at
//   run time, and getenv queries that list, I have not been able to
//   figure out a way to requery the environment variables during
//   execution.  

#ifndef CORE_UTIL_DEBUGSTREAM_H
#define CORE_UTIL_DEBUGSTREAM_H

#include <map>
#include <string>
#include <iostream>
#include <iomanip>

namespace Uintah {

class DebugStream;
class DebugBuf;

///////////////////
// Class DebugBuf
//
// For use with DebugStream.  This class overrides the overflow operator.
// Each time overflow is called it checks to see where to direct the output to.
class DebugBuf : public std::streambuf
{

private:

public:

  DebugBuf();
  ~DebugBuf();

  int overflow( int ch );

  // Points the the DebugStream that instantiated me.
  DebugStream * m_owner{nullptr};
};

///////////////////
// A general purpose debugging ostream.
class DebugStream : public std::ostream
{

public:
  DebugStream();

  DebugStream(const std::string& name, bool defaulton = true);

  DebugStream( const std::string& name
             , const std::string& component
                   , const std::string& description
                   , bool defaulton = true
                   );

  ~DebugStream();

  std::string getName()        { return m_name; }
  std::string getComponent()   { return m_component; }
  std::string getDescription() { return m_description; }

  std::string getFilename() { return m_filename; }
  void setFilename( const std::string & name ) { m_filename = name; }

  bool active() { return m_active; }
  void setActive( bool active ) { m_active = active; }

  void print() const
  {
    std::cout << std::setw(2)  << std::left << (m_active ? "+" : "-")
              << std::setw(30) << std::left << m_name.c_str()
              << std::setw(75) << std::left << m_description.c_str()
              << std::setw(30) << std::left << m_component.c_str()
              << std::endl;
  }

  static void printAll()
  {
    printf("--------------------------------------------------------------------------------\n");
    for (auto iter = m_all_debug_streams.begin(); iter != m_all_debug_streams.end(); ++iter) {
      (*iter).second->print();
    }
    printf("--------------------------------------------------------------------------------\n\n");
  }

  // The ostream that output should be redirected to. cout by default.
  std::ostream * m_outstream{nullptr};

  static std::map<std::string, DebugStream*> m_all_debug_streams;
  static bool                                m_all_dbg_streams_initialized;


private:

  // Identifies me uniquely.
  std::string m_name;
  std::string m_component;
  std::string m_description;

  // If false, all input is ignored.
  bool m_active;

  // The stream filename.
  std::string m_filename;

  // The buffer that is used for output redirection.
  DebugBuf m_dbgbuf;

  // Check the environment variable.
  void checkEnv();

  // Check for a previous name.
  void checkName();

  void instantiate_map();

};
    
} // namespace Uintah

#endif // CORE_UTIL_DEBUGSTREAM_H
