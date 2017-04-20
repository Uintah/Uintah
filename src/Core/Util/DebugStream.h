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

#ifndef SCI_project_DebugStream_h
#define SCI_project_DebugStream_h 1

#include <string>
#include <iostream>

namespace Uintah {

class DebugStream;
class DebugBuf;

///////////////////
// Class DebugBuf
//
// For use with DebugStream.  This class overrides the overflow
// operator.  Each time overflow is called it checks to see where
// to direct the output to. 
class DebugBuf : public std::streambuf {
private:
public:
  DebugBuf();
  ~DebugBuf();
  // Points the the DebugStream that instantiated me.
  DebugStream * m_owner;
  int overflow( int ch );
};

///////////////////
// class DebugStream
// A general purpose debugging ostream.
class DebugStream: public std::ostream {
private:
  // Identifies me uniquely.
  std::string m_name;
  // The stream filename.
  std::string m_filename;
  // The buffer that is used for output redirection.
  DebugBuf m_dbgbuf;
  // If false, all input is ignored.
  bool m_isactive;


  // Check the environment variable.
  void checkenv( const std::string & name );
        
public:
  DebugStream();
  DebugStream(const std::string& name, bool defaulton = true);
  ~DebugStream();
  std::string getName() { return m_name; }
  std::string getFilename() { return m_filename; }
  void setFilename( const std::string & name ) { m_filename = name; }
  bool active() { return m_isactive; }
  void setActive( bool active ) { m_isactive = active; }

  // The ostream that output should be redirected to. cout by default.
  std::ostream * m_outstream;
};
    
} // End namespace Uintah

#endif
