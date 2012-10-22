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

// DebugStream.h - An ostream used for debug messages
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
// DebugStream is an ostream that is useful for outputing debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active(identified by its name), and if so where to send the output.
// The syntax for the environment variable is:
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
// The + or - specifies wheather the named object is on or off.  If a file is 
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cerr.  The : and , characters are
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
// Annoyances:
// o Because the list of environment variables, "environ", is built at
// run time, and getenv queries that list, I have not been able to
// figure out a way to requery the environment variables during
// execution.  

#ifndef SCI_project_DebugStream_h
#define SCI_project_DebugStream_h 1

// temp fix to get pg compilers to resolve symbols
#ifdef __PGI
#define __mbstate_t mbstate_t
#endif

#include <cstdlib> // for getenv()
#include <string>
#include <iostream>


#include <Core/Util/share.h>

namespace SCIRun {

    using std::streambuf;
    using std::ostream;
    using std::string;

    
    class DebugStream;
    class DebugBuf;

    ///////////////////
    // class DebugBuf
    // For use with DebugStream.  This class overrides the overflow
    // operator.  Each time overflow is called it checks to see where
    // to direct the output to. 
    class DebugBuf:public streambuf{
    private:
    public:
      DebugBuf();
      ~DebugBuf();
      // points the the DebugStream that instantiated me
      DebugStream *owner;
      int overflow(int ch);
    };


    ///////////////////
    // class DebugStream
    // A general purpose debugging ostream.
    class SCISHARE DebugStream: public ostream{
    private:
      // identifies me uniquely
      string name;
      // my default action (used if nothing is specified in SCI_DEBUG)
      bool defaulton;
      // the buffer that is used for output redirection
      DebugBuf dbgbuf;
      // if false, all input is ignored
      bool isactive;
      // check the environment variable
      void checkenv(string);
            
    public:
      DebugStream();
      DebugStream(const string& name, bool defaulton = true);
      ~DebugStream();
      bool active() {return isactive;};
      void setActive(bool active) { isactive = active; };
      // the ostream that output should be redirected to. cout by default.
      ostream *outstream;
    };
    
} // End namespace SCIRun

#endif
