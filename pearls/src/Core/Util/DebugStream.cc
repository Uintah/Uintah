/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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
#include <iostream>
#include <fstream>

using namespace std;

namespace SCIRun {

static const char *ENV_VAR = "SCI_DEBUG";

DebugBuf::DebugBuf()
{}

//______________________________________________________________________
//
DebugBuf::~DebugBuf()
{}

//______________________________________________________________________
//
int DebugBuf::overflow(int ch)
{
  if (owner==NULL) {
    cout << "DebugBuf: owner not initialized? Maybe static object init order error." << endl;
  }
  else if (owner->active()) {
    return (*(owner->outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}

//______________________________________________________________________
//
DebugStream::DebugStream( const string& iname,
                          bool defaulton )
  : std::ostream(&dbgbuf),
    outstream(0)
{
  dbgbuf.owner = this;

  // Set default value for the current DebugStream
  setActive(defaulton);

  // Parse SCI_DEBUG and assign output streams as needed
  checkenv(iname);
}

//______________________________________________________________________
//
DebugStream::~DebugStream()
{
  if ( outstream && ( outstream != &cerr && outstream != &cout ) ) {
    delete(outstream);
  }
}

//______________________________________________________________________
//
void DebugStream::checkenv( string iname )
{
  char* vars = getenv(ENV_VAR);

  // SCI_DEBUG was not provided
  if (!vars) {

    // Assign output streams for DebugStreams enabled by default
    if (isactive) {
      outstream = &cout;
    }

    // Parsing SCI_DEBUG is not necessary
    return;
  }

  string var(vars);

  // SCI_DEBUG has been provided
  // Parse user input and assign outstreams as needed
  if (!var.empty()) {
    string name, file;

    // Define initial colon and comma positions
    unsigned long oldcomma = 0;
    string::size_type colonpos = var.find(':', 0);
    string::size_type commapos = var.find(',', 0);

    if (commapos == string::npos) {
      commapos = var.size();
    }

    // Iterate over user input until a match has been
    // found or the input has been exhausted
    while (colonpos != string::npos) {
      name.assign(var, oldcomma, colonpos-oldcomma);

      // User input matching the current DebugStream has been found
      if (name == iname) {
        file.assign(var, colonpos+1, commapos-colonpos-1);

        if (file[0] == '-') {
          setActive(false);

          // Further parsing is not necessary
          return;
        }
        else if ( file[0] == '+' ) {
          setActive(true);

          // Assign output streams for explicitly enabled DebugStreams
          if(file.length() == 1){
            outstream = &cout;
          }
          else{
            outstream = new ofstream(file.substr(1, file.size()-1).c_str());
          }

          // Further parsing is not necessary
          return;
        }
      }

      // Update colon and comma positions
      oldcomma = commapos+1;
      colonpos = var.find(':', oldcomma+1);
      commapos = var.find(',', oldcomma+1);

      if (commapos == string::npos) {
        commapos = var.size();
      }
    }

    // Assign output streams for DebugStreams enabled by default
    // that have not been enabled explicitly
    if (isactive) {
      outstream = &cout;
    }
  }
}

} // End namespace SCIRun
