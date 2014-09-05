// DebugStream.h - An ostream used for debug messages
//
// Written by:
// Eric Kuehne
// Department of Computer Science
// University of Utah
// Feb. 2000
//
// Copyright (C) 2000 SCI Group
//
// DebugStream is an ostream that is useful for outputing debug messages.
// When an instance is created, it is given a name.  An environment variable,
// SCI_DEBUG, is inspected to see if a particular instance should be
// active(identified by its name), and if so where to send the output.
// The syntax for the environment variable is:
//
// SCI_DEBUG = ([name]:[-|+|+FILENAME])(,[name]:[-|+|+FILENAME])*
//
// The + or - specifies wheather the named object is on or off.  If a file is 
// specified it is opened in ios::out mode.  If no file is specified,
// the stream is directed to cerr.  The : and , characters are
// restricted to deliminators.
//
// Example:
//
// SCI_DEBUG = modules.meshgen.warning:+meshgen.out,util.debugstream.error:-
//
// Future Additions:
// o Possible additions to constructor:
//   - Default file to output to
//   - Mode that the file will be opened in (append, out, etc.) (usefulness??)
// o Allow DEFAULT specification in the env variable which would override
//   all default settings. (usefulness??)
// o Time stamp option
//
// Annoyances:
// o Because the list of environment variables, "environ", is built at
// run time, and getenv queries that list, I have not been able to
// figure out a way to requery the environment variables during
// execution.  

#ifndef SCI_project_DebugStream_h
#define SCI_project_DebugStream_h 1

#include <SCICore/share/share.h>

#include <string>
#include <stdlib.h> // for getenv()
#include <iostream>
#include <fstream>


namespace SCICore{
  namespace Util{

    using std::streambuf;
    using std::ostream;
    using std::cerr;
    using std::endl;
    using std::ofstream;
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
    class DebugStream: public ostream{
    private:
      // identifies me uniquely
      string name;
      // my default action (used if nothing is specified in SCI_DEBUG)
      bool defaulton;
      // the buffer that is used for output redirection
      DebugBuf* dbgbuf;
      // if false, all input is ignored
      bool isactive;
      // check the environment variable
      void checkenv(string);
            
    public:
      DebugStream();
      DebugStream(string name, bool defaulton = true);
      ~DebugStream();
      bool active() {return isactive;};
      // the ostream that output should be redirected to. cerr by default.
      ostream *outstream;
    };
    
  } // End namespace Util
} // End namespace SCICore


  








#endif
