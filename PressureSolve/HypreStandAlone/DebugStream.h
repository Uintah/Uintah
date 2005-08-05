#ifndef __DEBUGSTREAM_H__
#define __DEBUGSTREAM_H__

#include <iostream>
#include <iomanip>
#include <sstream>
using std::string;

class DebugStream;
class DebugBuf;

class DebugBuf: public std::streambuf {
 public:
  DebugBuf(void);
  ~DebugBuf(void);
  int overflow(int ch);    // What does this function do?

  DebugStream *owner;      // points to the DebugStream that instantiated me

 private:
};

class DebugStream: public std::ostream {        
 public:
  DebugStream(void);
  DebugStream(const string& name, bool defaulton = true);
  ~DebugStream(void);
  bool active(void) { return _isactive; };
  void setActive(const bool active);

  std::ostream *outstream; // ostream that output redirected to. default: cout
  
 private: 
  string    _name;         // identifies me uniquely
  DebugBuf* _dbgbuf;       // the buffer that is used for output redirection
  bool      _isactive;     // if false, all input is ignored
  int       _verboseLevel; // verbose level for printouts, badly implemented...
};
    
#endif // __DEBUGSTREAM_H__
