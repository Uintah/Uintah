#ifndef __DEBUGSTREAM_H__
#define __DEBUGSTREAM_H__

#include <iostream>
#include <iomanip>
#include <sstream>
using std::streambuf;
using std::ostream;
using std::string;
using std::setw;

class DebugStream;
class DebugBuf;

class DebugBuf:public streambuf{
 private:
 public:
  DebugBuf();
  ~DebugBuf();
  // points the the DebugStream that instantiated me
  DebugStream *owner;
  int overflow(int ch);
};

class DebugStream: public ostream{
 private:
  
  string    _name;       // identifies me uniquely
  DebugBuf* _dbgbuf;     // the buffer that is used for output redirection
  bool      _isactive;   // if false, all input is ignored
            
 public:
  DebugStream(void);
  DebugStream(const string& name, bool defaulton = true);
  ~DebugStream(void);
  bool active(void) { return _isactive; };
  void setActive(const bool active);
  // the ostream that output should be redirected to. cerr by default.
  ostream *outstream;
};
    
#endif // __DEBUGSTREAM_H__
