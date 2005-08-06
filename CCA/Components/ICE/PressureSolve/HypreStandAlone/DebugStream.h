#ifndef __DEBUGSTREAM_H__
#define __DEBUGSTREAM_H__

#include <iostream>
#include <iomanip>
#include <sstream>
using std::string;

class DebugStream;
class DebugBuf;
extern std::string lineHeader(void);

class DebugBuf: public std::streambuf {
 public:
  DebugStream* owner;      // points to the DebugStream that instantiated me

  DebugBuf(void);
  ~DebugBuf(void);
  int overflow(int ch);
  std::streamsize xsputn (const char* s,
                          std::streamsize num);
 private:
   bool _lineBegin;
};

class DebugStream: public std::ostream {        
 public:
  DebugStream(void);
  DebugStream(const std::string& name,
    bool defaulton = true);
  ~DebugStream(void);
  bool active(void) { return _isactive; };
  int getVerboseLevel(void) { return _verboseLevel; }
  int getLevel(void) { return _level; }
  void setActive(const bool active);
  void setVerboseLevel(const int verboseLevel) { _verboseLevel = verboseLevel; }
  void setLevel(const int level) { _level = level; }
  
  std::ostream *outstream; // ostream that output redirected to. default: cout
  
 private: 
  std::string    _name;         // identifies me uniquely
  DebugBuf* _dbgbuf;       // the buffer that is used for output redirection
  bool      _isactive;     // if false, all input is ignored
  int       _verboseLevel; // verbose level for printouts, badly implemented...
  int       _level; // verbose level for printouts, badly implemented...
};
    
#endif // __DEBUGSTREAM_H__
