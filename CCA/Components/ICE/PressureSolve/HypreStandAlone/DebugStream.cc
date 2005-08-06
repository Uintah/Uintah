#include "DebugStream.h"
using namespace std;

DebugBuf::DebugBuf(void)
: _lineBegin(true)
{}


DebugBuf::~DebugBuf(void)
{}

int DebugBuf::overflow(int ch)
{
  if (owner->active()) {
    return(*(owner->outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}

streamsize DebugBuf::xsputn (const char* s,
                             streamsize num) {
  if (owner->getLevel() < owner->getVerboseLevel()) {
    if (s[num-1] == '\n')
    {
      _lineBegin = true;
      // Set back level to verboseLevel so that stream will
      // print everything until the next explicit setLevel() call.
      owner->setLevel(owner->getVerboseLevel());
    }
    return num;
  }
  if (_lineBegin) {
    *(owner->outstream) << lineHeader();
    _lineBegin = false;
  }
  *(owner->outstream) << s;
  if (s[num-1] == '\n')
  {
    _lineBegin = true;
    // Set back level to verboseLevel so that stream will
    // print everything until the next explicit setLevel() call.
    owner->setLevel(owner->getVerboseLevel());
  }
  
  return num;
}

DebugStream::DebugStream(const string& iname, bool active) :
    ostream(0), _verboseLevel(0), _level(0)
{
  _dbgbuf = new DebugBuf();
  init(_dbgbuf);
  _name = iname;
  _dbgbuf->owner = this;
  // set default values
  _isactive = active;
  if (_isactive){
    outstream = &cerr;
  } else {
    outstream = 0;
  }
}

DebugStream::~DebugStream(void)
{
  if (outstream && outstream != &cerr){
    delete(outstream);
  }
  delete _dbgbuf;
}

void DebugStream::setActive(const bool active)
{
  _isactive = active;
  if (_isactive) {
    outstream = &cerr;
  } else {
    outstream = 0;
  }
}
