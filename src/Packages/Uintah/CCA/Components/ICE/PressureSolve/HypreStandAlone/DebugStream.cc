#include "DebugStream.h"
using namespace std;

DebugBuf::DebugBuf(void)
{}


DebugBuf::~DebugBuf(void)
{}

int DebugBuf::overflow(int ch)
{
  if (owner->active()){
    return(*(owner->outstream) << (char)ch ? 0 : EOF);
  }
  return 0;
}

DebugStream::DebugStream(const string& iname, bool active) :
    ostream(0)
{
  _dbgbuf = new DebugBuf();
  init(_dbgbuf);
  _name = iname;
  _dbgbuf->owner = this;
  // set default values
  _isactive = active;
  if (_isactive){
    outstream = &cout;
  } else {
    outstream = 0;
  }
}

DebugStream::~DebugStream(void)
{
  if (outstream && outstream != &cout){
    delete(outstream);
  }
  delete _dbgbuf;
}

void DebugStream::setActive(const bool active)
{
  cerr << "setActive begin()\n";
  _isactive = active;
  if (_isactive){
    outstream = &cout;
  } else {
    outstream = 0;
  }
  cerr << "setActive end()\n";
}
