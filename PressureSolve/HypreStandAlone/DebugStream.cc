#include "DebugStream.h"
using namespace std;

DebugBuf::DebugBuf(void)
: _lineBegin(true)
{}


DebugBuf::~DebugBuf(void)
{}

int DebugBuf::overflow(int ch)
{
  //  cerr << " (xsputn begin) ";
  //  cerr << "*owner = " << *owner << "\n";
  if ((owner->getLevel() < owner->getVerboseLevel())
      || (!owner->active())) {
    if (ch == '\n') {
      //      cerr << "HERE1\n";
      _lineBegin = true;
      // Set back level to verboseLevel so that stream will
      // print everything until the next explicit setLevel() call.
      //      cerr << "HERE2\n";
      owner->setLevel(owner->getVerboseLevel());
      owner->flush();
      //      cerr << "HERE3\n";
    }
    //    cerr << "(xsputn end1) ";
    return 0;
  }
  //  cerr << "HERE4\n";
  if (_lineBegin) {
    //    cerr << "HERE5\n";
    //    cerr << "lineHeader = " <<lineHeader()<<"\n";
    *(owner->outstream) << lineHeader();
    //    cerr << "HERE6\n";
    _lineBegin = false;
  }
  if (ch == '\n') {
    _lineBegin = true;
    // Set back level to verboseLevel so that stream will
    // print everything until the next explicit setLevel() call.
    owner->setLevel(owner->getVerboseLevel());
    owner->flush();
    //    cerr << "HERE9\n";
  }
  //  cerr << "HERE7\n";
  return(*(owner->outstream) << (char)ch ? 0 : EOF);
  //  return 0;
}

streamsize DebugBuf::xsputn (const char* s,
                             streamsize num) {
  //  cerr << " (xsputn begin) ";
  //  cerr << "*owner = " << *owner << "\n";
  if ((owner->getLevel() < owner->getVerboseLevel())
      || (!owner->active())) {
    if ((num >= 1) && (s[num-1] == '\n')) {
      //      cerr << "HERE1\n";
      _lineBegin = true;
      // Set back level to verboseLevel so that stream will
      // print everything until the next explicit setLevel() call.
      //      cerr << "HERE2\n";
      owner->setLevel(owner->getVerboseLevel());
      owner->flush();
      //      cerr << "HERE3\n";
    }
    //    cerr << "(xsputn end1) ";
    return num;
  }
  //  cerr << "HERE4\n";
  if (_lineBegin) {
    //    cerr << "HERE5\n";
    //    cerr << "lineHeader = " <<lineHeader()<<"\n";
    *(owner->outstream) << lineHeader();
    //    cerr << "HERE6\n";
    _lineBegin = false;
  }
  //  cerr << "HERE7\n";
  *(owner->outstream) << s;
  //  cerr << "HERE8\n";
  if ((num >= 1) && (s[num-1] == '\n')) {
    _lineBegin = true;
    // Set back level to verboseLevel so that stream will
    // print everything until the next explicit setLevel() call.
    owner->setLevel(owner->getVerboseLevel());
    owner->flush();
    //    cerr << "HERE9\n";
  }
  //  cerr << "(xsputn end2) ";  
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
