/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include "DebugStream.h"
using namespace std;

DebugBuf::DebugBuf(void)
: _lineBegin(true)
{}


DebugBuf::~DebugBuf(void)
{}

int DebugBuf::overflow(int ch)
  // Writing one character: our implementation of the streambuf virtual
  // overflow(). Not sure that it properly works yet.
{
  if ((owner->getLevel() > owner->getVerboseLevel())
      || (!owner->active())) {
    if (ch == '\n') {
      _lineBegin = true;
      // To set back level to verboseLevel so that stream will
      // print everything until the next explicit setLevel() call:
      if (!owner->getStickyLevel()) {
        owner->setLevel(owner->getVerboseLevel());
      }
      owner->flush();
    }
    return 0;
  }
  if (_lineBegin) {
    *(owner->outstream) << lineHeader(owner->getIndent());
    _lineBegin = false;
  }
  if (ch == '\n') {
    _lineBegin = true;
    // To set back level to verboseLevel so that stream will
    // print everything until the next explicit setLevel() call:
    if (!owner->getStickyLevel()) {
      owner->setLevel(owner->getVerboseLevel());
    }
    owner->flush();
  }
  return(*(owner->outstream) << (char)ch ? 0 : EOF);
}

streamsize
DebugBuf::xsputn(const char* s,
                 streamsize num)
  // Writing num characters of the char array s: our implementation
  // of the virtual function streambuf::xsputn().
{
  //  cerr << "verbose=" << owner->getVerboseLevel() 
  //       << " level=" << owner->getLevel() << "\n";
  if ((owner->getLevel() > owner->getVerboseLevel())
      || (!owner->active())) {
    if ((num >= 1) && (s[num-1] == '\n')) {
      _lineBegin = true;
      // To set back level to verboseLevel so that stream will
      // print everything until the next explicit setLevel() call:
      if (!owner->getStickyLevel()) {
        owner->setLevel(owner->getVerboseLevel());
      }
      owner->flush();
    }
    return num;
  }
  if (_lineBegin) {
    *(owner->outstream) << lineHeader(owner->getIndent());
    _lineBegin = false;
  }
  // With setfill(), this seems to sometimes print some garbage.
  *(owner->outstream) << s;
  if ((num >= 1) && (s[num-1] == '\n')) {
    _lineBegin = true;
    // To set back level to verboseLevel so that stream will
    // print everything until the next explicit setLevel() call:
    if (!owner->getStickyLevel()) {
      owner->setLevel(owner->getVerboseLevel());
    }
    owner->flush();
  }
  return num;
}

DebugStream::DebugStream(const string& iname, bool active) :
  ostream(0), _verboseLevel(0), _level(0), _indent(0), _stickyLevel(false)
{
  _dbgbuf = scinew DebugBuf();
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

void DebugStream::indent(void)
{
  if (_indent >= 10) {
    cerr << "\n\nWarning: DebugStream indent overflow" << "\n";
  } else {
    _indent++;
  }
  //  cerr << "(indent=" << _indent << ") ";
}

void DebugStream::unindent(void)
{
  if (_indent == 0) {
    cerr << "\n\nWarning: DebugStream indent underflow" << "\n";
  } else {
    _indent--;
  }
  //  cerr << "(indent=" << _indent << ") ";
}
