#include <Core/Thread/Mutex.h>
#include <Core/Thread/AtomicCounter.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>

#include <sstream>
#include <iostream>

#include <stdio.h>

namespace sci_cca {

using SCIRun::AtomicCounter;
using SCIRun::Mutex;
using std::cerr;
using std::ostringstream;

static AtomicCounter* generation = 0;
static Mutex lock("Component generation counter initialization lock");

ComponentIdImpl::ComponentIdImpl()
{
}

void
ComponentIdImpl::init( const string &host, const string &program )
{
  if (!generation) {
    lock.lock();
    if(!generation)
      generation = new AtomicCounter("Component generation counter", 1);
    lock.unlock();
  }

  number_ = (*generation)++;

  host_ = host;
  program_ = program;

  ostringstream tmp;
  tmp << host_ << "/" << program_ << "-" << number_; 
  
  id_ = tmp.str();
}

ComponentIdImpl::~ComponentIdImpl()
{
  cerr << "ComponenetIdImpl " << id_ << " destructor\n";
}

string
ComponentIdImpl::toString()
{
  return id_;
}

string
ComponentIdImpl::fullString()
{
  char full[ 1024 ];
  
  sprintf( full, "%s, %s, %d, %s", host_.c_str(), program_.c_str(), number_,
	   id_.c_str() );

  return string( full );
}

} // namespace sci_cca

