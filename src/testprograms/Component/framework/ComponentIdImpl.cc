#include <Core/Thread/Mutex.h>
#include <Core/Thread/AtomicCounter.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>

#include <sstream>

using std::cerr;

namespace sci_cca {

using SCIRun::AtomicCounter;
using SCIRun::Mutex;

static AtomicCounter* generation = 0;
static Mutex lock("Component generation counter initialization lock");

ComponentIdImpl::ComponentIdImpl()
{
  if (!generation) {
    lock.lock();
    if(!generation)
      generation = new AtomicCounter("Component generation counter", 1);
    lock.unlock();
  }

  number_ = (*generation)++;
}

void
ComponentIdImpl::init( const string &host, const string &program )
{
  host_ = host;
  program_ = program;

  ostringstream tmp;
  tmp << host_ << "/" << program_ << "-" << number_; 
  
  id_ = tmp.str();
}

ComponentIdImpl::~ComponentIdImpl()
{
  cerr << "componenet " << id_ << " destructor\n";
}

string
ComponentIdImpl::toString()
{
  return id_;
}

} // namespace sci_cca

