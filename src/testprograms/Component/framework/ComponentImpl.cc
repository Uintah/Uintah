#include <Core/Thread/Mutex.h>
#include <Core/Thread/AtomicCounter.h>
#include <testprograms/Component/framework/ComponentImpl.h>
#include <testprograms/Component/framework/ComponentIdImpl.h>
#include <testprograms/Component/framework/SciServicesImpl.h>

#include <sstream>
#include <iostream>
#include <unistd.h>

namespace sci_cca {

using std::cerr;

ComponentImpl::ComponentImpl()
{
}

ComponentImpl::~ComponentImpl()
{
  cerr << "Component destructor\n";
}

void 
ComponentImpl::setServices( const Services &s )
{
  if ( s ) {
    services_ = pidl_cast<SciServices>(s);
  }
  else {
    // Component shutdown

    //    SciServicesImpl * ss = 
    //      dynamic_cast<SciServicesImpl*>( services_.getPointer() );
    //    ss->done();

    ComponentIdImpl * ci = dynamic_cast<ComponentIdImpl*>(
				 services_->getComponentID().getPointer() );

    cerr << "Component (" << getpid() << "): " 
	 << ci->toString() << " told to Shutdown...\n";
    /////
    //
    // If a component wishes to shut itself down, then it
    // should all the following.  It should NOT set its own
    // services to 0 (which is what this was doing before).
    //
    //services_->shutdown();

  }
}

} // namespace sci_cca

