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
}

void 
ComponentImpl::setServices( const Services::pointer &s )
{
  if ( !s.isNull() ) {
    services_ = pidl_cast<SciServices::pointer>(s);
  }
  else {
    // Component shutdown

    //     ComponentIdImpl * ci = dynamic_cast<ComponentIdImpl*>(
    // 				 services_->getComponentID().getPointer() );
    
    //     cerr << "Component (" << getpid() << "): " 
    // 	 << ci->toString() << " told to Shutdown...\n";

    /////
    //
    // If a component wishes to shut itself down, then it
    // should all the following.  It should NOT set its own
    // services to 0 (which is what this was doing before).
    //
    //services_->shutdown();

    services_ = SciServices::pointer(0);
  }
}

} // namespace sci_cca

