
#include <testprograms/Component/framework/LocalFramework.h>

#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/PortImpl.h>

#include <iostream>

namespace sci_cca {

using std::cerr; 

LocalFramework::LocalFramework()
{
}

LocalFramework::~LocalFramework()
{
}

void 
LocalFramework::setServices( const Services::pointer &services )
{
  services_ = services;

  array1<string> properties;

//   Port port = new PortImpl; // should be specific port 
//   PortInfo info = services_->createPortInfo( string("local framework"),
// 					    string("sci_cca.framework"),
// 					     properties);
  //services_->addProvidesPort( port, info );
  cerr << "local framework ready\n";
}

} // namespace sci_cca


