

#ifndef FrameworkImpl_h
#define FrameworkImpl_h

#include <map>
#include <Core/Thread/CrowdMonitor.h>
#include <testprograms/Component/framework/cca_sidl.h>


namespace sci_cca {

using std::map;
using std::string;
using SCIRun::CrowdMonitor;

class ComponentRecord;
class UsePortRecord;
class ProvidePortRecord;
class Registry;

class BuilderServicesImpl;
class RegistryServicesImpl;

class FrameworkImpl : public Framework {

public:
  FrameworkImpl();
  virtual ~FrameworkImpl();
  
  virtual bool registerComponent(const string&, const string&,
				 Component::pointer&);
  virtual void unregisterComponent(const ComponentID::pointer& );

  virtual Port::pointer getPort(const ComponentID::pointer&, const string&);
  virtual void registerUsesPort(const ComponentID::pointer&,
				const PortInfo::pointer&);
  virtual void unregisterUsesPort(const ComponentID::pointer&,
				  const string& );
  virtual void addProvidesPort(const ComponentID::pointer&,
			       const Port::pointer&,
			       const PortInfo::pointer&);
  virtual void removeProvidesPort(const ComponentID::pointer&,
				  const string&);
  virtual void releasePort(const ComponentID::pointer&, const string&);
  void shutdown();

private:

  string hostname_;
  ComponentID::pointer id_;
  Registry *registry_;
  map<string, Port::pointer> ports_;
  
  CrowdMonitor ports_lock_;

  typedef map<string, Port::pointer>::iterator port_iterator;

  friend class BuilderServicesImpl;
  friend class RegistryServicesImpl;
};

} // namespace sci_cca

#endif // FrameworkImpl_h
