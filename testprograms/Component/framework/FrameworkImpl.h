

#ifndef FrameworkImpl_h
#define FrameworkImpl_h

#include <map>
#include <testprograms/Component/framework/cca_sidl.h>

namespace sci_cca {

class ComponentRecord;
class UsePortRecord;
class ProvidePortRecord;

class FrameworkImpl : public Framework_interface {
private:
  map<ComponentID, ComponentRecord *> components_;
  map<string, UsePortRecord *> use_ports_;
  map<string, ProvidePortRecord *> provide_ports_;
  map<string, Port > framework_ports_;

  typedef map<string, UsePortRecord *> use_map;
  typedef map<string, ProvidePortRecord *> provide_map;
  typedef map<string, Port> framework_map;

public:
  FrameworkImpl();
  virtual ~FrameworkImpl();
  
  virtual Port getPort( const ComponentID &, const string &);
  virtual void registerUsesPort( const ComponentID &, const PortInfo &);
  virtual void unregisterUsesPort( const ComponentID &, const string & );
  virtual void addProvidesPort( const ComponentID &, const Port &,
				const PortInfo&);
  virtual void removeProvidesPort( const ComponentID &, const string &);
  virtual void releasePort( const ComponentID &, const string &);

  virtual bool registerComponent( const string &, const string &, Component &);
  virtual void unregisterComponent( const ComponentID & );
};

} // namespace sci_cca

#endif FrameworkImpl_h
