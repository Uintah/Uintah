
#ifndef Registry_h
#define Registry_h

#include <string>
#include <map>

#include <testprograms/Component/framework/cca_sidl.h>

namespace sci_cca {

class ConnectionRecord;

class PortRecord {
public:
  ComponentID id_;
  PortInfo info_;
  ConnectionRecord *connection_;
};

class UsePortRecord : public PortRecord {
public:
};

class ProvidePortRecord : public PortRecord {
public:
  Port port_;
  bool in_use_;
};

class ConnectionRecord {
public:
  void disconnect();

public:
  UsePortRecord *use_;
  ProvidePortRecord *provide_;
};


class FrameworkImpl;

class ComponentRecord {
private:
  typedef map<string,ProvidePortRecord *>::iterator provide_iterator;
  typedef map<string,UsePortRecord *>::iterator use_iterator;

  ComponentID id_;
  Component component_;
  Services services_;
  map<string, ProvidePortRecord *> provides_;
  map<string, UsePortRecord *> uses_;

public:
  ComponentRecord( const ComponentID &id );
  virtual ~ComponentRecord();

  virtual Port getPort( const string &);
  virtual void registerUsesPort( const PortInfo &);
  virtual void unregisterUsesPort( const string & );
  virtual void addProvidesPort( const Port &, const PortInfo&);
  virtual void removeProvidesPort( const string &);
  virtual void releasePort( const string &);

  virtual ProvidePortRecord *getProvideRecord( const string & );
  virtual UsePortRecord *getUseRecord( const string & );
  friend FrameworkImpl;
};


class Registry {
public:
  map<ComponentID, ComponentRecord *> components_;

  typedef map<ComponentID, ComponentRecord *>::iterator component_iterator;
};

} // namespace sci_cca

#endif Registry_h
