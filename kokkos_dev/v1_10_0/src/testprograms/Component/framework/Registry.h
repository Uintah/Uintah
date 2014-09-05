
#ifndef Registry_h
#define Registry_h

#include <string>
#include <map>

#include <Core/Thread/CrowdMonitor.h>
#include <testprograms/Component/framework/cca_sidl.h>

#include <string>

namespace sci_cca {

using SCIRun::CrowdMonitor;
using std::map;
using std::string; 

class ConnectionRecord;

class PortRecord {
public:
  virtual ~PortRecord() {}

  ComponentID::pointer id_;
  PortInfo::pointer info_;
  ConnectionRecord *connection_;
};

class UsePortRecord : public PortRecord {
public:
};

class ProvidePortRecord : public PortRecord {
public:
  Port::pointer port_;
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
public:
  typedef map<string,ProvidePortRecord *>::iterator provide_iterator;
  typedef map<string,UsePortRecord *>::iterator use_iterator;

  ComponentID::pointer id_;
  Component::pointer component_;
  Services::pointer services_;
  map<string, ProvidePortRecord *> provides_;
  map<string, UsePortRecord *> uses_;

public:
  ComponentRecord( const ComponentID::pointer &id );
  virtual ~ComponentRecord();

  virtual Port::pointer getPort( const string &);
  virtual void registerUsesPort( const PortInfo::pointer &);
  virtual void unregisterUsesPort( const string & );
  virtual void addProvidesPort( const Port::pointer &, const PortInfo::pointer&);
  virtual void removeProvidesPort( const string &);
  virtual void releasePort( const string &);

  virtual ProvidePortRecord *getProvideRecord( const string & );
  virtual UsePortRecord *getUseRecord( const string & );
  friend class FrameworkImpl;
};


class Registry {
public:
  map<string, ComponentRecord *> components_;

  typedef map<string, ComponentRecord *>::iterator component_iterator;

  CrowdMonitor connections_;

public:
  Registry();

  ProvidePortRecord *getProvideRecord( const ComponentID::pointer &, const string &);
  UsePortRecord *getUseRecord( const ComponentID::pointer &, const string &);

};

} // namespace sci_cca

#endif // Registry_h
