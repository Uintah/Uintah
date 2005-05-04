/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
