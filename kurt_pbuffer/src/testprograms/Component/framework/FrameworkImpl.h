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
