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


#include <testprograms/Component/framework/RegistryServicesImpl.h>
#include <testprograms/Component/framework/cca_sidl.h>
#include <testprograms/Component/framework/Registry.h>
#include <testprograms/Component/framework/FrameworkImpl.h>

#include <iostream>

namespace sci_cca {

using namespace std;

typedef Registry::component_iterator component_iterator;

RegistryServicesImpl::RegistryServicesImpl() 
{
}


RegistryServicesImpl::~RegistryServicesImpl()
{
}


void 
RegistryServicesImpl::init( const Framework::pointer &f ) 
{ 
  framework_ = f; 
  
  registry_ = dynamic_cast<FrameworkImpl *>(f.getPointer())->registry_;
}

void
RegistryServicesImpl::getActiveComponentList( array1<ComponentID::pointer> & cIds )
{
  registry_->connections_.readLock();
  
  Registry::component_iterator iter = registry_->components_.begin();

  for( ; iter != registry_->components_.end(); iter++ )
    {
      ComponentRecord * cr = (*iter).second;
      cIds.push_back( cr->id_ );
    }
  registry_->connections_.readUnlock();
}

void
RegistryServicesImpl::shutdown()
{
  FrameworkImpl *fw = dynamic_cast<FrameworkImpl*>( framework_.getPointer() );
  fw->shutdown();
}

} // namespace sci_cca
