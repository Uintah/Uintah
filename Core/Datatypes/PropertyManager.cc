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


/* PropertyManager.cc
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Institute
 *
 *  Manage properties of persistent objects.
 */

#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Datatypes/PropertyManager.h>

namespace SCIRun {


PersistentTypeID 
PropertyBase::type_id("PropertyBase", "Datatype", maker);

Persistent* PropertyBase::maker()
{
  return scinew PropertyBase(false);
}


Persistent* make_PropertyManager()
{
  return scinew PropertyManager;
}

PersistentTypeID PropertyManager::type_id("PropertyManager", 
					  "Datatype", 
					  make_PropertyManager);


PropertyManager::PropertyManager() : 
  frozen_(false)
{
}


// PropertyManagers are created thawed.  Only non transient data is copied.
PropertyManager::PropertyManager(const PropertyManager &copy) :
  frozen_(false)
{
  lock.lock();
  map_type::const_iterator pi = copy.properties_.begin();
  while (pi != copy.properties_.end())
  {
    if (! pi->second->transient())
    {
      properties_[pi->first] = pi->second->clone();
    }
    ++pi;
  }
  lock.unlock();
}


PropertyManager & 
PropertyManager::operator=(const PropertyManager &src)
{
  copy_properties(&src);
  return *this;
}


void
PropertyManager::copy_properties( const PropertyManager *src)
{
  thaw();
  lock.lock();

  map_type::const_iterator pi = src->properties_.begin();
  while (pi != src->properties_.end())
  {
    if (! pi->second->transient())
    {
      properties_[pi->first] = pi->second->clone();
    }
    ++pi;
  }
  
  lock.unlock();
  freeze();
}


bool
PropertyManager::operator==(const PropertyManager &pm)
{
  if (nproperties() != pm.nproperties())
    return false;

  lock.lock();

  bool result = true;
  map_type::const_iterator pi = pm.properties_.begin();
  
  while (pi != pm.properties_.end()) {
    
    map_type::iterator loc = properties_.find(pi->first);

    if (loc == properties_.end() )
    {
      result = false;
      break;
    }

    if( *(pi->second) != *(loc->second) )
    {
      result = false;
      break;
    }

    ++pi;
  }

  lock.unlock();

  return result;
}


bool
PropertyManager::operator!=(const PropertyManager &pm)
{
  return ! (*this == pm);
}


PropertyManager::~PropertyManager()
{
  // Clear all the properties.
  map_type::iterator pi = properties_.begin();
  while (pi != properties_.end())
  {
    delete pi->second;
    ++pi;
  }
}


void
PropertyManager::thaw()
{
  // Assert that detach has been called on any handles to this PropertyManager.
  ASSERT(ref_cnt <= 1);
  // Clean up properties.
  lock.lock();

  clear_transient();
  frozen_ = false;

  lock.unlock();
}


void
PropertyManager::freeze()
{
  lock.lock();

  frozen_ = true;

  lock.unlock();
}


bool 
PropertyManager::is_property(const string &name)
{
  lock.lock();

  bool ans;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end())
    ans = true;
  else
    ans = false;

  lock.unlock();

  return ans;
} 


string
PropertyManager::get_property_name(size_t index)
{
  if (index < nproperties())
  {
    lock.lock();

    map_type::const_iterator pi = properties_.begin();

    for(size_t i=0; i<index; i++ )
      ++pi;

    lock.unlock();

    return pi->first;
  }
  else
  {
    return string("");
  } 
}


void
PropertyManager::remove_property( const string &name )
{
  lock.lock();

  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end())
  {
    properties_.erase(name);
  }

  lock.unlock();
}


void
PropertyManager::clear_transient()
{
  map_type::iterator iter = properties_.begin();
  if (iter != properties_.end())
  {
    pair<const string, PropertyBase *> p = *iter;
    
    if (p.second->transient())
    {
      properties_.erase(iter);
    }
    ++iter;
  }
}


#define PROPERTYMANAGER_VERSION 2

void
PropertyManager::io(Piostream &stream)
{
  const int version =
    stream.begin_class("PropertyManager", PROPERTYMANAGER_VERSION);
  if ( stream.writing() )
  {
    lock.lock();
    unsigned int nprop = nproperties();
    Pio(stream, nprop);
    map_type::iterator i = properties_.begin(); 
    while ( i != properties_.end() )
    {
      string name = i->first;
      Pio(stream, name);
      Persistent *p = i->second;
      stream.io( p, PropertyBase::type_id );
      ++i;
    }
    lock.unlock();
  }
  else
  {
    unsigned int size;
    Pio( stream, size );
    lock.lock();

    string name;
    Persistent *p = 0;
    for (unsigned int i=0; i<size; i++ )
    {
      Pio(stream, name );
      stream.io( p, PropertyBase::type_id );
      properties_[name] = static_cast<PropertyBase *>(p);
      if (version < 2 && name == "minmax")
      {
	properties_[name]->set_transient(true);
      }
    }
    lock.unlock();
  }

  stream.end_class();
}


} // namespace SCIRun
