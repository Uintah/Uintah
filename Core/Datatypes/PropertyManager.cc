/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
#include <Core/Datatypes/PropertyManager.h>

namespace SCIRun {


PersistentTypeID 
PropertyBase::type_id("PropertyBase", "Datatype", maker);

Persistent* PropertyBase::maker()
{
  return scinew PropertyBase;
}


/*
 * Get
 */

template<> bool PropertyManager::get(const string &name, char &ref) 
{
  return get_scalar( name, ref );
}

template<> bool PropertyManager::get(const string &name, short &ref) 
{
  return get_scalar( name, ref );
}
template<> bool PropertyManager::get(const string &name, int &ref) 
{
  return get_scalar( name, ref );
}
template<> bool PropertyManager::get(const string &name, float &ref) 
{
  return get_scalar( name, ref );
}
template<> bool PropertyManager::get(const string &name, double &ref) 
{
  return get_scalar( name, ref );
}


Persistent* make_PropertyManager()
{
  return scinew PropertyManager;
}

PersistentTypeID PropertyManager::type_id("PropertyManager", 
					  "Datatype", 
					  make_PropertyManager);


PropertyManager::PropertyManager() : 
  size_(0),
  frozen_(false)
{
}

// PropertyManagers are created thawed.  Only non transient data is copied.
PropertyManager::PropertyManager(const PropertyManager &copy) :
  size_(0),
  frozen_(false)
{
  map_type::const_iterator pi = copy.properties_.begin();
  while (pi != copy.properties_.end()) {
    if (! pi->second->transient()) {
      properties_[pi->first]=pi->second->copy();
      ++size_;
    }
    ++pi;
  }
}


PropertyManager::~PropertyManager()
{
  // Clear all the properties.
  map_type::iterator pi = properties_.begin();
  while (pi != properties_.end()) {
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

void
PropertyManager::remove( const string &name )
{
  lock.lock();

  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    properties_.erase(name);
    size_--;
  }

  lock.unlock();
}

void
PropertyManager::clear_transient()
{
  map_type::iterator iter = properties_.begin();
  if (iter != properties_.end()) {
    pair<const string, PropertyBase *> p = *iter;
    
    if (p.second->transient()) {
      properties_.erase(iter);
      size_--;
    }
    ++iter;
  }
}


#define PROPERTYMANAGER_VERSION 1

void
PropertyManager::io(Piostream &stream)
{
  stream.begin_class("PropertyManager", PROPERTYMANAGER_VERSION);

  Pio( stream, size_ );

  if ( stream.writing() ) {
    map_type::iterator i = properties_.begin(); 
    while ( i != properties_.end() ) {
      string name = i->first;
      Pio(stream, name);
      Persistent *p = i->second;
      stream.io( p, PropertyBase::type_id );
      //i->second->io( stream );
      ++i;
    }
  }
  else {
    string name;
    Persistent *p = 0;
    for (int i=0; i<size_; i++ ) {
      Pio(stream, name );
      stream.io( p, PropertyBase::type_id );
      properties_[name] = static_cast<PropertyBase *>(p);
    }
  }
  // TODO: implement this.  read/write all properties not marked temporary.

  stream.end_class();
}


} // namespace SCIRun
