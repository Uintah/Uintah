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


PropertyManager::PropertyManager()
{
}


PropertyManager::PropertyManager(const PropertyManager &copy)
{
//   map_type::const_iterator pi = copy.properties_.begin();
//   while (pi != copy.properties_.end())
//     properties_[pi->first] = pi->second->clone();
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


#define PROPERTYMANAGER_VERSION 1

void
PropertyManager::io(Piostream &stream)
{
  stream.begin_class("PropertyManager", PROPERTYMANAGER_VERSION);

  // TODO: implement this.  read/write all properties not marked temporary.

  stream.end_class();
}


} // namespace SCIRun
