// PropertyManager.cc
//
//  Written by:
//   Michael Callahan
//   Department of Computer Science
//   University of Utah
//   January 2001
//
//  Copyright (C) 2001 SCI Institute
//
//  Manage dynamic properties of persistent objects.
//

#include <Core/Datatypes/PropertyManager.h>


namespace SCIRun {

PersistentTypeID PropertyManager::type_id("PropertyManager", "Datatype", NULL);


PropertyManager::PropertyManager()
{
}


PropertyManager::PropertyManager(const PropertyManager &copy)
{
  map_type::const_iterator pi = copy.properties_.begin();
  while (pi != copy.properties_.end())
  {
    // TODO: Reference count Datatype values.
    Property *oldprop = (*pi).second;
    Property *prop = new Property;
    prop->stringval = oldprop->stringval;
    prop->dataval = oldprop->dataval;
    prop->tmp = oldprop->tmp;
    properties_[(*pi).first] = prop;
  }
}


PropertyManager::~PropertyManager()
{
  // Clear all the properties.
  map_type::iterator pi = properties_.begin();
  while (pi != properties_.end())
  {
    Property *prop = (*pi).second;
    if (prop->dataval)
    {
      // TODO: ref count decrement and maybe delete Datatype.
    }
    delete prop;
  }
}


void
PropertyManager::set_string(const string name, const string property)
{
  map_type::iterator loc = properties_.find(name);
  Property *prop;
  if (loc == properties_.end())
  {
    prop = new Property;
    properties_[name] = prop;
  }
  prop->stringval = property;
}


const string
PropertyManager::get_string(const string name)
{
  map_type::iterator loc = properties_.find(name);
  if (loc == properties_.end())
  {
    // TODO: Default value?
    return "";
  }
  else
  {
    return (*loc).second->stringval;
  }
}


void
PropertyManager::set_data(const string name, Datatype *dval)
{
  // TODO: properly reference count the data.
  map_type::iterator loc = properties_.find(name);
  Property *prop;
  if (loc == properties_.end())
  {
    prop = new Property;
    properties_[name] = prop;
  }
  prop->dataval = dval;
}


Datatype *
PropertyManager::get_data(const string name)
{
  map_type::iterator loc = properties_.find(name);
  if (loc == properties_.end())
  {
    return NULL;
  }
  else
  {
    return (*loc).second->dataval;
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


PropertyManager::Property::Property()
{
  stringval = "";
  dataval = NULL;
  tmp = false;
}


} // namespace SCIRun
