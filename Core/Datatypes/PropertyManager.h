/* PropertyManager.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Institute
 *
 *  Manage dynamic properties of persistent objects.
 */

#ifndef PropertyManager_h
#define PropertyManager_h 

#include <map>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/builtin.h>

namespace SCIRun {


class PropertyBase {
public:
  void *obj_;
  virtual ~PropertyBase() {}
};

template<class T>
class Property : public T, public PropertyBase {
public:
  Property( T &o) { obj_= &o; } 
};

template<class T>
class Property<T *> : public T, public PropertyBase {
public:
  Property( T *o) { obj_ = o; } 
};




/*
 * PropertyManager
 */

class PropertyManager : public Datatype
{
public:
  PropertyManager();
  PropertyManager(const PropertyManager &copy);
  virtual ~PropertyManager();

  
  template<class T> void store( const string &, T &);
  template<class T> void store( const string &, const T &);
  void store( const string &name, const char *s )  { store(name,string(s));}
  void store( const string &name, const char s )   { store(name,Char(s));}
  void store( const string &name, const short s )  { store(name,Short(s));}
  void store( const string &name, const int s )    { store(name,Int(s));}
  void store( const string &name, const float s )  { store(name,Float(s));}
  void store( const string &name, const double s ) { store(name,Double(s));}
  
  template<class T> bool get( const string &, T &);

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

private:
  template<class T> bool get_scalar( const string &, T &);

  typedef map<string, PropertyBase *> map_type;

  map_type properties_;
};


template<class T>
void 
PropertyManager::store( const string &name, const T& obj )
{
  T *t = new T(obj);

  store( name, t );
}

template<class T>
void 
PropertyManager::store( const string &name,  T& obj )
{
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) 
    delete loc->second;
    
  properties_[name] = new Property<T>( obj );
}

  

template<class T>
bool 
PropertyManager::get_scalar(const string &name, T &ref)
{
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    if ( dynamic_cast<Scalar *>( loc->second ) ) {
      ref = T(*static_cast<Scalar *>(loc->second->obj_));
      return true;
    }
  }
  
  // either property not found, or it can not be cast to T
  return false;
}

template<class T>
bool 
PropertyManager::get(const string &name, T &ref)
{
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    if ( dynamic_cast<T *>( loc->second ) ) {
      ref = *static_cast<T *>(loc->second->obj_);
      return true;
    }
  }
  
  // either property not found, or it can not be cast to T
  return false;
}

template<> bool PropertyManager::get(const string &name, char &ref);
template<> bool PropertyManager::get(const string &name, short &ref);
template<> bool PropertyManager::get(const string &name, int &ref);
template<> bool PropertyManager::get(const string &name, float &ref);
template<> bool PropertyManager::get(const string &name, double &ref);

} // namespace SCIRun

#endif // SCI_project_PropertyManager_h
