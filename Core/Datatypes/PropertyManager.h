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
#include <Core/Containers/LockingHandle.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Datatypes/builtin.h>
#include <Core/Datatypes/Datatype.h>

namespace SCIRun {


class PropertyBase : public Datatype {
public:
  void *obj_;

  virtual ~PropertyBase() {}
  virtual void io(Piostream &) {}
  static  PersistentTypeID type_id;
private:
  static Persistent *maker();
};

template<class T>
class Property : public T, public PropertyBase {
public:
  Property() {}   // only Pio should use this constructor
  Property( T &o ) { obj_= &o; } 
  
  static const string type_name( int n = -1 );
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;
private:
  static Persistent *maker();
};

template<class T>
class Property<T *> : public Property<T> {
private:
  bool tmp;
public:
  Property( T *obj, bool temp=false ) : Property<T>( *obj ), tmp(temp) {}
  ~Property() { if (tmp) delete static_cast<T *>(obj_); }
};


/*
 * Persistent Io
 */

const int PROPERTY_VERSION = 1;

/*
 * Property<T>
 */

template<class T>
const string Property<T>::type_name(int n)
{
  if ( n == -1 ) {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if ( n == 0 ) {
    static const string nm("Property");
    return nm;
  }
  else
    return find_type_name( (T *) 0);
}

template <class T>
PersistentTypeID 
Property<T>::type_id(type_name(-1), "PropertyBase", maker);

template <class T>
Persistent*
Property<T>::maker()
{
  return scinew Property<T>;
}

template<class T>
void
Property<T>::io( Piostream &stream)
{
  stream.begin_class( type_name(-1), PROPERTY_VERSION);
  if ( stream.reading() ) {
    T *tmp = new T;
    Pio(stream, *tmp );
    obj_ = tmp;
  }
  else
    Pio(stream, *static_cast<T *>(obj_));
  stream.end_class();
}


/*
 * PropertyManager
 */

class PropertyManager : public Datatype
{
public:
  PropertyManager();
  PropertyManager(const PropertyManager &copy);
  virtual ~PropertyManager();

  
  template<class T> void store( const string &, const T &);
  template<class T> void store( const string &, T &); 

  void store( const string &name, const char *s )
  { store(name, *scinew string(s)); }
  void store( const string &name, const char s )
  { store(name, *scinew Char(s)); }
  void store( const string &name, const short s )
  { store(name, *scinew Short(s)); }
  void store( const string &name, const int s )
  { store(name, *scinew Int(s)); }
  void store( const string &name, const float s )
  { store(name, *scinew Float(s)); }
  void store( const string &name, const double s )
  { store(name, *scinew Double(s)); }

  template<class T> bool get( const string &, T &);
  template<class T> bool get( const string &, T *&);

  void remove( const string & );
  int size() { return size_; }

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

private:
  template<class T> bool get_scalar( const string &, T &);

  typedef map<string, PropertyBase *> map_type;

  int size_;
  map_type properties_;
};


template<class T>
void 
PropertyManager::store( const string &name,  T& obj)
{
  lock.lock();

  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) 
    delete loc->second;
  else
    size_++;
  properties_[name] = new Property<T>( obj );

  lock.unlock();
}

template<class T>
void 
PropertyManager::store( const string &name,  const T& obj )
{
  lock.lock();

  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) 
    delete loc->second;
  else
    size_++;
  properties_[name] = new Property<T*>( scinew T(obj), true );

  lock.unlock();

}

template<class T>
bool 
PropertyManager::get_scalar(const string &name, T &ref)
{
  lock.lock();

  bool ans = false;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    if ( dynamic_cast<Scalar *>( loc->second ) ) {
      ref = T(*static_cast<Scalar *>(loc->second->obj_));
      ans=true;
    }
  }
  
  lock.unlock();
  return ans;
}

template<class T>
bool 
PropertyManager::get(const string &name, T &ref)
{
  lock.lock();

  bool ans = false;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    if ( dynamic_cast<T *>( loc->second ) ) {
      ref = *static_cast<T *>(loc->second->obj_);
      ans = true;
    }
  }
  
  lock.unlock();
  return ans;
} 

template<class T>
bool 
PropertyManager::get(const string &name, T *&ref)
{
  lock.lock();

  bool ans = false;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    if ( dynamic_cast<T *>( loc->second ) ) {
      ref = static_cast<T *>(loc->second->obj_);
      ans=true;
    }
  }
  
  lock.unlock();

  return ans;
} 

template<> bool PropertyManager::get(const string &name, char &ref);
template<> bool PropertyManager::get(const string &name, short &ref);
template<> bool PropertyManager::get(const string &name, int &ref);
template<> bool PropertyManager::get(const string &name, float &ref);
template<> bool PropertyManager::get(const string &name, double &ref);

} // namespace SCIRun

#endif // SCI_project_PropertyManager_h
