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

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Datatypes/TypeName.h>
#include <Core/Datatypes/builtin.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Persistent/PersistentSTL.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <map>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using namespace std;

class PropertyBase : public Datatype {
public:
  PropertyBase(bool trans) : transient_(trans) {} 
  //PropertyBase(const PropertyBase &p) : transient_(p.transient_) {} 
  virtual PropertyBase* clone() const { 
    ASSERTFAIL("PropertyBase clone called");
  }

  virtual void io(Piostream &) {}
  static  PersistentTypeID type_id;

  bool transient() const { return transient_; }
  void set_transient(bool t) { transient_ = t; }

  virtual bool operator==(PropertyBase &/*pb*/) const {
    ASSERTFAIL( "Not defined."); }
  virtual bool operator!=(PropertyBase &/*pb*/) const {
    ASSERTFAIL( "Not defined."); }

protected:
  //! Transient properties are deleted when the PropertyManager that this
  //! Property belongs to is thawed. 
  bool transient_;
  static Persistent *maker();
};


class PropertyManager;

template<class T>
class Property : public PropertyBase {
public:
  friend class PropertyManager;

  Property(const T &o, bool trans) :  PropertyBase(trans), obj_(o) 
  {
  }
  //Property(const Property &p) :  PropertyBase(p), obj_(p.obj_) {}
  virtual PropertyBase *clone() const 
  { return scinew Property(obj_, transient()); }

  static const string type_name(int n = -1);
  virtual void io(Piostream &stream);
  static  PersistentTypeID type_id;

  virtual bool operator==(PropertyBase &pb) const {
    const Property<T> *prop = dynamic_cast<Property<T> *>(&pb);

    if (prop && obj_ == prop->obj_ )
      return true;

    return false;
  }

  virtual bool operator!=(PropertyBase &pb) const {
    const Property<T> *prop = dynamic_cast<Property<T> *>(&pb);

    if (prop && obj_ == prop->obj_ )
      return false;

    return true;
  }

protected:
  // Only Pio should use this constructor.
  // Default is for objects read in to be non-transient.
  Property() : PropertyBase(false) {}

private:
  T obj_;

  static Persistent *maker();
};



/*
 * Persistent Io
 */

const int PROPERTY_VERSION = 2;

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
  // Properties in a file start out to be non-transient.
  return scinew Property<T>();
}

template<class T>
void
Property<T>::io( Piostream &stream)
{
  const int version = stream.begin_class( type_name(-1), PROPERTY_VERSION);
  if (version > 1)
  {
    Pio(stream, transient_);
  }
  //else
  //{
    //cout << "Warning: Possible bad transient flag in property '"
    //<< type_name(-1) << "'\n";
  //}
  Pio(stream, obj_);
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

  PropertyManager & operator=(const PropertyManager &pm);
  bool operator==(const PropertyManager &pm);
  bool operator!=(const PropertyManager &pm);
  
  template<class T> void set_property(const string &, const T &,
				      bool is_transient);
  template<class T> bool get_property( const string &, T &);


  //! -- mutability --

  //! Transient data may only be stored in a frozen PropertyManager.
  virtual void freeze();
  
  //! thaw will remove all transient properties from the PropertyManager.
  virtual void thaw();
  
  //! query frozen state of a PropertyManager.
  bool is_frozen() const { return frozen_; }

  void remove_property( const string & );
  //int size() { return size_; }

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

private:
  //template<class T> bool get_scalar( const string &, T &);

  typedef map<string, PropertyBase *> map_type;

  int size_;
  map_type properties_;

protected:
  //! A frozen PropertyManager may store transient data.
  void clear_transient();

  bool frozen_;
};


template<class T>
void 
PropertyManager::set_property(const string &name,  const T& obj,
			      bool is_transient)
{
  if (is_transient && (! is_frozen())) {
    cerr << "WARNING::PropertyManager must be frozen to store transient data" 
	 <<" freezing now!" << std::endl;
    freeze();
  }
  lock.lock();
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) 
    delete loc->second;
  else
    size_++;
  properties_[name] = scinew Property<T>(obj, is_transient);
  lock.unlock();
}


template<class T>
bool 
PropertyManager::get_property(const string &name, T &ref)
{
  lock.lock();

  bool ans = false;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    const Property<T> *prop = dynamic_cast<Property<T> *>(loc->second);
    if (prop) {
      ref = prop->obj_;
      ans = true;
    }
  }
  
  lock.unlock();
  return ans;
} 


} // namespace SCIRun

#endif // SCI_project_PropertyManager_h
