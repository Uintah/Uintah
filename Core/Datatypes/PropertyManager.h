/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

/* PropertyManager.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
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
#include <iostream>
#include <map>

#include <Core/Datatypes/share.h>

namespace SCIRun {


class SCISHARE PropertyBase : public Datatype {
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


class SCISHARE PropertyManager;

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

  static const std::string type_name(int n = -1);
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
const std::string Property<T>::type_name(int n)
{
  if ( n == -1 ) {
    static const std::string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;
  }
  else if ( n == 0 ) {
    static const std::string nm("Property");
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
  void copy_properties(const PropertyManager *src);

  bool operator==(const PropertyManager &pm);
  bool operator!=(const PropertyManager &pm);
  
  template<class T> void set_property(const std::string &, const T &,
				      bool is_transient);
  template<class T> bool get_property( const std::string &, T &);
  bool is_property( const std::string & );
  std::string get_property_name( size_t index );


  //! -- mutability --

  //! Transient data may only be stored in a frozen PropertyManager.
  virtual void freeze();
  
  //! thaw will remove all transient properties from the PropertyManager.
  virtual void thaw();
  
  //! query frozen state of a PropertyManager.
  bool is_frozen() const { return frozen_; }

  void remove_property( const std::string & );
  size_t nproperties() const { return properties_.size(); }

  void    io(Piostream &stream);
  static  PersistentTypeID type_id;

  void set_name(std::string& name) 
    { set_property(std::string("name"),name,false); }
  std::string get_name()
    {  std::string name; if (get_property("name",name)) return (name); else return(std::string("")); }

private:

  typedef std::map<std::string, PropertyBase *> map_type;
  map_type properties_;

protected:
  //! A frozen PropertyManager may store transient data.
  void clear_transient();

  bool frozen_;
  Mutex pmlock_;
};


template<class T>
void 
PropertyManager::set_property(const std::string &name,  const T& obj,
			      bool is_transient)
{
  if (is_transient && (! is_frozen())) {
    std::cerr << "WARNING::PropertyManager must be frozen to store transient data" 
              <<" freezing now!" << std::endl;
    freeze();
  }
  pmlock_.lock();
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end())
  {
    delete loc->second;
  }
  properties_[name] = scinew Property<T>(obj, is_transient);
  pmlock_.unlock();
}


template<class T>
bool 
PropertyManager::get_property(const std::string &name, T &ref)
{
  pmlock_.lock();

  bool ans = false;
  map_type::iterator loc = properties_.find(name);
  if (loc != properties_.end()) {
    const Property<T> *prop = dynamic_cast<Property<T> *>(loc->second);
    if (prop) {
      ref = prop->obj_;
      ans = true;
    }
  }
  
  pmlock_.unlock();
  return ans;
} 


} // namespace SCIRun

#endif // SCI_project_PropertyManager_h
