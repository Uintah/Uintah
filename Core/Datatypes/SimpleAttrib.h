//  SimpleAttrib.h
//  Written by:
//   Yarden Livnat
//   Department of Computer Science
//   University of Utah
//   Dec 2000
//  Copyright (C) 2000 SCI Institute
//  Attribute containing a single value

#ifndef SCI_project_SimpleAttrib_h
#define SCI_project_SimpleAttrib_h 1

#include <vector>

#include <Core/Datatypes/Attrib.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/TypeName.h>
#include <iostream>
using namespace std;

namespace SCIRun {

class SimpleAttribBase : public Attrib 
{
public:
  SimpleAttribBase( const string& name, Attrib::Type type) 
    : Attrib(name, type) {}
  virtual ~SimpleAttribBase() {}
};

template <class T> class SimpleAttrib : public SimpleAttribBase 
{
public:
  SimpleAttrib( const string& name = string("SimpleAttrib"), 
		Attrib::Type type = Attrib::Normal);
  SimpleAttrib( const T& obj,const string& name = string("SimpleAtrib"), 
		Attrib::Type type = Attrib::Normal);
  
  SimpleAttrib(const SimpleAttrib& copy);

  virtual ~SimpleAttrib();

  // Implement begin()
  // Implement end()
  
  virtual string getInfo();
  virtual string getTypeName(int=0);

  void setObj( const T& obj ) { obj_ = obj; }
  T& getObj() { return obj_; }

  //////////
  // Persistent representation...
  virtual void io(Piostream &);
  static PersistentTypeID type_id;
  static string typeName(int);
  static Persistent* maker();

private:
  T obj_;
};


template <class T>
string SimpleAttrib<T>::typeName(int n=0){
  ASSERTRANGE(n, 0, 2);
  static string t1name    = findTypeName((T*)0);
  static string className = string("SimpleAttrib<") + t1name +">";
  
  switch (n){
  case 1:
    return t1name;
  default:
    return className;
  }
}

template <class T> Persistent*
SimpleAttrib<T>::maker(){
  return new SimpleAttrib<T>();
}



template <class T> 
PersistentTypeID SimpleAttrib<T>::type_id(SimpleAttrib<T>::typeName(0), 
					    Attrib::typeName(0), 
					    SimpleAttrib<T>::maker);

#define SIMPLEATTRIB_VERSION 1

template <class T> void
SimpleAttrib<T>::io(Piostream& stream)
{
  stream.begin_class(typeName(0).c_str(), SIMPLEATTRIB_VERSION);
  
  // -- base class PIO
  Attrib::io(stream);
  Pio(stream, obj_);
  
  stream.end_class();
}

//////////
// Constructors/Destructor

template <class T>
SimpleAttrib<T>::SimpleAttrib( const string& name, Attrib::Type type) :
  SimpleAttribBase(name, type)
{
}

template <class T>
SimpleAttrib<T>::SimpleAttrib( const T& obj, const string &name, 
			       Attrib::Type type )  
  : SimpleAttribBase(name, type), obj_(obj)
{
}


template <class T>
SimpleAttrib<T>::SimpleAttrib(const SimpleAttrib& copy) :
  SimpleAttribBase( copy ),
  obj_( copy.obj_ )
{
}

template <class T>
SimpleAttrib<T>::~SimpleAttrib()
{
}

template <class T> string
SimpleAttrib<T>::getInfo()
{
  ostringstream retval;
  retval <<
    "Name = " << name_ << '\n' <<
    "Type = SimpleAttrib" << '\n';
  return retval.str();
}


template <class T> string
SimpleAttrib<T>::getTypeName(int n){
  return typeName(n);
}
  
} // End namespace SCIRun

#endif
