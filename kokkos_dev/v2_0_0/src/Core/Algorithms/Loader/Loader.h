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

/* Loader.h
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

#ifndef Loader_h
#define Loader_h 

#include <sgi_stl_warnings_off.h>
#include <map>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

using namespace std;

/*
 * Loadable
 */

class LoadableBase {
public:
  void *obj_;
  virtual ~LoadableBase() {}
};

template<class T>
class Loadable: public T, public LoadableBase {
public:
  Loadable( T *o) { obj_ = o; } 
};


/*
 * Loader
 */

class Loader 
{
public:
  Loader() {}
  Loader(const Loader &) {}
  virtual ~Loader() {}

  
  template<class T> void store( const string &, T *);
  template<class T> bool get( const string &, T *&);

private:
  typedef map<string, LoadableBase *> map_type;

  map_type objects_;
};


template<class T>
void 
Loader::store( const string &name,  T * obj )
{
  map_type::iterator loc = objects_.find(name);
  if (loc != objects_.end()) 
    delete loc->second;
    
  objects_[name] = new Loadable<T>( obj );
}

  
template<class T>
bool 
Loader::get(const string &name, T *&ref)
{
  map_type::iterator loc = objects_.find(name);
  if (loc != objects_.end()) {
    if ( dynamic_cast<T *>( loc->second ) ) {
      ref = static_cast<T *>(loc->second->obj_);
      return true;
    }
  }
  
  // either property not found, or it can not be cast to T
  return false;
}

} // namespace SCIRun

#endif // SCI_project_Loader_h
