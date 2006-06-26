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
