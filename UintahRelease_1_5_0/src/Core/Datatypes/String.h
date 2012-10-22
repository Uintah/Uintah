/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


/*
 *  String.h:  String Object
 *
 *  Written by:
 *   Jeroen Stinstra
 *   Department of Computer Science
 *   University of Utah
 *   October 2005
 *
 */

#ifndef CORE_DATATYPES_STRING_H
#define CORE_DATATYPES_STRING_H 1

#include <Core/Containers/LockingHandle.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/PropertyManager.h>

#include <string>

#include <Core/Datatypes/share.h>

namespace SCIRun {

class String;
typedef LockingHandle<String> StringHandle;

class SCISHARE String : public PropertyManager {

  std::string str_;

public:
  //! Constructors
  String();
  String(const std::string& str);
  String(const String& str);
  String(const char* str);

  //! Destructor
  virtual ~String();
  
  //! Public member functions
  String* clone();
  inline void        set(std::string str);
  inline std::string get();
  inline void        setstring(std::string str);
  inline std::string getstring();

  //! Persistent representation...
  virtual string type_name() { return "String"; }
  virtual void io(Piostream&);
  static PersistentTypeID type_id;
  
};


inline void String::set(std::string str)
{
  str_ = str;
}

inline std::string String::get()
{
  return(str_);
}
inline void String::setstring(std::string str)
{
  str_ = str;
}

inline std::string String::getstring()
{
  return(str_);
}
    
} // End namespace SCIRun

#endif
