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

// AttribManager.h - Manage a collection of attributes
//
//  Written by:
//   Yarden Livnat
//   Department of Computer Science
//   University of Utah
//   Jan 2001
//
//  Copyright (C) 2001 SCI Institute

#ifndef SCI_project_AttirbManager_h
#define SCI_project_AttirbManager_h 1

#include <Core/Datatypes/Datatype.h>
#include <Core/Datatypes/Attrib.h>
#include <Core/Containers/Array1.h>

#include <functional>
#include <iostream>
#include <vector>
#include <string>
#include <map>

namespace SCIRun{

using std::vector;
using std::string;
using std::map;

typedef map<string, AttribHandle> AttribMap;
typedef Array1<AttribHandle>      AttribVector;

class SCICORESHARE AttribManager: public Datatype{

public:

  // GROUP: Constructors/Destructor
  //////////
  //
  AttribManager();
  AttribManager(const AttribManager&);
  virtual ~AttribManager();

  //////////
  // Persistent representation...
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  void io_map(Piostream&);

  // GROUP: Member functions to manipulate attributes and geometries
  //////////
  // 

  //////////
  // Total number of registred attributes
  inline int getNumAttribs() const{
    return attribHandles_.size();
  }

  //////////
  // Returns handle to the current active attribute
  virtual const AttribHandle getAttrib() const;
  
  //////////
  // set attribute with the name supplied as active one
  void setCurrAttrib(string aName);
  
  //////////
  // Return attribute with the name supplied
  const AttribHandle getAttrib(string aName) const;
  
  //////////
  // Returns non-const handle to share the attribute for writing
  virtual AttribHandle shareAttrib(string aName);

  //////////
  // Adds attribute to the field and sets current attribute name to its name
  void addAttribute(const AttribHandle&);
  
  //////////
  // Removes attribute with particular order number from the field
  void removeAttribute(string);
  

protected:
  AttribMap       attribHandles_;
  string          currAttrib_;
};

} // end namespace SCIRun

#endif


