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


