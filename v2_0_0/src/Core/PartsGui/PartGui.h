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


/*
 *  PartGui.h
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Sep 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_PartGui_h
#define SCI_PartGui_h 

#include <sgi_stl_warnings_off.h>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TclObj.h>
#include <Core/Util/Signals.h>
#include <Core/Parts/PartInterface.h>

namespace SCIRun {
  
class PartGui;
class GuiCreatorBase;


class GuiCreatorBase {
public:
  GuiCreatorBase(const string &name);
  virtual ~GuiCreatorBase() {}

  virtual PartGui *create( const string & ) = 0;
};

template<class T>
class GuiCreator : public GuiCreatorBase {
public:
  GuiCreator( const string &name ) : GuiCreatorBase( name ) {}

  virtual PartGui *create( const string &name) { return scinew T(name); }
};


class PartGui : public TclObj {
public:
  static map<string,GuiCreatorBase*> table;

  Signal3<int, const string &, vector<unsigned char> > set_property;
  Signal3<int, const string &, vector<unsigned char> & > get_property;

protected:
  string name_;
  int n_;

public:
  PartGui( const string &name, const string &script ) 
    : TclObj(script), name_(name), n_(0) {}
  virtual ~PartGui() {}

  string name() { return name_; }

  virtual void add_child( PartInterface *child );
  virtual void attach( PartInterface *interface ) 
  { 
    connect(set_property, interface, &PartInterface::set_property);  
    connect(get_property, interface, &PartInterface::get_property);
  }
};

} // namespace SCIRun

#endif // SCI_PartGui_h
