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
