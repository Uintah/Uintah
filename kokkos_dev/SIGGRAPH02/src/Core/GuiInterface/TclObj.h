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
 *  TclObj.h: C++ & TCL object
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   July 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#ifndef SCI_TclObj_h
#define SCI_TclObj_h 

#include <sstream>
#include <Core/GuiInterface/GuiCallback.h>

namespace SCIRun {
  class GuiInterface;
class SCICORESHARE TclObj : public GuiCallback {
public:
  std::ostringstream tcl_;
private:
  string id_;
  string window_;
  string script_;

  bool has_window_;
protected:
  GuiInterface* gui;
public:
  TclObj( GuiInterface* gui, const string &script);
  TclObj( GuiInterface* gui, const string &script, const string &id);
  virtual ~TclObj();

  bool has_window() { return has_window_; }
  string id() { return id_; }
  string window() { return window_; }
  std::ostream &to_tcl() { return tcl_; }
  void command( const string &s);
  int tcl_eval( const string &s, string &);
  void tcl_exec();

  virtual void set_id( const string &);
  virtual void set_window( const string&, const string &args, bool =true );
  virtual void set_window( const string&s ) { set_window(s,""); }
  virtual void tcl_command( GuiArgs &, void *) {}
};


} // namespace SCIRun

#endif // SCI_TclObj_h
