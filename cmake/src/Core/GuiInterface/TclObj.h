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
class TclObj : public GuiCallback {
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
