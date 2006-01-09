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
 *  GuiCallback.h: Interface to user interface
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCIRun_Core_GuiInterface_GuiCallback_h
#define SCIRun_Core_GuiInterface_GuiCallback_h

#include <Core/GuiInterface/TCLTask.h> // for TCLCONST

#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/GuiInterface/share.h>

namespace SCIRun {

  using std::string;
  using std::vector;

  class GuiVar;

  class SHARE GuiArgs {
    vector<string> args_;
  public:
    bool have_error_;
    bool have_result_;
    string string_;
    
    GuiArgs(int argc, TCLCONST char* argv[]);
    ~GuiArgs();
    int count();
    string operator[](int i);
    string get_string(int i);
    int get_int(int i);
    double get_double(int i);
    
    void error(const string&);
    void result(const string&);
    void append_result(const string&);
    void append_element(const string&);

    static string make_list(const string&, const string&);
    static string make_list(const string&, const string&, const string&);
    static string make_list(const vector<string>&);
  };

  class SHARE GuiCallback {
  public:
    GuiCallback();
    virtual ~GuiCallback();
    virtual void tcl_command(GuiArgs&, void*)=0;
  };
} // End namespace SCIRun


#endif
