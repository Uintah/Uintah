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

#include <Core/share/share.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {

  using std::string;
  using std::vector;

  class GuiVar;

  class SCICORESHARE GuiArgs {
    vector<string> args_;
  public:
    bool have_error_;
    bool have_result_;
    string string_;
    
    GuiArgs(int argc, char* argv[]);
    ~GuiArgs();
    int count();
    string operator[](int i);
  
    void error(const string&);
    void result(const string&);
    void append_result(const string&);
    void append_element(const string&);

    static string make_list(const string&, const string&);
    static string make_list(const string&, const string&, const string&);
    static string make_list(const vector<string>&);
  };

  class SCICORESHARE GuiCallback {
  public:
    GuiCallback();
    virtual ~GuiCallback();
    virtual void tcl_command(GuiArgs&, void*)=0;
  };
} // End namespace SCIRun


#endif
