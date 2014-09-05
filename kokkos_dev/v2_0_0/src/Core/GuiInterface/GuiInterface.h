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
 *  GuiInterface.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Core_GuiInterface_GuiInterface_h
#define SCIRun_Core_GuiInterface_GuiInterface_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  using namespace std;
  class GuiCallback;
  class GuiContext;
  class GuiInterface {
  public:
    virtual ~GuiInterface();
    virtual void execute(const string& str) = 0;
    virtual int eval(const string& str, string& result) = 0;
    virtual void source_once(const string&) = 0;
    virtual void add_command(const string&, GuiCallback*, void*) = 0;
    virtual void delete_command( const string& command ) = 0;
    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual GuiContext* createContext(const string& name) = 0;
    virtual void postMessage(const string& errmsg, bool err = false) = 0;
    virtual bool get(const std::string& name, std::string& value) = 0;
    virtual void set(const std::string& name, const std::string& value) = 0;
    static GuiInterface* getSingleton();
  protected:
    GuiInterface();
  private:
    GuiInterface(const GuiInterface&);
    GuiInterface& operator=(const GuiInterface&);
  };
}

#endif

