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
 *  GuiContext.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Core_GuiInterface_GuiContext_h
#define SCIRun_Core_GuiInterface_GuiContext_h

#include <sgi_stl_warnings_off.h>
#include <string>
#include <vector>
#include <iosfwd>
#include <sgi_stl_warnings_on.h>

namespace SCIRun {
  class GuiInterface;
  class GuiContext {
  public:
    GuiContext(GuiInterface* ctx, const std::string& name, bool save=true);

    GuiContext* subVar(const std::string& name, bool save=true);
    void erase( const std::string& subname );

    void lock();
    void unlock();
    bool get(std::string& value);
    bool getSub(const std::string& name, std::string& value);
    void set(const std::string& value);
    void setSub(const std::string& name, const std::string& value);

    bool get(double& value);
    bool getSub(const std::string& name, double& value);
    void set(double value);
    void setSub(const std::string& name, double value);

    bool get(int& value);
    bool getSub(const std::string& name, int& value);
    void set(int value);
    void setSub(const std::string& name, int value);
    void reset();
    void emit(std::ostream& out, const std::string& midx);

    GuiInterface* getInterface();
    std::string getfullname();
    void dontSave();
  private:
    std::string format_varname();
    GuiInterface* gui;
    std::string name;
    std::vector<GuiContext*> children;
    bool cached;
    bool save;

    GuiContext(const GuiContext&);
    GuiContext& operator=(const GuiContext&);
  };
}

#endif

