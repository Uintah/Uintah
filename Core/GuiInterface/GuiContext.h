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

using std::vector;
using std::string;
class GuiInterface;

class GuiContext {
  public:
    GuiContext(GuiInterface* ctx, const string& name, bool save=true);

    GuiContext*		subVar(const string& name, bool save=true);
    void		erase(const string& subname);

    void		lock();
    void		unlock();

    bool		get(string& value);
    void		set(const string& value);

    bool		get(double& value);
    void		set(double value);

    bool		get(int& value);
    void		set(int value);

    void		reset();
    void		emit(std::ostream& out, 
			     const string &midx,
			     const string& prefix="");

    GuiInterface*	getInterface();
    string		getfullname();

    void		dontSave();
    void		setUseDatadir(bool flag);
  private:
    string		format_varname();

    GuiInterface*	gui;
    string		name;
    vector<GuiContext*> children;
    bool		cached;
    bool		save;
    bool		usedatadir;

    GuiContext(const GuiContext&);
    GuiContext& operator=(const GuiContext&);
};

}

#endif

