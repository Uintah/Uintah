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
    GuiContext(GuiInterface* ctx, 
	       const string& name, 
	       bool save=true,
	       GuiContext *parent = 0);
    ~GuiContext();

    GuiInterface*	getInterface();
    GuiContext*		subVar(const string& name, bool save=true);

  //    

    void		lock();
    void		unlock();
  
    void		dontCache(); // always query GUI for value
    void		doCache();  // only query GUI if not cached already
    void		reset(); // resets the cache

    bool		get(string& value);
    void		set(const string& value);

    bool		get(double& value);
    void		set(double value);

    bool		get(int& value);
    void		set(int value);

    string		getfullname();

    string		getName();
    void		setName(const string&);

    void		tcl_setVarStates();  
  
    void		dontSave();
    void		doSave();
    
    void		setUseDatadir(bool flag);
  private:  
  //    void		erase(const string& subname);

    bool		setType();
  
    string		getPrefix();

    string		getMapKeyFromString(const string &); 
    string		getMapNameFromString(const string &str);
  
    int			popLastListIndexFromString(string &str);

    bool		stringIsaMap(const string &str);
    bool		stringIsaListElement(const string &str);

    bool		getString(const string &varname, string &value);
    bool		setString(const string &varname, const string &value);

    GuiInterface*	gui;
    GuiContext *	parent;
    string		name;
    vector<GuiContext*> children;

    enum  {
      SAVE_E			= 1 << 0,
      CACHE_E			= 1 << 1,
      CACHED_E			= 1 << 2,
      SUBSTITUTE_DATADIR_E	= 1 << 3//,
      //      TCL_VARIABLE_E		= 1 << 4,
      //      TCL_LIST_ELEMENT_E	= 1 << 5,
      //TCL_ARRAY_ELEMENT_E	= 1 << 6,
      //TCL_ARRAY_ELEMENT_E	= 1 << 7
    };

    unsigned int	context_state;

    GuiContext(const GuiContext&);
    GuiContext& operator=(const GuiContext&);
};

}

#endif

