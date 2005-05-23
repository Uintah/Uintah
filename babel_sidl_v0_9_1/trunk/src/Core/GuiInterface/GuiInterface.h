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
#include <vector>
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
    virtual string eval(const string&) = 0;
    virtual void source_once(const string&) = 0;
    virtual void add_command(const string&, GuiCallback*, void*) = 0;
    virtual void delete_command( const string& command ) = 0;
    virtual void lock() = 0;
    virtual void unlock() = 0;
    virtual GuiContext* createContext(const string& name) = 0;
    virtual void postMessage(const string& errmsg, bool err = false) = 0;
    // Get regular var as string
    virtual bool get(const std::string& name, std::string& value) = 0;
    virtual void set(const std::string& name, const std::string& value) = 0;

    // Get TCL array var, which resembles a STL a map<string, string>
    virtual bool get_map(const std::string& name, 
			 const std::string &key,
			 std::string& value) = 0;
    virtual bool set_map(const std::string& name, 
			 const std::string &key,
			 const std::string& value) =0;

    // Get an element of regular tcl list
    virtual bool extract_element_from_list(const std::string& list_contents, 
					   const vector<int> &indexes, 
					   std::string& value) = 0;
    virtual bool set_element_in_list(std::string& list_contents, 
				     const vector<int> &indexes, 
				     const std::string& value) = 0;


    static GuiInterface* getSingleton();
  protected:
    GuiInterface();
  private:
    GuiInterface(const GuiInterface&);
    GuiInterface& operator=(const GuiInterface&);
  };
}

#endif

