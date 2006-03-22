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
 *  TCLInterface.h:
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Core_GuiInterface_TCLInterface_h
#define SCIRun_Core_GuiInterface_TCLInterface_h

#include <Core/GuiInterface/GuiInterface.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

#include <Core/GuiInterface/share.h>

namespace SCIRun {
  using namespace std;
  class Semaphore;
  class TCLInterface;

  class SCISHARE EventMessage {
  public:
    EventMessage();
    virtual ~EventMessage();
    void                wait_for_message_delivery();
    void                mark_message_delivered();
  private:
    Semaphore *         delivery_semaphore_;
  };

  class SCISHARE PauseEventMessage : public EventMessage {
  public:
    PauseEventMessage(TCLInterface *);
    TCLInterface *      tcl_interface_;
  };

  class SCISHARE CommandEventMessage : public EventMessage {
  public:
    CommandEventMessage(const string &command);
    string &            command() { return command_; }
    string &            result() { return result_; }
    int &               code() { return return_code_; } 
  private:
    string              command_;
    string              result_;
    int                 return_code_;
  };


  class SCISHARE TCLInterface : public GuiInterface{
    Semaphore *         pause_semaphore_;
    bool                paused_;
  public:
    TCLInterface();
    virtual ~TCLInterface();
    virtual void execute(const string& str);
    virtual int eval(const string& str, string& result);
    virtual string eval(const string &);
    virtual void pause();
    virtual void real_pause();
    virtual void unpause();
    virtual void source_once(const string&);
    virtual void add_command(const string&, GuiCallback*, void*);
    virtual void delete_command( const string& command );
    virtual void lock();
    virtual void unlock();
    virtual GuiContext* createContext(const string& name);
    virtual void post_message(const string& errmsg, bool err = false);
    // Get TCL array var, which resembles a STL a map<string, string>
    virtual bool get(const std::string& name, std::string& value);
    virtual void set(const std::string& name, const std::string& value);
    // Get TCL array var, which resembles a STL a map<string, string>
    virtual bool get_map(const std::string& name, 
			 const std::string &key,
			 std::string& value);
    virtual bool set_map(const std::string& name, 
			 const std::string &key,
			 const std::string& value);

    // Get an element of regular tcl list
    virtual bool extract_element_from_list(const std::string& list_contents, 
					   const vector<int> &indexes, 
					   std::string& value);
    virtual bool set_element_in_list(std::string& list_contents, 
				     const vector<int> &indexes, 
				     const std::string& value);

    bool complete_command(const string&);


  };
}

#endif

