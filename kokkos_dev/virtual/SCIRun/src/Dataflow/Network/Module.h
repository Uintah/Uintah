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
 *  Module.h: Classes for module ports
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   April 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#ifndef SCIRun_Dataflow_Network_Module_h
#define SCIRun_Dataflow_Network_Module_h

#include <Dataflow/Network/Ports/Port.h>
#include <Core/Util/Assert.h>
#include <Core/GeomInterface/Pickable.h>
#include <Core/Thread/RecursiveMutex.h>
#include <Core/Thread/Mailbox.h>
#include <Core/Thread/FutureValue.h>
#include <Dataflow/GuiInterface/GuiCallback.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/DynamicLoader.h>
#include <Core/Util/DynamicCompilation.h>
#include <Core/Util/Timer.h>
#include <Core/Util/ProgressReporter.h>
#include <Core/Containers/StringUtil.h>
#include <sstream>
#include <iosfwd>
#include <string>
#include <vector>
#include <map>

#include <Dataflow/Network/share.h>

#ifdef _WIN32
#define DECLARE_MAKER(name) \
extern "C" __declspec(dllexport) Module* make_##name(GuiContext* ctx) \
{ \
  return new name(ctx); \
}
#else
#define DECLARE_MAKER(name) \
extern "C" Module* make_##name(GuiContext* ctx) \
{ \
  return new name(ctx); \
}
#endif

// CollabVis code begin
// Stupid ViewServer hack
#ifdef HAVE_COLLAB_VIS
#undef ASIP_SHORT_NAMES
#endif
// End of hack
// CollabVis code end

namespace SCIRun {
using namespace std;
class IPort;
class OPort;
class BaseWidget;
class GuiContext;
class GuiInterface;
class MessageBase;
class Module;
class ModuleHelper;
class Network;
class Scheduler;

template<class T> class SimpleIPort;
template<class T> class SimpleOPort;

// Define Handles for thread safety.
typedef LockingHandle<IPort> IPortHandle;
typedef LockingHandle<OPort> OPortHandle;

typedef IPort* (*iport_maker)(Module*, const string&);
typedef OPort* (*oport_maker)(Module*, const string&);
typedef Module* (*ModuleMaker)(GuiContext* ctx);
typedef std::multimap<string,int> port_map_type;
typedef std::pair<port_map_type::iterator,
		  port_map_type::iterator> port_range_type;


// PortManager class has been altered to be thread safe
// so the module code as well as the network editor can
// update the number of ports.
// The PortManager will use Handle instead of pointers 
// as well so we can delete a port on the network
// while executing and not crash the network.

template<class T> 
class PortManager {
private:
  port_map_type    namemap_;
  vector<T>        ports_;
  RecursiveMutex   lock_;   // Thread safety
  
public:
  PortManager();
  int size();
  void add(const T &item);
  void remove(int item);
  T operator[](int);
  port_range_type operator[](string);
  T get_port(int);
  std::vector<T> get_port_range(std::string item);  // atomic version of getting all handles with a certain name

  void lock() { lock_.lock(); }
  void unlock() { lock_.unlock(); }
};

template<class T>
PortManager<T>::PortManager() :
  lock_("port manager lock")
{
}

template<class T>
int
PortManager<T>::size()
{ 
  lock_.lock();
  size_t s = ports_.size();
  lock_.unlock();
  return (static_cast<int>(s)); 
}

template<class T>
T
PortManager<T>::get_port(int item)
{
  lock_.lock();
  T handle(0);
  if (item < ports_.size()) 
  {
    handle = ports_[item];
  }
  lock_.unlock();
  return(handle);
}

template<class T>
void
PortManager<T>::add(const T &item)
{ 
  lock_.lock();
  namemap_.insert(pair<string, int>(item->get_portname(), ports_.size())); 
  ports_.push_back(item);
  lock_.unlock();
}

template<class T>
void
PortManager<T>::remove(int item)
{
  lock_.lock();
 
  if (ports_.size() <= item)
  {
    lock_.unlock();
    throw "PortManager tried to remove a port that does not exist";
  }

  string name = ports_[item]->get_portname();
  port_map_type::iterator erase_me;

  port_range_type p = namemap_.equal_range(name);
  for (port_map_type::iterator i=p.first;i!=p.second;i++)
    if ((*i).second>item)
      (*i).second--;
    else if ((*i).second==item)
      erase_me = i;
    
  ports_.erase(ports_.begin() + item);
  namemap_.erase(erase_me);
  lock_.unlock();  
}

template<class T>
T
PortManager<T>::operator[](int item)
{
  lock_.lock();

  if (ports_.size() <= static_cast<size_t>(item))
  {
    lock_.unlock();
    throw "PortManager tried to access a port that does not exist";
  }

  T t = ports_[item];
  lock_.unlock();
  return (t);

}

template<class T>
port_range_type
PortManager<T>::operator[](string item)
{
  lock_.lock();
  port_range_type prt = static_cast<port_range_type>(namemap_.equal_range(item));
  lock_.unlock();
  return (prt);
}

template<class T>
std::vector<T>
PortManager<T>::get_port_range(std::string item)
{
  std::vector<T> ports;
  lock_.lock();
  port_range_type range = static_cast<port_range_type>(namemap_.equal_range(item));
  port_map_type::iterator pi = range.first;
  while (pi != range.second)
  {
    ports.push_back(ports_[pi->second]);
    ++pi;
  }
  lock_.unlock();
  return (ports);
}


class SCISHARE Module : public ProgressReporter, public ModulePickable, 
			public GuiCallback
{
public:
  enum SchedClass {
    Sink,
    Source,
    Filter,
    Iterator,
    ViewerSpecial
  };
  
  Module(const std::string& name, GuiContext* ctx, SchedClass,
	 const string& cat="unknown", const string& pack="unknown");
  virtual ~Module();
  void kill_helper();

  //! Used by SCIRun2
  Network* get_network() {
    return network_;
  }
  
  bool have_ui();
  void popup_ui();
  void delete_warn();
  bool show_stats() { return show_stats_; }
  void set_show_stats(bool v) {show_stats_ = v;}

  //! ProgressReporter function
  virtual void error(const std::string&);
  virtual void warning(const std::string&);
  virtual void remark(const std::string&);
  virtual void compile_error(const std::string&);
  virtual void add_raw_message(const std::string&);
  
  // This one isn't as thread safe as the other ProgressReporter functions.
  // Use add_raw_message or one of the others instead if possible.
  virtual std::ostream &msg_stream() { return msg_stream_; }
  virtual void msg_stream_flush();

  //! This one should go when we have redone the PowerApps
  virtual bool in_power_app();

  //! Compilation progress.  Should probably have different name.
  virtual void report_progress( ProgressState );

  //! Execution time progress.
  //! Percent is number between 0.0-1.0
  virtual void          update_progress(double percent);
  virtual void          update_progress(int current, int max);
  virtual void          increment_progress();

  //! The next 12 functions are all deprecated:
  //! Do not use these as they are not thread safe and return
  //! a pointer instead of handle to an object they may be deleted
  //! by another thread.
  
  //--------------------------------------------------------------------
  // They are still here for compatibility of old modules
  //
  // Use get_input_handle(), get_dynamic_input_handles(), send_output_handle
  // instead. These have been made thread safe
  //--------------------------------------------------------------------
      port_range_type get_iports(const string &name);
      IPort* get_iport(const string &name);
      OPort* get_oport(const string &name);
      OPort* get_oport(int idx);
      IPort* get_iport(int idx);
      int num_input_ports();
      int num_output_ports();
      port_range_type get_input_ports(const string &name);
      IPort* get_input_port(const string &name);
      OPort* get_output_port(const string &name);
      OPort* get_output_port(int idx);
      IPort* get_input_port(int idx);
  //--------------------------------------------------------------------

  
  bool oport_cached(const string &name);
  bool oport_supports_cache_flag(int p);
  bool get_oport_cache_flag(int p);
  void set_oport_cache_flag(int p, bool val);

  //! Used by widgets
  GuiInterface* get_gui();

  //! Used by ModuleHelper
  void set_pid(int pid);
  virtual void do_execute();
  virtual void do_synchronize();
  virtual void execute() = 0;

  void request_multisend(OPort*);
  Mailbox<MessageBase*> mailbox_;
  virtual void set_context(Network* network);

  //! Callbacks
  
  virtual void connection(Port::ConnectionState, int, bool);
  virtual void widget_moved(bool last, BaseWidget *widget);

  void get_position(int& x, int& y);
  string get_id() const { return id_; }
  //! CollabVis code end
  
  template<class DC>
  bool module_dynamic_compile( CompileInfoHandle ci, DC &result )
  {
    return DynamicCompilation::compile( ci, result, this );
  }

  //!Used by Bridge stuff for SR2
  int add_input_port_by_name(std::string name, std::string d_type);
  int add_output_port_by_name(std::string name, std::string d_type);
  void want_to_execute();


  // Get handles to ports, thread safety issues with
  // network editing and execution at the same time
  
  template<class DP>
  bool get_iport_handle(const string &name, DP& handle);

  template<class DP>
  bool get_iport_handles(const string &name, std::vector<DP>& handles);
  
  template<class DP>
  bool get_oport_handle(const string &name, DP& handle);
  
  template<class DP>
  bool get_iport_handle(int portnum, DP& handle);
  
  template<class DP>
  bool get_oport_handle(int portnum, DP& handle);

  void lock_iports() { iports_.lock(); }
  void unlock_iports() { iports_.unlock(); }

  void lock_oports() { oports_.lock(); }
  void unlock_oports() { oports_.unlock(); }
  
protected:

  friend class ModuleHelper;
  friend class NetworkEditor;
  friend class PackageDB;
  friend class Network;
  friend class OPort;
  friend class IPort;
  friend class Scheduler;

  enum State {
    NeedData,
    JustStarted,
    Executing,
    Completed
  };
  enum MsgState {  
    Remark,
    Warning,
    Error,
    Reset
  };
  void update_state(State);
  void update_msg_state(MsgState);  
  virtual void tcl_command(GuiArgs&, void*);
  virtual void presave() {}
  GuiContext* get_ctx() { return ctx_; }
  void set_stack_size(unsigned long stack_size);
  void reset_vars();




  //! Get the handle for a single port.
  template<class DH>
  bool get_input_handle(std::string name, DH& handle, bool required = true);
  template<class DH>
  bool get_input_handle(int num, DH& handle, bool required = true);


  //! Get the handles for dynamic ports.
  template<class DH>
  bool get_dynamic_input_handles(std::string name, std::vector< DH > &handles,
				 bool required = true);

  //! cache_in_module should be true if the module is doing it's own
  //! caching of the handle.  Otherwise that will be handled by the
  //! port.
  template<class DH>
  bool send_output_handle(string name, DH& handle,
			  bool cache_in_module = false,
			  bool send_intermediate = false);

  template<class DH>
  bool send_output_handle(int numport, DH& handle,
			  bool cache_in_module = false,
			  bool send_intermediate = false);


  //! Specialization for geometry ports.
  bool send_output_handle(string port_name,
			  GeomHandle& handle,
			  string obj_name);

  bool send_output_handle(string port_name,
			  GeomHandle& handle,
			  const char *obj_name)
  {
    return send_output_handle(port_name, handle, string(obj_name));
  }

  bool send_output_handle(string port_name,
			  vector<GeomHandle>& handle,
			  vector<string>& obj_name);

  GuiInterface          *gui_;
  string                 module_name_;
  string                 package_name_;
  string                 category_name_;
  Scheduler             *sched_;
  bool                   lastportdynamic_;
  int                    pid_;
  bool                   have_own_dispatch_;
  string                 id_;
  bool                   abort_flag_;
  CPUTimer               timer_;
  ostringstream          msg_stream_;
  bool                   need_execute_;
  SchedClass             sched_class_;

  //! Used by the execute module;
  //! True if either the gui vars or data has changed.
  bool                   inputs_changed_;   
  //! Error during execution not related to checks.
  bool                   execute_error_;    

private:

  void remove_iport(int);
  void add_iport(IPort*);
  void add_oport(OPort*);

  GuiContext*            ctx_;
  State                  state_;
  MsgState               msg_state_;  
    		         
  PortManager<OPortHandle>  oports_;
  PortManager<IPortHandle>  iports_;
  iport_maker               dynamic_port_maker_;
  string                    lastportname_;
  int                       first_dynamic_port_;
  unsigned long             stacksize_;
  ModuleHelper              *helper_;
  Thread                    *helper_thread_;
  Network                   *network_;
		         
  bool                      show_stats_;
		         
  GuiString                 log_string_;

  RecursiveMutex            lock_;

  Module(const Module&);
  Module& operator=(const Module&);
};


//! Used to get handles with error checking.
template<class DH>
bool
Module::get_input_handle(std::string name,
			 DH& handle,
			 bool required)
{
//  update_state(NeedData);

  bool return_state = false;

  LockingHandle<SimpleIPort<DH> > dataport;

  handle = 0;

  //! We always require the port to be there.
  if (!(get_iport_handle(name,dataport)))
  {
    throw "Incorrect data type sent to input port '" + name +
      "' (dynamic_cast failed).";
  }
 
  //! Get the handle and check for data.
  else if (dataport->get(handle) && handle.get_rep())
  {
    //! See if the data has changed. Note only change the boolean if
    //! it is false this way it can be cascaded with other handle gets.
    if( inputs_changed_ == false ) {
      inputs_changed_ = dataport->changed();
    }
    //! we have a valid handle, return true.
    return_state = true;
  }

  else if( required )
  {
    //! The first input on the port was required to have a valid
    //! handle and data so report an error.
    error( "No handle or representation for input port '" +
           name + "'."  );
  }
  else
  {
    //! See if the data has changed. Note only change the boolean if
    //! it is false this way it can be cascaded with other handle gets.
    if( inputs_changed_ == false ) {
      inputs_changed_ = dataport->changed();
    }  
  }


  return return_state;
}


//! Used to get handles with error checking.
template<class DH>
bool
Module::get_input_handle(int num,
			 DH& handle,
			 bool required)
{
//  update_state(NeedData);

  bool return_state = false;

  LockingHandle< SimpleIPort<DH> > dataport;
  handle = 0;

  //! We always require the port to be there.
  if (!(get_iport_handle(num,dataport)))
  {
    std::ostringstream oss;
    oss << "port " << num;
    throw "Incorrect data type sent to input port '" + oss.str() +
      "' (dynamic_cast failed).";
  }
 
  //! Get the handle and check for data.
  else if (dataport->get(handle) && handle.get_rep())
  {
    //! See if the data has changed. Note only change the boolean if
    //! it is false this way it can be cascaded with other handle gets.
    if( inputs_changed_ == false ) {
      inputs_changed_ = dataport->changed();
    }
    //! we have a valid handle, return true.
    return_state = true;
  }

  else if( required )
  {
    std::string name = dataport->get_portname();
    //! The first input on the port was required to have a valid
    //! handle and data so report an error.
    error( "No handle or representation for input port '" +
           name + "'."  );
  }
  else
  {
    //! See if the data has changed. Note only change the boolean if
    //! it is false this way it can be cascaded with other handle gets.
    if( inputs_changed_ == false ) {
      inputs_changed_ = dataport->changed();
    }  
  }

  return return_state;
}




//! Used to get handles with error checking.
//! If valid handles are set in the vector, return true.
//! Only put valid handles in the vector.
template<class DH>
bool
Module::get_dynamic_input_handles(std::string name,
				  vector<DH> &handles,
				  bool data_required)
{
  bool return_state = false;

//  update_state(NeedData);

  unsigned int nPorts = 0;
  handles.clear();

  std::vector<LockingHandle<SimpleIPort<DH> > > dataports;
  
  if (!(get_iport_handles(name,dataports)))
  {
    throw "Unable to initialize dynamic input port '" + name + "'.";
  }
  else
  {
    for (size_t p=0; p < dataports.size(); p++)
    {
      //! Get the handle and check for data.
      DH handle;

      if (dataports[p]->get(handle) && handle.get_rep())
      {
        handles.push_back(handle);
  
        //! See if the data has changed. Note only change the boolean if
        //! it is false this way it can be cascaded with other handle gets.
        if( inputs_changed_ == false ) 
        {
          inputs_changed_ = dataports[p]->changed();
        }
        return_state = true;
      } 
      else 
      {
        if( inputs_changed_ == false ) 
        {
          inputs_changed_ = dataports[p]->changed();
        }
        handles.push_back(0);
      }
      ++nPorts;
    }
  }

  //! The last port is always empty so remove it.
  handles.pop_back();

  if (return_state == false) 
  {
    //! At least one port was required to have a valid ! handle and
    //data so report an error.
    if( data_required ) {
      error( "No handle or representation for dynamic input port #" +
      to_string(nPorts) + " ' " +name + "'." );
    }
    //! If we have no valid handles, make sure iteration over 
    //! the set of handles is empty.
    handles.clear();
  }

  return return_state;
}


//! Used to send handles with error checking.
template<class DH>
bool
Module::send_output_handle(string name, DH& handle,
			   bool cache, bool send_intermediate)
{
  //! Don't send on empty, assume cached version is more valid instead.
  //! Dunno if false return value is correct.  We don't check returns
  //! on this one.
  if (!handle.get_rep()) return false;

  LockingHandle<SimpleOPort<DH> > dataport;

  //! We always require the port to be there.
  if (!(get_oport_handle(name,dataport)))
  {
    throw "Incorrect data type sent to output port '" + name +
      "' (dynamic_cast failed).";
    return false;
  }

  if( send_intermediate )
    dataport->send_intermediate( handle );
  else
    dataport->send_and_dereference( handle, cache );

  return true;
}


template<class DH>
bool
Module::send_output_handle(int portnum, DH& handle,
			   bool cache, bool send_intermediate)
{
  //! Don't send on empty, assume cached version is more valid instead.
  //! Dunno if false return value is correct.  We don't check returns
  //! on this one.
  if (!handle.get_rep()) return false;

  LockingHandle<SimpleOPort<DH> > dataport;

  //! We always require the port to be there.
  if (!(get_oport_handle(portnum,dataport)))
  {
    std::ostringstream oss;
    oss << portnum;
    throw "Incorrect data type sent to output port number '" + oss.str() +
      "' (dynamic_cast failed).";
    return false;
  }

  if( send_intermediate )
    dataport->send_intermediate( handle );
  else
    dataport->send_and_dereference( handle, cache );

  return true;
}


template<class DP>
bool Module::get_iport_handle(const string &name, DP& handle)
{
  std::vector<IPortHandle> iports;
  iports = iports_.get_port_range(name);
  if (iports.size() == 0)
  {
    throw "Unable to initialize iport '" + name + "'.";  
  }
  handle = dynamic_cast<typename DP::pointer_type>(iports[0].get_rep());
  return (handle.get_rep());
}


template<class DP>
bool Module::get_iport_handles(const string &name, std::vector<DP>& handles)
{
  std::vector<IPortHandle> iports;
  iports = iports_.get_port_range(name);
  if (iports.size() == 0)
  {
    throw "Unable to initialize iport '" + name + "'.";  
  }
  
  DP handle;
  for (size_t p=0; p<iports.size();p++)
  {
    handle = dynamic_cast<typename DP::pointer_type>(iports[p].get_rep());
    if (handle.get_rep()) handles.push_back(handle);
  }

  return (handles.size());
}

template<class DP>
bool Module::get_oport_handle(const string &name, DP& handle)
{

  std::vector<OPortHandle> oports;
  oports = oports_.get_port_range(name);
  if (oports.size() == 0)
  {
    throw "Unable to initialize oport '" + name + "'.";  
  }
  handle = dynamic_cast<typename DP::pointer_type>(oports[0].get_rep());


  return (handle.get_rep());
}

template<class DP>
bool Module::get_iport_handle(int item, DP& handle)
{

  IPortHandle h = iports_[item];
  handle = dynamic_cast<typename DP::pointer_type>(h.get_rep());


  return (handle.get_rep());
}

template<class DP>
bool Module::get_oport_handle(int item, DP& handle)
{

  OPortHandle h = oports_[item];
  handle = dynamic_cast<typename DP::pointer_type>(h.get_rep());

  return (handle.get_rep());
}

}

#endif
