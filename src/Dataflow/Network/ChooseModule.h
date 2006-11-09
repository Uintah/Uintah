/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  ChooseModule.cc: Choose one input module to be passed downstream
 *
 *  Written by:
 *   Allen Sanderson
 *   Department of Computer Science
 *   University of Utah
 *   March 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/Handle.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <iostream>

namespace SCIRun {

template< class HANDLE_TYPE >
class ChooseModule : public Module {

public:
  ChooseModule(const std::string& module_name,
	       GuiContext* ctx, SchedClass,
	       const string& catagory  = "unknown",
	       const string& package   = "unknown",
	       const string& port_name = "unknown" );

  virtual ~ChooseModule();
  virtual void execute();

  virtual std::string get_names( std::vector< HANDLE_TYPE >& handles,
				 std::string prop_name = string("name") );

private:
  string port_name_;

  GuiInt gui_use_first_valid_;
  GuiInt gui_port_valid_index_;
  GuiInt gui_port_selected_index_;

  bool gui_is_built_;
  HANDLE_TYPE output_handle_;
};

template< class HANDLE_TYPE >
ChooseModule< HANDLE_TYPE >::ChooseModule(const std::string& module_name,
					  GuiContext* ctx, SchedClass,
					  const string& catagory,
					  const string& package,
					  const string& port_name )
  : Module(module_name, ctx, Filter, catagory, package),
    port_name_(port_name),
    gui_use_first_valid_(get_ctx()->subVar("use-first-valid"), 1),
    gui_port_valid_index_(get_ctx()->subVar("port-valid-index"), 0),
    gui_port_selected_index_(get_ctx()->subVar("port-selected-index"), 0),
    gui_is_built_(false),
    output_handle_(0)
{
}

template< class HANDLE_TYPE >
ChooseModule< HANDLE_TYPE >::~ChooseModule()
{
}

template< class HANDLE_TYPE >
std::string
ChooseModule< HANDLE_TYPE >::get_names( std::vector< HANDLE_TYPE >& handles,
					std::string prop_name)
{
  std::string port_names("");
  std::string prop;

  for( unsigned int idx = 0; idx<handles.size(); ++idx ) {

    port_names += "--Port_" + to_string(idx) + "_";

    if( handles[idx].get_rep() ) {
      if( (handles[idx]->get_property(prop_name, prop )) ) {
	port_names += prop;

      } else {
	port_names += "UNKNOWN";
      }
    } else {
      port_names +=  "INVALID_OR_DISABLED";
    }

    port_names +=  "-- ";
  }
  return port_names;
}

template< class HANDLE_TYPE >
void
ChooseModule< HANDLE_TYPE >::execute()
{

  std::vector< HANDLE_TYPE > handles;
  string port_names("");
  string visible;

  if( !get_dynamic_input_handles( port_name_, handles, true ) ) return;


  if( !gui_is_built_) { // build gui first time through
    gui_->eval(id_ + " isVisible", visible);
    if( visible == "0" ){
      gui_->execute(id_ + " build_top_level");
      gui_is_built_ = true;
    }
  }

  // Check to see if any values have changed.
  if( inputs_changed_ ||
      
      !output_handle_.get_rep() ||

      gui_use_first_valid_.changed( true ) ||

      (gui_use_first_valid_.get() == 1 ) ||      
      (gui_use_first_valid_.get() == 0 && 
       gui_port_selected_index_.changed( true )) ||

      execute_error_ ) {

    update_state(Executing);

    execute_error_ = false;

    port_names += get_names( handles );
    gui_->eval(id_ + " isVisible", visible);
    if( visible == "1"){
      gui_->execute(id_ + " destroy_frames");
      gui_->execute(id_ + " build_frames");
      gui_->execute(id_ + " set_ports " + port_names.c_str()); 
      gui_->execute(id_ + " build_port_list");
    }
  
    // use the first valid field
    if (gui_use_first_valid_.get()) {

      unsigned int idx = 0;  // reset
      while( idx < handles.size() && !handles[idx].get_rep() ) idx++;

      if( idx < handles.size() && handles[idx].get_rep() ) {
	output_handle_ = handles[idx];

	gui_port_valid_index_.set( idx );
	gui_port_valid_index_.reset();

      } else {
	error("Did not find any valid fields.");

	execute_error_ = true;
	return;
      }

    } else {
      // use the index specified
      int idx = gui_port_selected_index_.get();

      if ( 0 <= idx && idx < (int) handles.size() ) {
	if( handles[idx].get_rep() ) {

	  output_handle_ = handles[idx];

	} else {

	  output_handle_ = 0;
	  
	  error( "Port " + to_string(idx) + " did not contain a valid field.");
	  execute_error_ = true;
	  return;
	}

      } else {
	error("Selected port index out of range.");
	execute_error_ = true;
	return;
      }
    }
  }

  // Send the data downstream
  send_output_handle( port_name_, output_handle_, true );
}

} // End namespace SCIRun

