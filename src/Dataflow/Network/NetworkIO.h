//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : NetworkIO.h
//    Author : Martin Cole
//    Date   : Tue Jan 24 11:28:22 2006

#if !defined NetworkIO_h
#define NetworkIO_h

#include <Core/XMLUtil/XMLUtil.h>
#include <libxml/xmlreader.h>
#include <string>
#include <map>
#include <stack>

#include <Dataflow/Network/share.h>
namespace SCIRun {

using std::string;
using std::map;
using std::stack;

class SCISHARE NetworkIO {
public:
  NetworkIO() : 
    doc_(0),
    out_fname_(""),
    sn_count_(0),
    sn_ctx_(0)
  {
    netid_to_modid_.push(id_map_t());
    netid_to_conid_.push(id_map_t());
  }
  virtual ~NetworkIO() {}
  static void load_net(const string &net);
  static bool done_writing() { return done_writing_; }
  static bool has_file() { return net_file_ != string(""); }
  static bool autoview_pending() { return autoview_pending_; }
  static void clear_autoview_pending() { autoview_pending_ = false; }
  bool load_network();

  //! Interface to build up an xml document for saving.
  void start_net_doc(const string &fname, const string &vers);
  void write_net_doc();

  void add_net_var(const string &var, const string &val);
  void add_environment_sub(const string &var, const string &val);
  void add_net_note(const string &val);
  void add_module_node(const string &id, const string &pack, 
		       const string &cat, const string &mod);
  void add_module_position(const string &id, const string &x, 
			   const string &y);
  void add_module_note(const string &id, const string &note); 
  void add_module_note_position(const string &id, const string &pos); 
  void add_module_note_color(const string &id, const string &col); 
  void add_module_variable(const string &id, const string &var, 
			   const string &val, bool filename = false, bool substitute = false, bool userelfilenames = false);
  void set_module_gui_visible(const string &id);
  void add_module_gui_callback(const string &id, const string &call);

  void add_connection_node(const string &id, const string &fmod, 
			   const string &fport, const string &tmod, 
			   const string &tport);
  void set_disabled_connection(const string &id); 
  void add_connection_route(const string &id, const string &route); 
  void add_connection_note(const string &id, const string &note); 
  void add_connection_note_position(const string &id, const string &pos); 
  void add_connection_note_color(const string &id, const string &col); 
  void set_port_caching(const string &id, const string &port, 
			const string &val);
  void push_subnet_scope(const string &id, const string &name);
  void pop_subnet_scope();

private:
  void process_environment(const xmlNodePtr enode);
  void process_modules_pass1(const xmlNodePtr enode);
  void process_modules_pass2(const xmlNodePtr enode);
  void process_connections(const xmlNodePtr enode);
  void process_network_node(const xmlNodePtr nnode);
 
	string process_filename(const string &src);
  string process_substitute(const string &src);
	string make_absolute_filename(string name);
	string make_relative_filename(string name,string path);
	
  inline
  string get_mod_id(const string &id); 

  //! Interface from xml reading to tcl.
  //! this could be virtualized and used to interface with another gui type.
  void gui_add_module_at_position(const string &mod_id, 
				  const string &package, 
				  const string &category, 
				  const string &module, 
				  const string &x, 
				  const string &y);
  
  void gui_add_connection(const string &con_id,
			  const string &from_id, const string &from_port,
			  const string &to_id, const string &to_port);

  void gui_set_connection_disabled(const string &con_id);
  void gui_set_module_port_caching(const string &mid, const string &pid,
			       const string &val);

  void gui_call_module_callback(const string &id, const string &call);

  void gui_set_variable(const string &var, const string &val);
  void gui_set_modgui_variable(const string &mod_id, const string &var, 
			   const string &val);
  void gui_set_module_note(const string &mod_id, const string &pos, 
		       const string &col, const string &note);
  void gui_set_connection_note(const string &mod_id, const string &pos, 
			   const string &col, const string &note);
  void gui_set_connection_route(const string &con_id, const string &route);
  void gui_open_module_gui(const string &mod_id);

  void gui_add_subnet_at_position(const string &mod_id, 
				  const string &module, 
				  const string& x, 
				  const string &y);
  string gui_push_subnet_ctx();
  void gui_pop_subnet_ctx(string ctx);

  xmlNode* get_module_node(const string &id);
  xmlNode* get_connection_node(const string &id);

  typedef map<string, string> id_map_t;

  stack<id_map_t> netid_to_modid_; 
  stack<id_map_t> netid_to_conid_; 
  //! the enviroment variable substitutions
  map<string, string> env_subs_; 
  static string net_file_;
  static bool done_writing_;
  static bool autoview_pending_;

  //! document for writing nets.
  xmlDocPtr                          doc_;  
  stack<xmlNodePtr>                  subnets_;
  string                             out_fname_;
  int                                sn_count_;
  int                                sn_ctx_;
};

} // end namespace SCIRun

#endif //NetworkIO_h

