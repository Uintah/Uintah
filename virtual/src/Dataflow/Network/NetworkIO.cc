//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//
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
//    File   : NetworkIO.cc
//    Author : Martin Cole
//    Date   : Mon Feb  6 14:32:15 2006

#include <Dataflow/Network/NetworkIO.h>
#include <Dataflow/Network/Network.h>
#include <Dataflow/Network/NetworkEditor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/GuiInterface/GuiInterface.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>
#include <Core/OS/Dir.h>
#include <libxml/catalog.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

string NetworkIO::net_file_= "";
bool NetworkIO::done_writing_= false;
bool NetworkIO::autoview_pending_= false;

inline 
string
NetworkIO::get_mod_id(const string& id)
{
  id_map_t &mmap = netid_to_modid_.top();
  const string sn("Subnet"); 
  return (id == sn) ? sn : mmap[id];
}

string 
NetworkIO::gui_push_subnet_ctx()
{
  GuiInterface *gui = GuiInterface::getSingleton();
  string cmmd = "set sn_ctx $Subnet(Loading)";
  string s = gui->eval(cmmd);
  return s;
}

void 
NetworkIO::gui_pop_subnet_ctx(string ctx)
{
  GuiInterface *gui = GuiInterface::getSingleton();
  string cmmd = "set Subnet(Loading) " + ctx;
  gui->eval(cmmd);

  --sn_ctx_;
  netid_to_modid_.pop();
  netid_to_conid_.pop();
}
void 
NetworkIO::gui_add_subnet_at_position(const string &mod_id, 
				      const string &module, 
				      const string& x, 
				      const string &y)
{
  GuiInterface *gui = GuiInterface::getSingleton();

  ++sn_count_;
  // map the subnet to a local var before we push maps.
  id_map_t &mmap = netid_to_modid_.top();

  ostringstream snic;
  snic << "SubnetIcon" << sn_count_;
  mmap[mod_id] = snic.str();

  netid_to_modid_.push(id_map_t());
  netid_to_conid_.push(id_map_t());

  ostringstream cmmd;

  cmmd << "set Subnet(Loading) [makeSubnetEditorWindow " 
       << sn_ctx_ << " " << x  << " " << y << "]";
  gui->eval(cmmd.str());
  ++sn_ctx_;


  ostringstream cmmd1;
  cmmd1 << "set Subnet(Subnet" << sn_count_ << "_Name) \"" << module << "\"";
  gui->eval(cmmd1.str());


}

void 
NetworkIO::gui_add_module_at_position(const string &mod_id, 
				      const string &package, 
				      const string &category, 
				      const string &module, 
				      const string& x, 
				      const string &y)
{
  // create the module.
  Module* mod = NetworkEditor::get_network()->add_module(package, 
							 category, 
							 module);

  // TODO: Fix crash bug here when package is not available.
  // Invoke nice gui for not loading this network rather than crash to
  // command line.
  if (!mod)
  {
    // add_module already outputs a message.
    Thread::exitAll(0);
  }

  // Now tell tcl about the module.
  GuiInterface *gui = GuiInterface::getSingleton();

  string cmmd = "addModuleAtAbsolutePosition " + package + " " + 
    category + " " + module + " " + x + " " + y + " " + mod->get_id();
  string mid = gui->eval(cmmd);
  id_map_t &mmap = netid_to_modid_.top();
  mmap[mod_id] = mid;
}

void 
NetworkIO::gui_add_connection(const string &con_id,
			  const string &from_id, 
			  const string &from_port,
			  const string &to_id, 
			  const string &to_port)
{
  string from = get_mod_id(from_id);
  string to = get_mod_id(to_id);
  string arg = "1";
  if (from.find("Subnet") == string::npos && 
      to.find("Subnet") == string::npos) 
  {
    arg = "0";
    // create the connection.
    Network *net = NetworkEditor::get_network();
    Module* omod = net->get_module_by_id(from);
    Module* imod = net->get_module_by_id(to);

    if (omod == 0 || imod == 0)
    {
      cerr << "Bad connection made, one or more modules not available.\n";
      return;
    }
  
    int owhich = atoi(from_port.c_str());
    int iwhich = atoi(to_port.c_str());
  
    net->connect(omod, owhich, imod, iwhich);
  }
  // Now tell tcl about the connection.
  GuiInterface *gui = GuiInterface::getSingleton();

  // tell tcl about the connection, last argument tells it not to creat the 
  // connection on the C side, since we just did that above.
  string cmmd = "createConnection [list " + from + " " + from_port +
    " " + to + " " + to_port + "] 0 " + arg;

  string cid = gui->eval(cmmd);
  id_map_t &cmap = netid_to_conid_.top();
  cmap[con_id] = cid;
}

void 
NetworkIO::gui_set_connection_disabled(const string &con_id)
{ 
  GuiInterface *gui = GuiInterface::getSingleton();
  
  id_map_t &cmap = netid_to_conid_.top();
  string con = cmap[con_id];
  string cmmd = "set Disabled(" + con + ") {1}";
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_module_port_caching(const string &mid, const string &pid,
				   const string &val)
{
  GuiInterface *gui = GuiInterface::getSingleton();
  
  string modid = get_mod_id(mid);
  string cmmd = "setPortCaching " + modid + " " + pid + " " + val;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_call_module_callback(const string &id, const string &call)
{
  GuiInterface *gui = GuiInterface::getSingleton();
  
  string modid = get_mod_id(id);
  string cmmd = modid + " " + call;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_modgui_variable(const string &mod_id, const string &var, 
			      const string &val)
{  
  GuiInterface *gui = GuiInterface::getSingleton();
  string cmmd;
  string mod = get_mod_id(mod_id);

  // Some variables in tcl are quoted strings with spaces, so in that 
  // case insert the module identifying string after the first quote.
  size_t pos = var.find_first_of("\"");
  if (pos == string::npos) {
    cmmd = "set " + mod + "-" + var +  " " + val;
  } else {
    string v = var;
    v.insert(++pos, mod + "-");
    cmmd = "set " + v +  " " + val;
  }
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_connection_route(const string &con_id, const string &route)
{  
  GuiInterface *gui = GuiInterface::getSingleton();
  
  id_map_t &cmap = netid_to_conid_.top();
  string con = cmap[con_id];
  string cmmd = "set ConnectionRoutes(" + con + ") " + route;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_module_note(const string &mod_id, const string &pos, 
			   const string &col, const string &note)
{  
  GuiInterface *gui = GuiInterface::getSingleton();
  
  string mod = get_mod_id(mod_id);
  string cmmd = "set Notes(" + mod + ") " + note;
  gui->eval(cmmd);
  cmmd = "set Notes(" + mod + "-Position) " + pos;
  gui->eval(cmmd);
  cmmd = "set Notes(" + mod + "-Color) " + col;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_connection_note(const string &con_id, const string &pos, 
			       const string &col, const string &note)
{ 
  GuiInterface *gui = GuiInterface::getSingleton();
  
  id_map_t &cmap = netid_to_conid_.top();
  string con = cmap[con_id];
  string cmmd = "set Notes(" + con + ") " + note;
  gui->eval(cmmd);
  cmmd = "set Notes(" + con + "-Position) " + pos;
  gui->eval(cmmd);
  cmmd = "set Notes(" + con + "-Color) " + col;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_set_variable(const string &var, const string &val)
{  
  GuiInterface *gui = GuiInterface::getSingleton();
  
  string cmmd = "set " + var +  " " + val;
  gui->eval(cmmd);
}

void 
NetworkIO::gui_open_module_gui(const string &mod_id)
{
  GuiInterface *gui = GuiInterface::getSingleton();
  
  string mod = get_mod_id(mod_id);
  string cmmd = mod + " initialize_ui";
  gui->eval(cmmd);
}

void 
NetworkIO::process_environment(const xmlNodePtr enode)
{
  xmlNodePtr node = enode->children;
  for (; node != 0; node = node->next) {
    if (string(to_char_ptr(node->name)) == string("var")) {
      xmlAttrPtr name_att = get_attribute_by_name(node, "name");
      xmlAttrPtr val_att = get_attribute_by_name(node, "val");
      env_subs_[string(to_char_ptr(name_att->children->content))] = 
	string(to_char_ptr(val_att->children->content));
    }
  }
}


void 
NetworkIO::process_modules_pass1(const xmlNodePtr enode)
{
  xmlNodePtr node = enode->children;
  for (; node != 0; node = node->next) {
    if (string(to_char_ptr(node->name)) == string("module") ||
	string(to_char_ptr(node->name)) == string("subnet")) 
    {
      bool do_subnet = string(to_char_ptr(node->name)) == string("subnet");
      xmlNodePtr network_node = 0;

      string x,y;
      xmlAttrPtr id_att = get_attribute_by_name(node, "id");
      xmlAttrPtr package_att = get_attribute_by_name(node, "package");
      xmlAttrPtr category_att = get_attribute_by_name(node, "category");
      xmlAttrPtr name_att = get_attribute_by_name(node, "name");
      
      string mname = string(to_char_ptr(name_att->children->content));
      string mid = string(to_char_ptr(id_att->children->content));
      xmlNodePtr pnode = node->children;
      for (; pnode != 0; pnode = pnode->next) {
	if (string(to_char_ptr(pnode->name)) == string("position")) 
	{
	  xmlAttrPtr x_att = get_attribute_by_name(pnode, "x");
	  xmlAttrPtr y_att = get_attribute_by_name(pnode, "y");
	  x = string(to_char_ptr(x_att->children->content));
	  y = string(to_char_ptr(y_att->children->content)); 
	  if (do_subnet) {
	    string old_ctx = gui_push_subnet_ctx();
	    gui_add_subnet_at_position(mid, mname, x, y);

	    ASSERT(network_node != 0);
	    process_network_node(network_node);
	    gui_pop_subnet_ctx(old_ctx);
	  } else {
	    gui_add_module_at_position(mid,
			  string(to_char_ptr(package_att->children->content)),
			  string(to_char_ptr(category_att->children->content)),
			  mname, x, y);
	  }
	}
	else if (string(to_char_ptr(pnode->name)) == string("network")) 
	{
	  network_node = pnode;
	} 	
	else if (string(to_char_ptr(pnode->name)) == string("note")) 
	{
	  xmlAttrPtr pos_att = get_attribute_by_name(pnode, "position");
	  xmlAttrPtr col_att = get_attribute_by_name(pnode, "color");
	  string pos, col, note;
	  if (pos_att)
	    pos = string(to_char_ptr(pos_att->children->content));
	  if (col_att)
	    col = string(to_char_ptr(col_att->children->content));
	  
	  note = string(to_char_ptr(pnode->children->content));
	  gui_set_module_note(mid, pos, col, note);
	}
	else if (string(to_char_ptr(pnode->name)) == string("port_caching")) 
	{
	  xmlNodePtr pc_node = pnode->children;
	  for (; pc_node != 0; pc_node = pc_node->next) {
	    if (string(to_char_ptr(pc_node->name)) == string("port")) 
	    {
	      xmlAttrPtr pid_att = get_attribute_by_name(pc_node, "id");
	      xmlAttrPtr val_att = get_attribute_by_name(pc_node, "val");
	      gui_set_module_port_caching(mid, 
			      string(to_char_ptr(pid_att->children->content)),
			      string(to_char_ptr(val_att->children->content)));
				   
	    }
	  }
	}
      }
    }
  }
}

void 
NetworkIO::process_modules_pass2(const xmlNodePtr enode)
{
  xmlNodePtr node = enode->children;
  for (; node != 0; node = node->next) 
	{
    if (string(to_char_ptr(node->name)) == string("module")) 
		{
      string x,y;
      xmlAttrPtr id_att = get_attribute_by_name(node, "id");
      xmlAttrPtr visible_att = get_attribute_by_name(node, "gui_visible");

      xmlNodePtr pnode = node->children;
      for (; pnode != 0; pnode = pnode->next) 
			{	
				if (string(to_char_ptr(pnode->name)) == string("gui_callback")) 
				{
					xmlNodePtr gc_node = pnode->children;
					for (; gc_node != 0; gc_node = gc_node->next) {
						if (string(to_char_ptr(gc_node->name)) == string("callback")) 
						{
							string call = string(to_char_ptr(gc_node->children->content));
							gui_call_module_callback(
									string(to_char_ptr(id_att->children->content)),
									call);
								 
						}
					}
				}
				else if (string(to_char_ptr(pnode->name)) == string("var")) 
				{
					xmlAttrPtr name_att = get_attribute_by_name(pnode, "name");
					xmlAttrPtr val_att = get_attribute_by_name(pnode, "val");
					xmlAttrPtr filename_att = get_attribute_by_name(pnode,"filename");
					xmlAttrPtr substitute_att = get_attribute_by_name(pnode,"substitute");

					string val = string(to_char_ptr(val_att->children->content));
					
					string filename = "no";
					if (filename_att != 0) filename = string(to_char_ptr(filename_att->children->content));
					if (filename == "yes") 
					{
						val = process_filename(val); 
					}
					else
					{
						string substitute = "yes";
						if (substitute_att != 0) substitute = string(to_char_ptr(substitute_att->children->content));
					  if (substitute == "yes") val = process_substitute(val);
					}
					
					gui_set_modgui_variable(
									string(to_char_ptr(id_att->children->content)),
									string(to_char_ptr(name_att->children->content)),
									val);
				}
      }
      if (visible_att && string(to_char_ptr(visible_att->children->content)) == "yes")
      {
				gui_open_module_gui(string(to_char_ptr(id_att->children->content)));
      }
    }
  }
}

void 
NetworkIO::process_connections(const xmlNodePtr enode)
{
  xmlNodePtr node = enode->children;
  for (; node != 0; node = node->next) {
    if (string(to_char_ptr(node->name)) == string("connection")) {
      xmlAttrPtr id_att = get_attribute_by_name(node, "id");
      xmlAttrPtr from_att = get_attribute_by_name(node, "from");
      xmlAttrPtr fromport_att = get_attribute_by_name(node, "fromport");
      xmlAttrPtr to_att = get_attribute_by_name(node, "to");
      xmlAttrPtr toport_att = get_attribute_by_name(node, "toport");
      xmlAttrPtr dis_att = get_attribute_by_name(node, "disabled");

      string id = string(to_char_ptr(id_att->children->content));

      gui_add_connection(id,
		     string(to_char_ptr(from_att->children->content)),
		     string(to_char_ptr(fromport_att->children->content)),
		     string(to_char_ptr(to_att->children->content)),
		     string(to_char_ptr(toport_att->children->content)));

      if (dis_att && 
	  string(to_char_ptr(dis_att->children->content)) == "yes") 
      {
	gui_set_connection_disabled(id);
      }


      xmlNodePtr cnode = node->children;
      for (; cnode != 0; cnode = cnode->next) {	
	if (string(to_char_ptr(cnode->name)) == string("route")) 
	{
	  gui_set_connection_route(id, 
			       string(to_char_ptr(cnode->children->content)));
	} 
	else if (string(to_char_ptr(cnode->name)) == string("note")) 
	{
	  xmlAttrPtr pos_att = get_attribute_by_name(cnode, "position");
	  xmlAttrPtr col_att = get_attribute_by_name(cnode, "color");
	  string pos, col, note;
	  if (pos_att)
	    pos = string(to_char_ptr(pos_att->children->content));
	  if (col_att)
	    col = string(to_char_ptr(col_att->children->content));
	  
	  note = string(to_char_ptr(cnode->children->content));
	  gui_set_connection_note(id, pos, col, note);
	} 
      }
    }
  }
}

string
NetworkIO::process_filename(const string &orig)
{
  // This function reinterprets a filename
	
	// Copy the string and remove TCL brackets
	std::string filename = orig.substr(1,orig.size()-2);
	
	// Remove blanks and tabs from the input (Some could have editted the XML file manually and may have left spaces)
	while (filename.size() > 0 && ((filename[0] == ' ')||(filename[0] == '\t'))) filename = filename.substr(1);
	while (filename.size() > 0 && ((filename[filename.size()-1] == ' ')||(filename[filename.size()-1] == '\t'))) filename = filename.substr(1,filename.size()-1);
	
	// Check whether filename is absolute:
	
	if ( filename.size() > 0 && filename[0] == '/') return (std::string("{")+filename+std::string("}")); // Unix absolute path
	if ( filename.size() > 1 && filename[1] == ':') return (std::string("{")+filename+std::string("}")); // Windows absolute path
	
	// If not substitute: 

	// Create a dynamic substitute called NETWORKDIR for relative path names
	std::string net_file = make_absolute_filename(net_file_);
	std::string::size_type backslashpos = net_file.find_last_of("\\");	
	std::string::size_type slashpos = net_file.find_last_of("/");	
  if (slashpos != std::string::npos && backslashpos != std::string::npos)
	{
		std::cerr << "Path to network file seems to contain both '\\' and '/' \n";
	}
	else
	{
	  std::string net_path = "";
		if (slashpos != std::string::npos) { net_path = net_file.substr(0,slashpos); }
		if (backslashpos != std::string::npos) { net_path = net_file.substr(0,backslashpos); }
		env_subs_[std::string("scisub_networkdir")] = std::string("SCIRUN_NETWORKDIR");
		sci_putenv("SCIRUN_NETWORKDIR",net_path);
	}
	
	map<string, string>::const_iterator iter = env_subs_.begin();
	while (iter != env_subs_.end()) 
  {
    const pair<const string, string> &kv = *iter++;
    const string &key = kv.first;
		
    map<string, string>::size_type idx = filename.find(key);

    if (idx != string::npos) {
      const string &env_var = kv.second;
      const char* env = sci_getenv(env_var);
      string subst = (env != 0)?env:"";
      
      if (env_var == string("SCIRUN_DATASET") && subst.size() == 0)
      {
				subst = string("sphere");
      }
      while (idx != string::npos) {
				filename = filename.replace(idx, key.size(), subst);
				idx = filename.find(key);
      }
    }
  }

  for (size_t p = 0 ; p<filename.size(); p++)
  {
    if (filename[p] == '\\') filename[p] = '/';	
  }


  return (std::string("{")+filename+std::string("}"));
}


string
NetworkIO::process_substitute(const string &orig)
{
	string src = orig;
	map<string, string>::const_iterator iter = env_subs_.begin();
	while (iter != env_subs_.end()) 
  {
    const pair<const string, string> &kv = *iter++;
    const string &key = kv.first;
		
    map<string, string>::size_type idx = src.find(key);

    if (idx != string::npos) {
      const string &env_var = kv.second;
      const char* env = sci_getenv(env_var);
      string subst = (env != 0)?env:"";
      
      if (env_var == string("SCIRUN_DATASET") && subst.size() == 0)
      {
				subst = string("sphere");
      }
      while (idx != string::npos) {
				src = src.replace(idx, key.size(), subst);
				idx = src.find(key);
      }
    }
  }

  return (src);
}


void
NetworkIO::load_net(const string &net)
{
  net_file_ = net;
}


void
NetworkIO::process_network_node(xmlNode* network_node)
{
  // have to multi pass this document to workaround tcl timing issues.
  // PASS 1 - create the modules and connections
  xmlNode* node = network_node;
  for (; node != 0; node = node->next) {
    // skip all but the component node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      //! set attributes
      //xmlAttrPtr version_att = get_attribute_by_name(node, "version");
      xmlAttrPtr name_att = get_attribute_by_name(node, "name");
      gui_set_variable(string("name"), 
		       string(to_char_ptr(name_att->children->content)));
      xmlAttrPtr bbox_att = get_attribute_by_name(node, "bbox");
      gui_set_variable(string("bbox"), 
		       string(to_char_ptr(bbox_att->children->content)));
      xmlAttrPtr cd_att = get_attribute_by_name(node, "creationDate");
      gui_set_variable(string("creationDate"), 
		       string(to_char_ptr(cd_att->children->content)));
      xmlAttrPtr ct_att = get_attribute_by_name(node, "creationTime");
      gui_set_variable(string("creationTime"), 
		       string(to_char_ptr(ct_att->children->content)));
      xmlAttrPtr geom_att = get_attribute_by_name(node, "geometry");
      gui_set_variable(string("geometry"), 
		       string(to_char_ptr(geom_att->children->content)));
      
      xmlNode* enode = node->children;
      for (; enode != 0; enode = enode->next) {

	if (enode->type == XML_ELEMENT_NODE && 
	    string(to_char_ptr(enode->name)) == string("environment")) 
	{
	  process_environment(enode);
	} else if (enode->type == XML_ELEMENT_NODE && 
		   string(to_char_ptr(enode->name)) == string("modules")) 
	{
	  process_modules_pass1(enode);
	} else if (enode->type == XML_ELEMENT_NODE && 
		   string(to_char_ptr(enode->name)) == string("connections")) 
	{
	  process_connections(enode);
	} else if (enode->type == XML_ELEMENT_NODE && 
		   string(to_char_ptr(enode->name)) == string("note")) 
	{
	  gui_set_variable(string("notes"), 
			   string(to_char_ptr(enode->children->content)));
	}
      }
    }
  }

  // PASS 2 -- call the callbacks and set the variables
  node = network_node;
  for (; node != 0; node = node->next) {
    // skip all but the component node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      xmlNode* enode = node->children;
      for (; enode != 0; enode = enode->next) {

	if (enode->type == XML_ELEMENT_NODE && 
	    string(to_char_ptr(enode->name)) == string("modules")) 
	{
	  process_modules_pass2(enode);
	}
      }
    }
  }
}

bool
NetworkIO::load_network()
{

  /*
   * this initialize the library and check potential ABI mismatches
   * between the version it was compiled for and the actual shared
   * library used.
   */
  LIBXML_TEST_VERSION;
  
  xmlParserCtxtPtr ctxt; /* the parser context */
  xmlDocPtr doc; /* the resulting document tree */

  string src_dir = string(sci_getenv("SCIRUN_SRCDIR")) + 
    string("/nets/network.dtd");
  xmlInitializeCatalog();
  xmlCatalogAdd(BAD_CAST "public", BAD_CAST "-//SCIRun/Network DTD", 
		BAD_CAST src_dir.c_str());

  /* create a parser context */
  ctxt = xmlNewParserCtxt();
  if (ctxt == 0) {
    std::cerr << "ComponentNode.cc: Failed to allocate parser context" 
	      << std::endl;
    return false;
  }
  /* parse the file, activating the DTD validation option */
  doc = xmlCtxtReadFile(ctxt, net_file_.c_str(), 0, (XML_PARSE_DTDATTR | 
						     XML_PARSE_DTDVALID | 
						     XML_PARSE_PEDANTIC));
  /* check if parsing suceeded */
  if (doc == 0) {
    std::cerr << "ComponentNode.cc: Failed to parse " << net_file_ 
	      << std::endl;
    return false;
  } else {
    /* check if validation suceeded */
    if (ctxt->valid == 0) {
      std::cerr << "ComponentNode.cc: Failed to validate " << net_file_ 
		<< std::endl;
      return false;
    }
  }

  GuiInterface *gui = GuiInterface::getSingleton();
  gui->eval("::netedit dontschedule");
  
  // parse the doc at network node.
  process_network_node(doc->children);

  xmlFreeDoc(doc);
  /* free up the parser context */
  xmlFreeParserCtxt(ctxt);  
#ifndef _WIN32
  // there is a problem on windows when using Uintah 
  // which is either caused or exploited by this
  xmlCleanupParser();
#endif

  gui->eval("setGlobal NetworkChanged 0");
  gui->eval("set netedit_savefile {" + net_file_ + "}");
  gui->eval("::netedit scheduleok");

  // first draw autoview.
  autoview_pending_ = true;
  return true;
}


// push a new network root node.
void 
NetworkIO::push_subnet_scope(const string &id, const string &name)
{
  // this is a child node of the network.
  xmlNode* mod_node = 0;
  xmlNode* net_node = 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      net_node = node;
      xmlNode* mnode = node->children;
      for (; mnode != 0; mnode = mnode->next) {
	if (string(to_char_ptr(mnode->name)) == string("modules")) {
	  mod_node = mnode;
	  break;
	}
      }
    }
  }
  if (! mod_node) { 
    if (! net_node) { 
      cerr << "ERROR: could not find top level node." << endl;
      return;
    }
    mod_node = xmlNewChild(net_node, 0, BAD_CAST "modules", 0);
  }
  xmlNodePtr tmp = xmlNewChild(mod_node, 0, BAD_CAST "subnet", 0);
  xmlNewProp(tmp, BAD_CAST "id", BAD_CAST id.c_str());
  xmlNewProp(tmp, BAD_CAST "package", BAD_CAST "subnet");
  xmlNewProp(tmp, BAD_CAST "category", BAD_CAST "subnet");
  xmlNewProp(tmp, BAD_CAST "name", BAD_CAST name.c_str());

  xmlNodePtr sn_node = xmlNewChild(tmp, 0, BAD_CAST "network", 0);
  xmlNewProp(sn_node, BAD_CAST "version", BAD_CAST "contained");

  subnets_.push(sn_node);
}

void 
NetworkIO::pop_subnet_scope()
{
  subnets_.pop();
}

void 
NetworkIO::start_net_doc(const string &fname, const string &vers)
{
  out_fname_ = fname;
  xmlNodePtr root_node = 0; /* node pointers */
  xmlDtdPtr dtd = 0;        /* DTD pointer */
  
  LIBXML_TEST_VERSION;
  
  /* 
   * Creates a new document, a node and set it as a root node
   */
  doc_ = xmlNewDoc(BAD_CAST "1.0");
  root_node = xmlNewNode(0, BAD_CAST "network");
  subnets_.push(root_node);
  xmlDocSetRootElement(doc_, root_node);
  
  /*
   * Creates a DTD declaration.
   */
  string dtdstr = string("network.dtd");

  dtd = xmlCreateIntSubset(doc_, BAD_CAST "network", 
			   BAD_CAST "-//SCIRun/Network DTD", 
			   BAD_CAST dtdstr.c_str());
  
  /* 
   * xmlNewChild() creates a new node, which is "attached" as child node
   * of root_node node. 
   */
  xmlNewProp(root_node, BAD_CAST "version", BAD_CAST vers.c_str());
}

void
NetworkIO::write_net_doc()
{
  // write the file
  xmlSaveFormatFileEnc(out_fname_.c_str(), doc_, "UTF-8", 1);
  
  // free the document
  xmlFreeDoc(doc_);
  doc_ = 0;
  out_fname_ = "";
  done_writing_ = true;
}

void 
NetworkIO::add_net_var(const string &var, const string &val)
{
  // add these as attributes of the network node.
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      break;
    }
  }
  if (! node) {
    cerr << "ERROR: could not find top level node." << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST var.c_str(), BAD_CAST val.c_str());
}

void 
NetworkIO::add_environment_sub(const string &var, const string &val)
{
  // this is a child node of the network.
  xmlNode* env_node = 0;
  xmlNode* net_node = 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      net_node = node;
      xmlNode* enode = node->children;
      for (; enode != 0; enode = enode->next) {
	if (string(to_char_ptr(enode->name)) == string("environment")) {
	  env_node = enode;
	  break;
	}
      }
    }
  }
  if (! env_node) { 
    if (! net_node) { 
      cerr << "ERROR: could not find top level node." << endl;
      return;
    }
    env_node = xmlNewChild(net_node, 0, BAD_CAST "environment", 0);
  }
  xmlNodePtr tmp = xmlNewChild(env_node, 0, BAD_CAST "var", 0);
  xmlNewProp(tmp, BAD_CAST "name", BAD_CAST var.c_str());
  xmlNewProp(tmp, BAD_CAST "val", BAD_CAST val.c_str());
}

void 
NetworkIO::add_net_note(const string &val)
{
  // this is a child node of the network, must come after 
  // environment node if it exists.
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      break;
    }
  }
  if (! node) { 
    cerr << "ERROR: could not find 'network' node." << endl;
    return;
  }

  xmlNewTextChild(node, 0, BAD_CAST "note", BAD_CAST val.c_str()); 
}


void 
NetworkIO::add_module_node(const string &id, const string &pack, 
			   const string &cat, const string &mod)
{
  // this is a child node of the network.
  xmlNode* mod_node = 0;
  xmlNode* net_node = 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      net_node = node;
      xmlNode* mnode = node->children;
      for (; mnode != 0; mnode = mnode->next) {
	if (string(to_char_ptr(mnode->name)) == string("modules")) {
	  mod_node = mnode;
	  break;
	}
      }
    }
  }
  if (! mod_node) { 
    if (! net_node) { 
      cerr << "ERROR: could not find top level node." << endl;
      return;
    }
    mod_node = xmlNewChild(net_node, 0, BAD_CAST "modules", 0);
  }
  xmlNodePtr tmp = xmlNewChild(mod_node, 0, BAD_CAST "module", 0);
  xmlNewProp(tmp, BAD_CAST "id", BAD_CAST id.c_str());
  xmlNewProp(tmp, BAD_CAST "package", BAD_CAST pack.c_str());
  xmlNewProp(tmp, BAD_CAST "category", BAD_CAST cat.c_str());
  xmlNewProp(tmp, BAD_CAST "name", BAD_CAST mod.c_str());
}

xmlNode*
NetworkIO::get_module_node(const string &id)
{  
  xmlNode* mid_node = 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      xmlNode* msnode = node->children;
      for (; msnode != 0; msnode = msnode->next) {

	if (string(to_char_ptr(msnode->name)) == string("modules")) {
	  xmlNode* mnode = msnode->children;
	  for (; mnode != 0; mnode = mnode->next) {

	    if (string(to_char_ptr(mnode->name)) == string("module") ||
		string(to_char_ptr(mnode->name)) == string("subnet")) 
	    {
	      xmlAttrPtr name_att = get_attribute_by_name(mnode, "id");
	      string mid = string(to_char_ptr(name_att->children->content));
	      if (mid == id) 
	      {
		mid_node = mnode;
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  return mid_node;
}

void 
NetworkIO::add_module_variable(const string &id, const string &var, 
			       const string &val, bool filename, bool substitute, bool userelfilenames)
{
  xmlNode* node = get_module_node(id);

  if (! node) { 
    cerr << "ERROR: could not find module node with id: " << id << endl;
    return;
  }
  xmlNodePtr tmp = xmlNewChild(node, 0, BAD_CAST "var", 0);
  xmlNewProp(tmp, BAD_CAST "name", BAD_CAST var.c_str());
	
	string nval = val;
	if (filename && userelfilenames)
	{
		if ((nval.size() >0) &&  (nval[0] == '{'))
		{
			nval = string("{") + make_relative_filename(nval.substr(1,nval.size()-2),out_fname_) + string("}");
		}
		else
		{
			nval = make_relative_filename(nval,out_fname_);		
		}
	}
	
  xmlNewProp(tmp, BAD_CAST "val", BAD_CAST nval.c_str());
	if (filename) xmlNewProp(tmp, BAD_CAST "filename", BAD_CAST "yes"); 

	if (substitute) xmlNewProp(tmp, BAD_CAST "substitute", BAD_CAST "yes"); 
	else xmlNewProp(tmp, BAD_CAST "substitute", BAD_CAST "no");
}

void 
NetworkIO::set_module_gui_visible(const string &id)
{
  xmlNode* node = get_module_node(id);

  if (! node) { 
    cerr << "ERROR: could not find module node with id: " << id << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST "gui_visible", BAD_CAST "yes");
}

void 
NetworkIO::add_module_gui_callback(const string &id, const string &call)
{
  xmlNode* gc_node = 0;
  xmlNode* mod_node = get_module_node(id);
  if (! mod_node) { 
    cerr << "ERROR: could not find node for module id: " << id << endl;
    return;
  }
  xmlNode *node = mod_node->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("gui_callback")) {
      gc_node = node;
      break;
    }
  }
      
  if (! gc_node) { 
    gc_node = xmlNewChild(mod_node, 0, BAD_CAST "gui_callback", 0);
  }
  xmlNewTextChild(gc_node, 0, BAD_CAST "callback", BAD_CAST call.c_str());
}

void 
NetworkIO::add_module_position(const string &id, const string &x, 
			       const string &y)
{
  xmlNode* mid_node = get_module_node(id);

  if (! mid_node) { 
    cerr << "ERROR: could not find module node with id: " << id << endl;
    return;
  }
  xmlNodePtr tmp = xmlNewChild(mid_node, 0, BAD_CAST "position", 0);
  xmlNewProp(tmp, BAD_CAST "x", BAD_CAST x.c_str());
  xmlNewProp(tmp, BAD_CAST "y", BAD_CAST y.c_str());

}

void 
NetworkIO::add_module_note(const string &id, const string &note)
{
  xmlNode* mnode = get_module_node(id);
  
  if (! mnode) { 
    cerr << "ERROR: could not find module node with id: " << id << endl;
    return;
  }
  xmlNewTextChild(mnode, 0, BAD_CAST "note", BAD_CAST note.c_str());
}
 
void 
NetworkIO::add_module_note_position(const string &id, const string &pos)
{
  bool found = false;
  xmlNode* node = get_module_node(id);
  if (! node) { 
    cerr << "ERROR: could not find node for module id: " << id << endl;
    return;
  }
  node = node->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("note")) {
      found = true;
      break;
    }
  }
      
  if (! found) { 
    cerr << "ERROR: could not find note node for module id: " << id << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST "position", BAD_CAST pos.c_str());
}
 
void 
NetworkIO::add_module_note_color(const string &id, const string &col)
{
  bool found = false;
  xmlNode* node = get_module_node(id);
  if (! node) { 
    cerr << "ERROR: could not find node for module id: " << id << endl;
    return;
  }
  node = node->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("note")) {
      found = true;
      break;
    }
  }
      
  if (! found) { 
    cerr << "ERROR: could not find note node for module id: " << id << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST "color", BAD_CAST col.c_str());
}
 
void 
NetworkIO::add_connection_node(const string &id, const string &fmod, 
			       const string &fport, const string &tmod, 
			       const string &tport)
{
  // this is a child node of the network.
  xmlNode* con_node= 0;
  xmlNode* net_node= 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      net_node = node;
      xmlNode* cnode = node->children;
      for (; cnode != 0; cnode = cnode->next) {
	if (string(to_char_ptr(cnode->name)) == string("connections")) {
	  con_node = cnode;
	  break;
	}
      }
    }
  }
  if (! con_node) { 
    if (! net_node) { 
      cerr << "ERROR: could not find top level node." << endl;
      return;
    }
    con_node = xmlNewChild(net_node, 0, BAD_CAST "connections", 0);
  }
  xmlNodePtr tmp = xmlNewChild(con_node, 0, BAD_CAST "connection", 0);
  xmlNewProp(tmp, BAD_CAST "id", BAD_CAST id.c_str());
  xmlNewProp(tmp, BAD_CAST "from", BAD_CAST fmod.c_str());
  xmlNewProp(tmp, BAD_CAST "fromport", BAD_CAST fport.c_str());
  xmlNewProp(tmp, BAD_CAST "to", BAD_CAST tmod.c_str());
  xmlNewProp(tmp, BAD_CAST "toport", BAD_CAST tport.c_str());
}


xmlNode*
NetworkIO::get_connection_node(const string &id)
{  
  xmlNode* cid_node = 0;
  xmlNode* node = subnets_.top();
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("network")) 
    {
      xmlNode* msnode = node->children;
      for (; msnode != 0; msnode = msnode->next) {

	if (string(to_char_ptr(msnode->name)) == string("connections")) {
	  xmlNode* mnode = msnode->children;
	  for (; mnode != 0; mnode = mnode->next) {

	    if (string(to_char_ptr(mnode->name)) == string("connection")) {
	      xmlAttrPtr name_att = get_attribute_by_name(mnode, "id");
	      string cid = string(to_char_ptr(name_att->children->content));
	      if (cid == id) {
		cid_node = mnode;
		break;
	      }
	    }
	  }
	}
      }
    }
  }
  return cid_node;
}

void 
NetworkIO::set_disabled_connection(const string &id)
{
  xmlNode* cid_node = get_connection_node(id);

  if (! cid_node) { 
    cerr << "ERROR: could not find connection node with id: " << id << endl;
    return;
  }
  xmlNewProp(cid_node, BAD_CAST "disabled", BAD_CAST "yes");
}
 
void 
NetworkIO::add_connection_route(const string &id, const string &route)
{
  xmlNode* cid_node = get_connection_node(id);

  if (! cid_node) { 
    cerr << "ERROR: could not find connection node with id: " << id << endl;
    return;
  }

  xmlNewTextChild(cid_node, 0, BAD_CAST "route", BAD_CAST route.c_str());
}
 
void 
NetworkIO::add_connection_note(const string &id, const string &note)
{
  xmlNode* cid_node = get_connection_node(id);

  if (! cid_node) { 
    cerr << "ERROR: could not find connection node with id: " << id << endl;
    return;
  }

  xmlNewTextChild(cid_node, 0, BAD_CAST "note", BAD_CAST note.c_str());
}
 
void 
NetworkIO::add_connection_note_position(const string &id, const string &pos)
{
  bool found = false;
  xmlNode* node = get_connection_node(id);

  if (! node) { 
    cerr << "ERROR: could not find node for connection id: " << id << endl;
    return;
  }
  node = node->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("note")) {
      found = true;
      break;
    }
  }
      
  if (! found) { 
    cerr << "ERROR: could not find note node for module id: " << id << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST "position", BAD_CAST pos.c_str());
}
 
void 
NetworkIO::add_connection_note_color(const string &id, const string &col)
{
  bool found = false;
  xmlNode* node = get_connection_node(id);

  if (! node) { 
    cerr << "ERROR: could not find node for connection id: " << id << endl;
    return;
  }
  node = node->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("note")) {
      found = true;
      break;
    }
  }
      
  if (! found) { 
    cerr << "ERROR: could not find note node for module id: " << id << endl;
    return;
  }
  xmlNewProp(node, BAD_CAST "color", BAD_CAST col.c_str());
}
 

void 
NetworkIO::set_port_caching(const string &id, const string &port, 
			    const string &val)
{
  xmlNode* mnode = get_module_node(id);
  xmlNode* pcnode = 0;
  if (! mnode) { 
    cerr << "ERROR: could not find module node with id: " << id << endl;
    return;
  }

  xmlNode* node = mnode->children;
  for (; node != 0; node = node->next) {
    // skip all but the network node.
    if (node->type == XML_ELEMENT_NODE && 
	string(to_char_ptr(node->name)) == string("port_caching")) 
    {
      pcnode = node;
    }
  }

  if (! pcnode) { 
    pcnode = xmlNewChild(mnode, 0, BAD_CAST "port_caching", 0);
  }

  xmlNode *tmp;
  tmp = xmlNewChild(pcnode, 0, BAD_CAST "port", 0);
  xmlNewProp(tmp, BAD_CAST "id", BAD_CAST port.c_str());
  xmlNewProp(tmp, BAD_CAST "val", BAD_CAST val.c_str());
}


string
NetworkIO::make_absolute_filename(string name)
{
	// Remove blanks and tabs from the input (Some could have editted the XML file manually and may have left spaces)
	while (name.size() > 0 && ((name[0] == ' ')||(name[0] == '\t'))) name = name.substr(1);
	while (name.size() > 0 && ((name[name.size()-1] == ' ')||(name[name.size()-1] == '\t'))) name = name.substr(1,name.size()-1);
	
	// Check whether filename is absolute:
	
	if ( name.size() > 0 && name[0] == '/') return (name); // Unix absolute path
	if ( name.size() > 2 && name[1] == ':' && ((name[2] == '\\')||(name[2] == '/')))
  {
    for (size_t i=0; i<name.size();i++) if (name[i] == '\\') name[i] = '/';
    return (name); // Windows absolute path
	}
  
  Dir CWD = Dir::current_directory();
  string cwd = CWD.getName();

  for (size_t i=0; i<name.size();i++) if (name[i] == '\\') name[i] = '/';
  for (size_t i=0; i<cwd.size();i++) if (cwd[i] == '\\') cwd[i] = '/';


	if (cwd.size() > 0)
	{
		if(cwd[0] == '/')
		{
			if (cwd[cwd.size()-1]!='/') cwd +='/';
			name = cwd+name;
			
					// collapse name further
			
			std::string::size_type ddpos = name.find("../");

			while (ddpos != std::string::npos)
			{
				if (ddpos > 1 && name[ddpos-1] == '/')
				{
					std::string::size_type slashpos = name.find_last_of("/",ddpos-2);
					if (slashpos == std::string::npos)
					{
						if ((name.substr(0,ddpos-1) != "..")&&(name.substr(0,ddpos-1) != "."))
						{
							name = name.substr(ddpos+3); 
							ddpos = name.find("../");
						}
						else 
						{
							ddpos = name.find("../",ddpos+3);
						}
					}
					else
					{
						if ((name.substr(slashpos+1,ddpos-1)!="..")&&(name.substr(slashpos+1,ddpos-1)!=".")) 
						{
							name = name.substr(0,slashpos+1)+name.substr(ddpos+3);
							ddpos = name.find("../");
						}
						else
						{
							ddpos = name.find("../",ddpos+3);
						}
					}
					
				}
				else
				{
					ddpos = name.find("../",ddpos+3);
				}
			}
		}
		else
		{
      // Windows filename
      
			if (cwd[cwd.size()-1]!='/') cwd +='/';
			
			name = cwd+name;
			
					// collapse name further
			
			std::string::size_type ddpos = name.find("../");

			while (ddpos != std::string::npos)
			{
				if (ddpos > 1 && name[ddpos-1] == '/')
				{
					std::string::size_type slashpos = name.find_last_of("/",ddpos-2);
					if (slashpos == std::string::npos)
					{
						if ((name.substr(0,ddpos-1) != "..")&&(name.substr(0,ddpos-1) != "."))
						{
							name = name.substr(ddpos+3); 
							ddpos = name.find("../");
						}
						else 
						{
							ddpos = name.find("../",ddpos+3);
						}
					}
					else
					{
						if ((name.substr(slashpos+1,ddpos-1)!="..")&&(name.substr(slashpos+1,ddpos-1)!=".")) 
						{
							name = name.substr(0,slashpos+1)+name.substr(ddpos+3);
							ddpos = name.find("../");
						}
						else
						{
							ddpos = name.find("../",ddpos+3);
						}
					}

        }
				else
				{
					ddpos = name.find("../",ddpos+3);
				}
			}			

		}
	}
	
	return (name);
}


string
NetworkIO::make_relative_filename(string name, string path)
{
	std::cout << "path="<<path<<"\n";
	// if it is not absolute assume it is relative to current directory
	path = make_absolute_filename(path);

	// Remove blanks and tabs from the input (Some could have editted the XML file manually and may have left spaces)
	while (name.size() > 0 && ((name[0] == ' ')||(name[0] == '\t'))) name = name.substr(1);
	while (name.size() > 0 && ((name[name.size()-1] == ' ')||(name[name.size()-1] == '\t'))) name = name.substr(1,name.size()-1);

	// Check whether filename is absolute:
	
	bool abspath = false;
	if ( name.size() > 0 && name[0] == '/') abspath = true; // Unix absolute path
	if ( name.size() > 2 && name[1] == ':' && ((name[2] == '\\') ||(name[2] == '/'))) abspath = true; // Windows absolute path

	if (abspath == false) return (name); // We could not make it relative as it is already relative

	if ( name.size() > 0 && name[0] == '/')
	{
		string npath = path;
	  string nname = name;
		string::size_type slashpos = path.find("/");
		bool backtrack = false;
		while(slashpos != string::npos)
		{
			if (npath.substr(0,slashpos) == nname.substr(0,slashpos) && backtrack == false)
			{
				npath = npath.substr(slashpos+1);
				nname = nname.substr(slashpos+1);
			}
			else
			{
				backtrack = true;
				npath = npath.substr(slashpos+1);
				nname = "../" + nname;
			}
			slashpos = npath.find("/");
		}
		
		// collapse name further
		
		std::string::size_type ddpos = nname.find("../");

		while (ddpos != std::string::npos)
		{
			if (ddpos > 1 && nname[ddpos-1] == '/')
			{
				std::string::size_type slashpos = nname.find_last_of("/",ddpos-2);
				if (slashpos == std::string::npos)
				{
					if ((nname.substr(0,ddpos-1) != "..")&&(nname.substr(0,ddpos-1) != "."))
					{
						nname = nname.substr(ddpos+3); 
						ddpos = nname.find("../");
					}
					else 
					{
						ddpos = nname.find("../",ddpos+3);
					}
				}
				else
				{
					if ((nname.substr(slashpos+1,ddpos-1)!="..")&&(nname.substr(slashpos+1,ddpos-1)!=".")) 
					{
						nname = nname.substr(0,slashpos+1)+nname.substr(ddpos+3);
						ddpos = nname.find("../");
					}
					else
					{
						ddpos = nname.find("../",ddpos+3);
					}
				}
				
			}
			else
			{
				ddpos = nname.find("../",ddpos+3);
			}
		}

		nname = "scisub_networkdir/"+nname;		
		return (nname);
	}
	else if ( name.size() > 2 && name[1] == ':' && ((name[2] == '\\')||(name[2] == '/' )))
	{
    // Convert everything to forward slash
    for (size_t i=0; i< name.size(); i++) if (name[i] == '\\') name[i] = '/';

		if (path.size() > 2)
		{
			if (path.substr(0,3) != name.substr(0,3))
			{
				std::cerr << "WARNING: Could not make pathname relative as it is on another drive\n";
				return (name);
			}
		}
		else
		{
			std::cerr << "WARNING: Failed to convert network pathname to an absolute path name\n";
			return (name);
		}
	
		string npath = path;
	  string nname = name;
		string::size_type slashpos = path.find("/");
		bool backtrack = false;
		while(slashpos != string::npos)
		{
			if (npath.substr(0,slashpos) == nname.substr(0,slashpos) && backtrack == false)
			{
				npath = npath.substr(slashpos+1);
				nname = nname.substr(slashpos+1);
			}
			else
			{
				backtrack = true;
				npath = npath.substr(slashpos+1);
				nname = "../" + nname;
			}
			slashpos = npath.find("/");
		}

		// collapse name further
		
		std::string::size_type ddpos = nname.find("../");


		while (ddpos != std::string::npos)
		{
			if (ddpos > 1 && nname[ddpos-1] == '/')
			{
				std::string::size_type slashpos = nname.find_last_of("/",ddpos-2);
				if (slashpos == std::string::npos)
				{
					if ((nname.substr(0,ddpos-1) != "..")&&(nname.substr(0,ddpos-1) != "."))
					{
						nname = nname.substr(ddpos+3); 
						ddpos = nname.find("../");
					}
					else 
					{
						ddpos = nname.find("../",ddpos+3);
					}
				}
				else
				{
					if ((nname.substr(slashpos+1,ddpos-1)!="..")&&(nname.substr(slashpos+1,ddpos-1)!=".")) 
					{
						nname = nname.substr(0,slashpos+1)+nname.substr(ddpos+3);
						ddpos = nname.find("../");
					}
					else
					{
						ddpos = nname.find("../",ddpos+3);
					}
				}
			}
			else
			{
				ddpos = nname.find("../",ddpos+3);
			}
		}
    nname = "scisub_networkdir/"+nname;
		return (nname);
	}
	
	std::cerr << "WARNING: Could not convert filename into a relative filename\n";
	return (name);
}

} // end namespace SCIRun
