//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2006 Scientific Computing and Imaging Institute,
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
//  
//    File   : XMLIO.cc
//    Author : McKay Davis
//    Date   : Tue Jun 27 13:01:28 2006

#include <Core/Skinner/XMLIO.h>
#include <Core/Skinner/Variables.h>
#include <Core/Skinner/Root.h>

#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>

#include <libxml/xmlreader.h>
#include <libxml/catalog.h>
#include <iostream>


namespace SCIRun {
  namespace Skinner {
    XMLIO::DrawableMakerMap_t XMLIO::makers_;

    XMLIO::XMLIO()
    {
    }

    XMLIO::~XMLIO()
    {
    }

    Root *
    XMLIO::load(const string &filename)
    {
      /*
       * this initialize the library and check potential ABI mismatches
       * between the version it was compiled for and the actual shared
       * library used.
       */
      
      LIBXML_TEST_VERSION;
      
      xmlParserCtxtPtr ctxt; /* the parser context */
      xmlDocPtr doc; /* the resulting document tree */
      
      string dtd = string(sci_getenv("SCIRUN_SRCDIR")) + 
        string("/Core/Skinner/skinner.dtd");
      xmlInitializeCatalog();
      xmlCatalogAdd(XMLUtil::char_to_xmlChar("public"), 
                    XMLUtil::char_to_xmlChar("-//Skinner/Drawable DTD"), 
                    XMLUtil::char_to_xmlChar(dtd.c_str()));
     
      /* create a parser context */
      ctxt = xmlNewParserCtxt();
      if (!ctxt) {
        std::cerr << "XMLIO::load failed xmlNewParserCtx()\n";
        return 0;
      }

      /* parse the file, activating the DTD validation option */
      doc = xmlCtxtReadFile(ctxt, filename.c_str(), 0, (XML_PARSE_DTDATTR | 
                                                        XML_PARSE_DTDVALID | 
                                                        XML_PARSE_PEDANTIC));
      /* check if parsing suceeded */
      if (!doc) {
        std::cerr << "Skinner::XMLIO::load failed to parse " 
                  << filename << std::endl;
        return 0;
      } 
      if (!ctxt->valid) {
          std::cerr << "Skinner::XMLIO::load dailed to validate " 
                    << filename << std::endl;
          return 0;
      }
      

      // parse the doc at network node.
      Root *root;
      for (xmlNode *cnode=doc->children; cnode!=0; cnode=cnode->next) {
        if (XMLUtil::node_is_dtd(cnode, "skinner")) 
          continue;
        if (XMLUtil::node_is_element(cnode, "skinner")) 
          root = eval_skinner_node(cnode, filename);
        else if (!XMLUtil::node_is_comment(cnode))
          throw "Unknown node type";
      }               

      //Drawable *obj = 0;
      xmlFreeDoc(doc);
      /* free up the parser context */
      xmlFreeParserCtxt(ctxt);  
      xmlCleanupParser();

      return root;
    }
    

    xmlNodePtr
    XMLIO::find_definition(definition_nodes_t &definitions,
                           const string &classname) 
    {
      // Go backwards through the vector becasue we want to search
      // UP the current object context tree defined in the skinner file
      // looking for definitions in the nearest ancestor before looking
      // at their parent node
      definition_nodes_t::reverse_iterator def_map_iter = definitions.rbegin();
      definition_nodes_t::reverse_iterator def_map_end = definitions.rend();
      for (;def_map_iter != def_map_end; ++def_map_iter) {
        string_node_map_t::iterator found_def = def_map_iter->find(classname);
        if (found_def != def_map_iter->end()) {
          return found_def->second;
        }
      }
      return 0;
    }




    void
    XMLIO::eval_merged_object_nodes_and_push_definitions
    (merged_nodes_t &merged_nodes, definition_nodes_t &defs)
    {
      defs.push_back(string_node_map_t());
      string_node_map_t &node_def_map = defs.back();

      // Spin through the xml var nodes and set their values
      for (merged_nodes_t::iterator mnode = merged_nodes.begin(); 
           mnode != merged_nodes.end(); ++mnode) {        
        for (xmlNode *cnode = (*mnode)->children; cnode; cnode = cnode->next) {
          if (XMLUtil::node_is_element(cnode, "definition")) {
            eval_definition_node(cnode, node_def_map);
          } 
        }     
      } 

      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << std::endl;
      }

    }



    Drawable *
    XMLIO::eval_object_node(const xmlNodePtr node, 
                            Variables *variables,
                            definition_nodes_t &definitions,
                            //TargetSignalMap_t &signals,
                            SignalThrower::SignalCatchers_t &catchers) 
    {
      const bool root_node = XMLUtil::node_is_element(node, "skinner");
      ASSERT(root_node || XMLUtil::node_is_element(node, "object"));

      // classname is exact class type for this skinner drawable
      string classname = "Skinner::Root";
      bool foundclassname = 
        XMLUtil::maybe_get_att_as_string(node, "class", classname);

      if (!root_node && !foundclassname) { // redundant, as dtd should fail
        cerr << "Object does not have classname\n";
        return 0;
      }

      // If the string in the xml file is proceeded with a $, then
      // its a variable dereference
      while (classname[0] == '$' &&
             variables->maybe_get_string
             (classname.substr(1,classname.length()-1), classname)) 
      {
        if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
            cerr << variables->get_id() << " is actually a classname: " 
                 << classname << "\n";
        }
      }

      
      // Object id is not required, create unique one if not found
      // The first merged node contains the id
      string unique_id = "";
      if (!XMLUtil::maybe_get_att_as_string(node, "id",  unique_id)) {
        // Note: This isnt absolutely guaranteed to make unique ids
        // It can be broken by hardcoding the id in the XML
        static int objcount = 0;
        unique_id = classname+"."+to_string(objcount++);
      }


      // First, before we can even construct the object, we need to 
      // get the Variables that determine this instances unique properties
      // Create a new Variables context with our Unique ID
      // This creates new memory that needs to be freed by the object
      variables = new Variables(unique_id, variables);
      

      // The classname could refer to a previously parsed <definition> node
      // in that case, we instance the contained single object node as it were 
      // createed in the current context.  <var>s and <signals> are tags
      // are allowed in both this <object> tag and the <definiition>
      // encapsulated <object> tag.  The Variables, signals/catchers of the
      // encapsulated tag are merged last.
      // Is this object tag, just an alias to another object node?
      // Definitions can contain nested already decalred definitions as
      // their object classname, unroll the definitions  until we find
      // an non previously defined classname
      merged_nodes_t merged_nodes(1,node);
      xmlNodePtr dnode = find_definition(definitions, classname);
      while (dnode)
      {
        // When searching for vars, signals, and children
        // we need to merge the encapsulated nodes
        merged_nodes.push_back(dnode);
        // If we are just a reference to a definition, we need to switch
        // classnames to the encapsulated object
        classname = XMLUtil::node_att_as_string(dnode, "class");

        if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
          cerr << variables->get_id() << " is actually a classname: " 
               << classname << "\n";
        }

        // If the string in the xml file is proceeded with a $, then
        // its a variable dereference
        while (classname[0] == '$' &&
               variables->maybe_get_string
               (classname.substr(1,classname.length()-1), classname)) 
        {
          if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
            cerr << variables->get_id() << " is actually a classname: " 
                 << classname << "\n";
          }
        }

        // Iteratre to the next nested definition, if there is one...
        dnode = find_definition(definitions, classname);
      }


      // Spin through the xml var nodes and set their values
      for (merged_nodes_t::iterator mnode = merged_nodes.begin(); 
           mnode != merged_nodes.end(); ++mnode) {        
        for (xmlNode *cnode = (*mnode)->children; cnode; cnode = cnode->next) {
          if (XMLUtil::node_is_element(cnode, "var")) {
            eval_var_node(cnode, variables);
          }
        }
      }

      // Now we have Variables, Create the Object!
      Drawable * object = 0;

      if (root_node) {
        object = new Root(variables);
      } else {
      
        // First, see if the current catchers can create and throw back 
        // an object of type "classname"
        string makerstr = classname+"_Maker";
        // The special Signal to ask for a maker is created
        event_handle_t find_maker = new MakerSignal(makerstr, variables);
        // And thrown to the Catcher...
        event_handle_t catcher_return = 
          SignalThrower::throw_signal(catchers, find_maker);
        // And we see what the cather returned
        MakerSignal *made = dynamic_cast<MakerSignal*>(catcher_return.get_rep());
        if (made && made->get_signal_name() == (makerstr+"_Done")) {
          // It returned a Maker that we wanted... Hooray!
          object = dynamic_cast<Drawable *>(made->get_signal_thrower());
        } else {

          // Search the static_makers table and see if it contains
          // the classname we want....DEPRECIATED, maybe goes away?
          if (makers_.find(classname) != makers_.end()) {
            object = (*makers_[classname])(variables);
          }
        }
      }
      
      // At this point, the if the object is uninstantiatable, return
      if (!object) {
        delete variables;
        cerr << "Skinner::XMLIO::eval_object_node - object class=\"" 
             << classname << "\" cannot find maker\n";
        return 0;
      }

      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << object->get_id() << " - adding catcher";
      }

      // Now the object is created, fill the catchersondeck tree
      // with targets contained by the new object.
      SignalCatcher::TargetIDs_t catcher_targets =object->get_all_target_ids();
      SignalCatcher::TargetIDs_t::iterator id_iter;
      for(id_iter  = catcher_targets.begin();
          id_iter != catcher_targets.end(); ++id_iter) {
        const string &id = *id_iter;
        SignalThrower::Catcher_t catcher_target = 
          make_pair(object, object->get_catcher(id));
        catchers[id].first.push_front(catcher_target);
      }


      // Now the Catchers On Deck are ready, look if the xml file has
      // created any signals to hookup
      for (merged_nodes_t::iterator mnode = merged_nodes.begin();
           mnode != merged_nodes.end(); ++mnode) {
        for (xmlNode *cnode = (*mnode)->children; cnode; cnode = cnode->next) {
          if (XMLUtil::node_is_element(cnode, "signal")) {
            eval_signal_node(cnode, object, catchers);
          } 
        }
      }
         

      // Search for definitions
      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << object->get_id() << " - adding definitions:";
      }

      eval_merged_object_nodes_and_push_definitions(merged_nodes, definitions);
                               
      // Time to look for children object nodes
      Drawables_t children(0);
      bool have_unwanted_children = false;
      Parent *parent = dynamic_cast<Parent *>(object);
      // Search the merged nodes for object nodes.
      for (merged_nodes_t::iterator mnode = merged_nodes.begin(); 
           mnode != merged_nodes.end(); ++mnode) {        
        for (xmlNode *cnode = (*mnode)->children; cnode; cnode = cnode->next) {
          if (XMLUtil::node_is_element(cnode, "object")) {
            if (parent) { 
              Drawable *child = 
                eval_object_node(cnode, variables, definitions, catchers);
              if (child) {
                children.push_back(child);
              }
            } else {
              have_unwanted_children = true;
            }
          }
        }
      }
      
      if (parent && children.size()) {
        parent->set_children(children);
      }
            
      if (have_unwanted_children) { 
          cerr << "class : " << classname << " does not allow <object>\n";
      }
      

      // We are done w/ local definitions as all children have been made
      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << object->get_id() << " - popping definitions\n";
      }
      definitions.pop_back();

      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << object->get_id() << " - removing catcher";
      }


      // After having children, this node is done/ w/ local catcher targets, 
      // Re-get all object ids, as we may have pushed 
      // some aliases during eval_signal_node
      catcher_targets = object->get_all_target_ids();
      for(id_iter  = catcher_targets.begin();
          id_iter != catcher_targets.end(); ++id_iter) {
        const string &id = *id_iter;
        ASSERTMSG(!catchers[id].first.empty(), "Catchers Empty!");
        catchers[id].first.pop_front();
      }
      
      return object;
    }


    Root *
    XMLIO::eval_skinner_node(const xmlNodePtr node, const string &id)
    {
      ASSERT(XMLUtil::node_is_element(node, "skinner"));
      definition_nodes_t definitions;
      SignalThrower::SignalCatchers_t catchers;
      
      Drawable * object = eval_object_node(node, 0, definitions, catchers);
      Skinner::Root *root = dynamic_cast<Skinner::Root*>(object);
      ASSERT(root);
      return root;
    }

    void
    XMLIO::eval_definition_node(const xmlNodePtr node,
                                string_node_map_t &definitions)
    {
      ASSERT(XMLUtil::node_is_element(node, "definition"));
      string classname = XMLUtil::node_att_as_string(node, "class");
      if (sci_getenv_p("SKINNER_XMLIO_DEBUG")) {
        cerr << classname << ", ";
      }

      for (xmlNode *cnode=node->children; cnode!=0; cnode=cnode->next) {
        if (XMLUtil::node_is_element(cnode, "object")) {
          definitions[classname] = cnode;
        } 
      }
    }

    void
    XMLIO::eval_var_node(const xmlNodePtr node,
                         Variables *variables) 
    {
      ASSERT(XMLUtil::node_is_element(node, "var"));
      ASSERT(variables);
      bool propagate = 
        XMLUtil::node_att_as_string(node, "propagate") == "yes" ? true : false;
      bool overwrite = 
        XMLUtil::node_att_as_string(node, "overwrite") == "yes" ? true : false;

      const string varname = XMLUtil::node_att_as_string(node, "name");

      if (!overwrite && variables->exists(varname)) {
        return;
      }
      
      const char *contents = 0;
      if (node->children) { 
        contents = XMLUtil::xmlChar_to_char(node->children->content);
      }
      const string value = contents ? contents : "";
      const string typestr = XMLUtil::node_att_as_string(node, "type");
      variables->insert(varname, value, typestr, propagate);
    }
                        



    void
    XMLIO::eval_signal_node(const xmlNodePtr node,
                            Drawable *object,
                            SignalThrower::SignalCatchers_t &catchers)
      //                            TargetSignalMap_t &signals) 
    {
      ASSERT(XMLUtil::node_is_element(node, "signal"));
      string signalname = XMLUtil::node_att_as_string(node, "name");
      string signaltarget = XMLUtil::node_att_as_string(node, "target");
      SignalThrower *thrower = dynamic_cast<SignalThrower *>(object);
      SignalCatcher *thrower_as_catcher = dynamic_cast<SignalCatcher*>(object);
      ASSERT(thrower);
      ASSERT(thrower_as_catcher);

      SignalThrower::SignalCatchers_t::iterator catchers_iter = 
        catchers.find(signaltarget);

      if (catchers_iter == catchers.end()) {
        cerr << "Signal " << signalname 
             << " cannot find target " << signaltarget << std::endl;
        return;
      }

      const char *node_contents = 0;
      if (node->children) { 
        node_contents = XMLUtil::xmlChar_to_char(node->children->content);
        if (node_contents && catchers_iter->second.second == "") {
          catchers_iter->second.second = node_contents;
        }
      }

      SignalThrower::CatchersOnDeck_t &signal_catchers = catchers_iter->second.first;
      SignalThrower::CatchersOnDeck_t::iterator catcher_iter = 
        signal_catchers.begin();
      for (;catcher_iter != signal_catchers.end(); ++catcher_iter) {
        SignalCatcher *catcher = catcher_iter->first;
        SignalCatcher::CatcherFunctionPtr function = catcher_iter->second;
        //          catcher->get_catcher(signaltarget);
        ASSERT(function);
          
          if (!thrower->hookup_signal_to_catcher_target
              (signalname, catchers_iter->second.second, catcher, function) &&
              object->catcher_targets_.find(signalname) == object->catcher_targets_.end())
          {
//             cerr << "Translating signal: " 
//                  << signalname << " to: " << signaltarget << std::endl;
            catchers[signalname].first.push_front(make_pair(catcher, function));
            catchers[signalname].second = catchers_iter->second.second;

            object->catcher_targets_[signalname] = function;
            //            cerr << object->get_id() << " now has catcher targets: ";
            thrower_as_catcher->get_all_target_ids();
          }
      }
    }
        
    void
    XMLIO::register_maker(const string &name, DrawableMakerFunc_t *maker)
    {
      makers_[name] = maker;
    }





  }
}
