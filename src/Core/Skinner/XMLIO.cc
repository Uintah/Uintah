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
#include <Core/Skinner/Box.h>
#include <Core/Skinner/Collection.h>
#include <Core/Skinner/Grid.h>
#include <Core/Skinner/Layout.h>
#include <Core/Skinner/Text.h>
#include <Core/Skinner/Texture.h>
#include <Core/Skinner/Window.h>

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

    Drawables_t
    XMLIO::load(const string &filename)
    {
      Drawables_t objs;
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
        return objs;
      }

      /* parse the file, activating the DTD validation option */
      doc = xmlCtxtReadFile(ctxt, filename.c_str(), 0, (XML_PARSE_DTDATTR | 
                                                        XML_PARSE_DTDVALID | 
                                                        XML_PARSE_PEDANTIC));
      /* check if parsing suceeded */
      if (!doc) {
        std::cerr << "Skinner::XMLIO::load failed to parse " 
                  << filename << std::endl;
        return objs;
      } 
      if (!ctxt->valid) {
          std::cerr << "Skinner::XMLIO::load dailed to validate " 
                    << filename << std::endl;
          return objs;
      }
      

      // parse the doc at network node.

      for (xmlNode *cnode=doc->children; cnode!=0; cnode=cnode->next) {
        if (XMLUtil::node_is_dtd(cnode, "skinner")) 
          continue;
        if (XMLUtil::node_is_element(cnode, "skinner")) 
          objs = eval_skinner_node(cnode, filename);
        else if (!XMLUtil::node_is_comment(cnode))
          throw "Unknown node type";
      }               

      //Drawable *obj = 0;
      xmlFreeDoc(doc);
      /* free up the parser context */
      xmlFreeParserCtxt(ctxt);  
      xmlCleanupParser();

      return objs;
    }
    
    Drawable *
    XMLIO::eval_object_node(const xmlNodePtr node, 
                            Variables *variables,
                            string_node_map_t &definitions,
                            bool instantiating_definition) {

      ASSERT(XMLUtil::node_is_element(node, "object"));

      string classname = XMLUtil::node_att_as_string(node, "class");

      string_node_map_t::iterator definition = definitions.find(classname);
      DrawableMakerMap_t::iterator maker_info = makers_.find(classname);
      
      // Determine if this object is instantiatable, return 0 if not
      if (definition == definitions.end() && maker_info == makers_.end()) {
        cerr << "Skinner::XMLIO::eval_object_node - object class=\"" 
             << classname << "\" not found in makers or definitions.\n";
        return 0;
      }

      if (!instantiating_definition) {
        // Spawn sub-vars for this class
        // Object id is not required, create unique one if not found
        string id = "";
        if (!XMLUtil::maybe_get_att_as_string(node, "id", id)) {
          static int objcount = 0;
          id = classname+"."+to_string(objcount++);
        }

        variables = variables->spawn(id);
      }

      for (xmlNode *cnode = node->children; cnode; cnode = cnode->next) {
        if (XMLUtil::node_is_element(cnode, "var")) {
          eval_var_node(cnode, variables);
        } 
      }

      if (definition != definitions.end()) {
        return eval_object_node(definition->second, variables, 
                                definitions, true);
      }         

      Drawables_t children(0);
      for (xmlNode *cnode = node->children; cnode; cnode = cnode->next) {
        if (XMLUtil::node_is_element(cnode, "object")) {
          Drawable *child = eval_object_node(cnode, variables, definitions);
          if (child) {
            children.push_back(child);
          }
        }        
      }               

      DrawableMakerFunc_t &maker = *(maker_info->second.first);
      void *data = maker_info->second.second;
      return maker(variables, children, data);
    }


    vector<Drawable *>
    XMLIO::eval_skinner_node(const xmlNodePtr node, const string &id)
    {
      ASSERT(XMLUtil::node_is_element(node, "skinner"));
      Drawables_t children;
      string_node_map_t definitions;
      Variables *variables = new Variables(id);

      for (xmlNode *cnode=node->children; cnode!=0; cnode=cnode->next) {
        if (XMLUtil::node_is_element(cnode, "definition")) {
          eval_definition_node(cnode, definitions);
        } else if (XMLUtil::node_is_element(cnode, "object")) {
          children.push_back(eval_object_node(cnode, variables, definitions));
        } 
      }
      ASSERT(!children.empty());
      return children;
    }

    void
    XMLIO::eval_definition_node(const xmlNodePtr node,
                                string_node_map_t &definitions)
    {
      ASSERT(XMLUtil::node_is_element(node, "definition"));
      string classname = XMLUtil::node_att_as_string(node, "class");

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

      variables->insert(XMLUtil::node_att_as_string(node, "name"),
                        XMLUtil::xmlChar_to_char(node->children->content),
                        XMLUtil::node_att_as_string(node, "type"),
                        propagate);
    }
                        
      
      
    

    void
    XMLIO::register_maker(const string &name,
                          DrawableMakerFunc_t *maker,
                          void *data)
    {
      makers_[name] = make_pair(maker, data);
    }





  }
}
