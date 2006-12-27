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

#include <libxml/xmlreader.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Util/Environment.h>
#include <Core/OS/Dir.h>
#include <Packages/CardioWaveInterface/Core/XML/NeuroWaveXML.h>

#include <stdio.h>
#include <stdlib.h>

#include <iostream>

#ifndef _WIN32
#include <unistd.h>
#include <dirent.h>
#endif

namespace CardioWaveInterface {

using namespace SCIRun;

NeuroWaveXML::NeuroWaveXML()
{
  parse_xml_files();
}

bool NeuroWaveXML::parse_xml_files()
{
  const char *srcdir = sci_getenv("SCIRUN_SRCDIR");

  std::string xmldir = std::string(srcdir) + "/Packages/CardioWaveInterface/Core/XML";
  std::vector<std::string> files;

  DIR* dir = opendir(xmldir.c_str());
  if (dir)
  {
   dirent* file = readdir(dir);
   while (file)
   {
     std::string filename(file->d_name);
     if (filename.size() > 4)
     {
       if (filename.substr(filename.size()-4,4) == std::string(".xml"))
       {
        files.push_back(xmldir + "/" + filename);
       }
     }
     file = readdir(dir);
   }
   closedir(dir);
  }
  
  NeuroWaveItem item;
  item.name = "NONE";
  list_.push_back(item);
  
  for (size_t p = 0; p < files.size(); p++)
  {
    add_file(files[p]);
  }
  
  return (true);
}

bool NeuroWaveXML::add_file(std::string filename)
{

  LIBXML_TEST_VERSION;
  
  xmlParserCtxtPtr ctxt; /* the parser context */
  xmlDocPtr doc; /* the resulting document tree */

  if(!(ctxt = xmlNewParserCtxt()))
  {
    std::cerr << "XMLError: could not create XML context" << std::endl;
    return (false);
  }
  
  doc = xmlCtxtReadFile(ctxt, filename.c_str(), 0, (XML_PARSE_DTDATTR |XML_PARSE_DTDVALID | XML_PARSE_NOERROR));

  if (doc == 0 || ctxt->valid == 0) 
  {
    xmlError* error = xmlCtxtGetLastError(ctxt);
    xmlFreeParserCtxt(ctxt);  
    std::cerr << "XMLError for file '" << filename << "': " << error->message << std::endl;
    return (false);
  } 

  xmlNodePtr cnode = doc->children;

  for (; cnode != 0; cnode = cnode->next) 
  {
    if (cnode->type == XML_ELEMENT_NODE && std::string(to_char_ptr(cnode->name)) == std::string("cw")) 
    {
      xmlNodePtr node = cnode->children;
  
      for (; node != 0; node = node->next) 
      {
        if (node->type == XML_ELEMENT_NODE && std::string(to_char_ptr(node->name)) == std::string("neurowave")) 
        {
          NeuroWaveItem item;
          xmlNodePtr inode = node->children;
          for (;inode != 0; inode = inode->next)
          {
            if (std::string(to_char_ptr(inode->name)) == std::string("name"))
            {
              item.name = get_serialized_children(inode);
            }
            if (std::string(to_char_ptr(inode->name)) == std::string("file"))
            {
              item.file = get_serialized_children(inode);
            }
            if (std::string(to_char_ptr(inode->name)) == std::string("parameters"))
            {
              item.parameters = get_serialized_children(inode);
            }
            if (std::string(to_char_ptr(inode->name)) == std::string("description"))
            {
              item.description = get_serialized_children(inode);
            }            
          }
          list_.push_back(item);
        } 
        
        if (node->type == XML_ELEMENT_NODE && std::string(to_char_ptr(node->name)) == std::string("defaultneurowave"))
        {
          xmlNodePtr inode = node->children;
          for (;inode != 0; inode = inode->next)
          {
            if (std::string(to_char_ptr(inode->name)) == std::string("name"))
            {
              default_name_ = get_serialized_children(inode);
            }
          }
        }
      }
    }
  }
  
  xmlFreeDoc(doc);
  xmlFreeParserCtxt(ctxt);  
  xmlCleanupParser();
  
  return (true);
}

std::vector<std::string> NeuroWaveXML::get_names()
{
  std::vector<std::string> names(list_.size());
  for (size_t p=0; p < names.size();p++) names[p] = list_[p].name;
  return (names);
}

std::string NeuroWaveXML::get_default_name()
{
  if (default_name_.size()) return(default_name_);
  return ("NONE");
}

NeuroWaveItem NeuroWaveXML::get_neurowave()
{
  NeuroWaveItem item;
  if (list_.size() > 1) item = list_[1];
  return (item);
}

} // end namespace
