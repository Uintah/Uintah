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
//    File   : SessionWriter.cc
//    Author : McKay Davis
//    Date   : Tue Oct 17 21:27:22 2006

#include <StandAlone/Apps/Painter/SessionWriter.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>
#include <Core/Util/FileUtils.h>
#include <Core/Skinner/XMLIO.h>
#include <Core/Skinner/Variables.h>
#include <libxml/xmlreader.h>
#include <libxml/catalog.h>
#include <libxml/xinclude.h>
#include <libxml/xpathInternals.h>
#include <iostream>



namespace SCIRun {

bool
SessionWriter::write_session(const string &filename, NrrdVolumes &volumes) {
  /*
   * this initialize the library and check potential ABI mismatches
   * between the version it was compiled for and the actual shared
   * library used.
   */

  pair<string, string> dir_file = split_filename(filename);
  if (!write_volumes(volumes, dir_file.first)) {
    return false;
  }
  
  LIBXML_TEST_VERSION;
  
  /* the parser context */
  xmlParserCtxtPtr ctxt = xmlNewParserCtxt();
  if (!ctxt) {
    std::cerr << "SessionWriter failed xmlNewParserCtx()\n";
    return false;
  }
      
  /* parse the file, activating the DTD validation option */
  xmlDocPtr doc = xmlNewDoc(to_xml_ch_ptr("1.0"));
  xmlNodePtr root = xmlNewNode(0,to_xml_ch_ptr("lexov"));
  xmlDocSetRootElement(doc, root);

  xmlNewProp(root, to_xml_ch_ptr("version"), to_xml_ch_ptr("1.0"));

  add_volume_nodes(root, volumes);
  
  xmlSaveFormatFileEnc(filename.c_str(), doc, "UTF-8", 1);

  xmlFreeDoc(doc);


  return true;
}


bool
SessionWriter::write_volumes(NrrdVolumes &volumes, const string &dir) {
  //  NrrdVolumes::reverse_iterator iter = volumes.rbegin();  
  //  NrrdVolumes::reverse_iterator end = volumes.rend();
  NrrdVolumes::iterator iter = volumes.begin();  
  NrrdVolumes::iterator end = volumes.end();

  for (; iter != end; ++iter) {
    NrrdVolumeHandle volume = *iter;
    pair<string, string> dir_file = split_filename(volume->filename_);

    if (!ends_with(string_tolower(dir_file.second), ".hdr")) {
      dir_file.second = dir_file.second + ".hdr";
    }

    volume->filename_ = dir_file.second;
    if (!volume->write(dir + "/" + volume->filename_)) return false;
  }
  return true;
}
    

void
SessionWriter::add_volume_nodes(xmlNodePtr node, NrrdVolumes &volumes) {
  //  NrrdVolumes::reverse_iterator iter = volumes.rbegin();  
  //  NrrdVolumes::reverse_iterator end = volumes.rend();
  //  NrrdVolumes::reverse_iterator iter = volumes.rbegin();  
  //  NrrdVolumes::reverse_iterator end = volumes.rend();
  NrrdVolumes::iterator iter = volumes.begin();  
  NrrdVolumes::iterator end = volumes.end();

  for (; iter != end; ++iter) {
    NrrdVolumeHandle volume = *iter;
    xmlNodePtr cnode = xmlNewChild(node, 0, to_xml_ch_ptr("volume"),0);
    add_var_node(cnode, "name", volume->name_);
    if (!volume->parent_.get_rep())
      add_var_node(cnode, "filename", volume->filename_);
    add_var_node(cnode, "label", to_string(volume->label_));
    add_var_node(cnode, "visible", to_string(volume->visible_ ? 1 : 0));
    add_var_node(cnode, "opacity", to_string(volume->opacity_));
    add_volume_nodes(cnode, volume->children_);
  }
}

void
SessionWriter::add_var_node(xmlNodePtr node, 
                            const string &name,
                            const string &value) {
  xmlNodePtr cnode = xmlNewChild(node, 0, to_xml_ch_ptr("var"), 
                                 to_xml_ch_ptr(value.c_str()));
  xmlNewProp(cnode, to_xml_ch_ptr("name"), to_xml_ch_ptr(name.c_str()));
}


} // end namespace SCIRun
