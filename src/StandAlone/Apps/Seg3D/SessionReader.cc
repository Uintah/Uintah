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
//    File   : SessionReader.cc
//    Author : McKay Davis
//    Date   : Tue Oct 17 15:55:03 2006

#include <StandAlone/Apps/Seg3D/SessionReader.h>
#include <Core/XMLUtil/XMLUtil.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Util/Environment.h>
#include <Core/Util/Assert.h>
#include <Core/Skinner/XMLIO.h>
#include <Core/Skinner/Variables.h>
#include <libxml/xmlreader.h>
#include <libxml/catalog.h>
#include <libxml/xinclude.h>
#include <libxml/xpathInternals.h>
#include <iostream>



namespace SCIRun {

SessionReader::SessionReader(Painter *painter) :
  painter_(painter),
  dir_()
{
}

SessionReader::~SessionReader()
{
}


bool
SessionReader::load_session(string filename) {
  /*
   * this initialize the library and check potential ABI mismatches
   * between the version it was compiled for and the actual shared
   * library used.
   */
  filename = substituteTilde(filename);
  pair<string, string> dir_file = split_filename(filename);
  dir_ = dir_file.first;
  LIBXML_TEST_VERSION;
  
  /* the parser context */
  xmlParserCtxtPtr ctxt = xmlNewParserCtxt();
  if (!ctxt) {
    std::cerr << "SessionReader failed xmlNewParserCtx()\n";
    return false;
  }
      
  /* parse the file, activating the DTD validation option */
  xmlDocPtr doc = 
    xmlCtxtReadFile(ctxt, filename.c_str(), 0, XML_PARSE_PEDANTIC);
  
  if (!doc) {
    std::cerr << "Skinner::XMLIO::load failed to parse " 
              << filename << std::endl;
    return false;
  } 

  // parse the doc at network node.
  NrrdVolumes volumes;
  for (xmlNode *cnode=doc->children; cnode!=0; cnode=cnode->next) {
    if (XMLUtil::node_is_element(cnode, "Seg3D")) {
      eval_seg3d_node(cnode);
    } 
    else if (!XMLUtil::node_is_comment(cnode))
      throw "Unknown node type";
  }               
  
  xmlFreeDoc(doc);
  xmlFreeParserCtxt(ctxt);  
  xmlCleanupParser();


  return true;
}

void
SessionReader::eval_seg3d_node(const xmlNodePtr node)
{
  NrrdVolumes volumes;
  for (xmlNode *cnode=node->children; cnode!=0; cnode=cnode->next) {
    if (XMLUtil::node_is_element(cnode, "volume")) {
      NrrdVolumeHandle vol = eval_volume_node(cnode, 0);
      if (vol.get_rep()) {
        volumes.push_back(vol);
      }
    } 
  }

  if (volumes.size()) {
    painter_->volumes_ = volumes;
    painter_->current_volume_ = volumes.back();
    painter_->extract_all_window_slices();
    painter_->rebuild_layer_buttons();
    painter_->redraw_all();
  }
}




NrrdVolumeHandle
SessionReader::eval_volume_node(const xmlNodePtr node, NrrdVolumeHandle parent)
{
  Skinner::Variables *vars = new Skinner::Variables("");
  for (xmlNode *cnode=node->children; cnode!=0; cnode=cnode->next) {
    if (XMLUtil::node_is_element(cnode, "var")) {
      Skinner::XMLIO::eval_var_node(cnode, vars);
    } 
  }
  NrrdVolumeHandle volume = 0;


  unsigned int label = 0;
  if (vars->exists("label")) {
    label = vars->get_int("label");
  }

  if (!parent.get_rep()) {
    string filename = vars->get_string("filename");  
    pair<string, string> dir_file = split_filename(filename);
    if (dir_file.first.empty()) dir_file.first = dir_;
    if (!label) {
      volume = painter_->load_volume<float>(dir_file.first+filename);
    } else {
      volume = painter_->load_volume<unsigned int>(dir_file.first+filename);
    }
      
    if (!volume.get_rep()) {
      cerr << "Error loading : " << filename << std::endl;
      return 0;
    }
    volume->filename_ = filename;
  } else {
    volume = parent->create_child_label_volume(vars->get_int("label"));
  }


  volume->label_ = label;


  if (vars->exists("name")) {
    volume->name_ = vars->get_string("name");
  }

  if (vars->exists("opacity")) {
    volume->opacity_ = vars->get_double("opacity");
  }


  for (xmlNode *cnode=node->children; cnode!=0; cnode=cnode->next) {
    if (XMLUtil::node_is_element(cnode, "volume")) {
      eval_volume_node(cnode, volume);
    } 
  }

  return volume;
}



} // end namespace SCIRun
