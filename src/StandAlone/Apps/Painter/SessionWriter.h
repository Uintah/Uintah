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
//    File   : SessionWriter.h
//    Author : McKay Davis
//    Date   : Tue Oct 17 21:25:45 2006

#ifndef LEXOV_SessionWriter_h
#define LEXOV_SessionWriter_h

#include <StandAlone/Apps/Painter/NrrdVolume.h>
#include <libxml/xmlreader.h>
#include <string>

namespace SCIRun {
using std::string;
class SessionWriter {
public:
  static bool   write_session(const string &filename, NrrdVolumes &);
private:
  static void   add_volume_nodes(xmlNodePtr node, NrrdVolumes &volumes);
  static void   add_var_node(xmlNodePtr node, const string &, const string &);
};


#endif


} // end namespace SCIRun
