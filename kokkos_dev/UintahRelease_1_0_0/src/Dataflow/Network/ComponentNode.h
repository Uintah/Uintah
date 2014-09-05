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
//    File   : ComponentNode.h
//    Original Author: Chris Moulding Sep, 2000
//    Recent Author : Martin Cole
//    Date   : Wed Nov 23 08:26:19 2005

#if !defined(ComponentNode_H)
#define ComponentNode_H

#include <Dataflow/Network/PackageDB.h>

namespace SCIRun {
class GuiInterface;

//! write xml file from info in the ModuleInfo.
void write_component_file(const ModuleInfo &mi, const char* filename);

//! read the xml file and set all the ModuleInfo needed for instantiation.
//! return false if expected nodes are empty. (error with the xml file)
bool read_component_file(ModuleInfo &mi, const char* filename);
} // End namespace SCIRun

#endif //ComponentNode_H
