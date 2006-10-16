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
//    File   : BaseTool.cc
//    Author : Martin Cole
//    Date   : Thu May 25 21:08:59 2006

#include <Core/Events/BaseEvent.h>
#include <Core/Events/Tools/BaseTool.h>
#include <string>
#include <iostream>

namespace SCIRun {

using namespace std;

BaseTool::BaseTool(string name) :
  ref_cnt(0),
  name_(name)
{
}

BaseTool::~BaseTool()
{
}

PointerTool::PointerTool(string name) :
  BaseTool(name)
{
}

PointerTool::~PointerTool()
{
}

KeyTool::KeyTool(string name) :
  BaseTool(name)
{
}

KeyTool::~KeyTool()
{
}

WindowTool::WindowTool(string name) :
  BaseTool(name)
{
}

WindowTool::~WindowTool()
{
}

TMNotificationTool::TMNotificationTool(string name) :
  BaseTool(name)
{
}

TMNotificationTool::~TMNotificationTool()
{
}

CommandTool::CommandTool(string name) :
  BaseTool(name)
{
}

CommandTool::~CommandTool()
{
}

} // namespace SCIRun

