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
//    File   : StreamReader.cc
//    Author : Martin Cole
//    Date   : Tue Aug 15 14:16:14 2006

  
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Malloc/Allocator.h>



#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/StringUtil.h>
#include <Packages/DDDAS/share/share.h>
#include <Core/Thread/Runnable.h>
#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h> 
#include <Core/Thread/ConditionVariable.h>

#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/GenericField.h>

#include <iostream>
#include <fstream>
#include <assert.h>
#include <sys/types.h>

namespace DDDAS {

using namespace SCIRun;

class DDDASSHARE StreamReader : public Module {

public:
  //! Virtual interface
  StreamReader(GuiContext* ctx);

  virtual ~StreamReader();
  virtual void execute();

private:

  //! GUI variables
  GuiString brokerip_;
  GuiInt brokerport_;
  GuiString groupname_;
  GuiInt listenport_;
};


DECLARE_MAKER(StreamReader);

StreamReader::StreamReader(GuiContext* ctx) : 
  Module("StreamReader", ctx, Source, "DataIO", "DDDAS"),
  brokerip_(get_ctx()->subVar("brokerip")),   
  brokerport_(get_ctx()->subVar("brokerport")),   
  groupname_(get_ctx()->subVar("groupname")),   
  listenport_(get_ctx()->subVar("listenport"))
{  
  cout << "(StreamReader::StreamReader) Inside" << endl;  
}


StreamReader::~StreamReader()
{
}

void StreamReader::execute()
{
  cout << "(StreamReader::execute) Inside" << endl;
}


} // End namespace DDDAS




