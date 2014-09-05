/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  License for the specific language governing rights and limitations under
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

#include <CCA/Components/ZListWriter/ZListWriter.h>
#include <sci_wx.h>

#include <iostream>

extern "C" sci::cca::Component::pointer make_SCIRun_ZListWriter()
{
  return sci::cca::Component::pointer(new ZListWriter());
}


ZListWriter::ZListWriter()
{
}

ZListWriter::~ZListWriter()
{
  services->unregisterUsesPort("listport");
  services->removeProvidesPort("ui");
  services->removeProvidesPort("icon");
}

void ZListWriter::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  //register provides ports here ...

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  ZLUIPort *uip = new ZLUIPort(services);
  uip->setParent(this);
  ZLUIPort::pointer uiPortPtr = ZLUIPort::pointer(uip);

  svc->addProvidesPort(uiPortPtr, "ui", "sci.cca.ports.UIPort", props);

  ZLComponentIcon::pointer ciPortPtr =
    ZLComponentIcon::pointer(new ZLComponentIcon);

  svc->addProvidesPort(ciPortPtr, "icon", "sci.cca.ports.ComponentIcon", props);
  svc->registerUsesPort("listport","ZListPort", props);
}

int ZLUIPort::ui()
{
  sci::cca::ports::ZListPort::pointer lport;
  try {
    sci::cca::Port::pointer pp = services->getPort("listport");
    lport =
      pidl_cast<sci::cca::ports::ZListPort::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("ZListWriter"), wxOK|wxICON_ERROR, 0);
    return -1;
  }
  SSIDL::array1<double> data = lport->getList();
  services->releasePort("listport");

  wxString filename = wxFileSelector(wxT("Save ZList file"),  wxT(""), wxT(""), wxT(".lst"), wxT("ZList File (*.lst)"), wxSAVE|wxOVERWRITE_PROMPT);
  if (filename.IsEmpty()) {
    return -1;
  }
  if (filename.Find(".lst") == -1) {
    filename += ".lst";
  }

  std::ofstream saveOutputFile(filename.c_str());
  for (unsigned int i = 0; i < data.size(); i++) {
    saveOutputFile << data[i] << std::endl;
  }
  saveOutputFile.close();

  return 0;
}
