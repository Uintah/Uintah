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

#include <iostream>

#include <qapplication.h>
#include <qmessagebox.h>
#include <qfiledialog.h>
#include <qstring.h>

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

  svc->addProvidesPort(ciPortPtr, "icon",
                       "sci.cca.ports.ComponentIcon", props);
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
    QMessageBox::warning(0, "ZListWriter", e->getNote());
    return 1;
  }  
  SSIDL::array1<double> data = lport->getList();  
  services->releasePort("listport");

  QString filename;
  QString fn = QFileDialog::getSaveFileName(QString::null, "ZList File (*.lst)");
  if (fn.isEmpty()) {
    return 2;
  }
  if (fn.endsWith(".lst")) {
    filename = fn;
  } else {
    QString fnExt = fn + ".lst";
    filename = fnExt;
  }

  std::ofstream saveOutputFile(filename);
  for (unsigned int i = 0; i < data.size(); i++) {
    saveOutputFile << data[i] << std::endl;
  }
  saveOutputFile.close();

  return 0;
}


std::string ZLComponentIcon::getDisplayName()
{
    return "ZList Writer";
}

std::string ZLComponentIcon::getDescription()
{
    return "ZList Writer Component: write output read from a ZList component to file.";
}

int ZLComponentIcon::getProgressBar()
{
    return STEPS;
}
 
std::string ZLComponentIcon::getIconShape()
{
    return "RECT";
}
