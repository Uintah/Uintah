
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
  ciPort.setParent(this);
}

ZListWriter::~ZListWriter()
{
    services->unregisterUsesPort("listport");
    services->unregisterUsesPort("progress");
}

void ZListWriter::setServices(const sci::cca::Services::pointer& svc)
{
  services=svc;
  //register provides ports here ...

  sci::cca::TypeMap::pointer props = svc->createTypeMap();
  uiPort = new ZLUIPort(services);
  uiPort->setParent(this);
  ZLUIPort::pointer uip(uiPort);
  ZLComponentIcon::pointer cip(&ciPort);

  svc->addProvidesPort(uip,"ui","sci.cca.ports.UIPort", props);
  svc->addProvidesPort(cip, "icon",
                       "sci.cca.ports.ComponentIcon", props);
  svc->registerUsesPort("listport","ZListPort", props);

  svc->registerUsesPort("progress","sci.cca.ports.Progress",
    sci::cca::TypeMap::pointer(0));
}

int ZLUIPort::ui()
{
  sci::cca::Port::pointer pp=services->getPort("listport"); 
  if(pp.isNull()){
    QMessageBox::warning(0, "ListPlotter", "listport is not available!");
    return 1;
  }  
  sci::cca::ports::ZListPort::pointer lport =
      pidl_cast<sci::cca::ports::ZListPort::pointer>(pp);
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

  //sci::cca::Port::pointer progPort = services->getPort("progress");	
  //if (progPort.isNull()) {
  //  std::cerr << "progress is not available!\n";
  //  return 1;
  //}  
  //sci::cca::ports::Progress::pointer pPtr =
  //  pidl_cast<sci::cca::ports::Progress::pointer>(progPort);

  std::ofstream saveOutputFile(filename);
  for (unsigned int i = 0; i < data.size(); i++) {
  //    pPtr->updateProgress(i, data.size());
      saveOutputFile << data[i] << std::endl;
  }
  saveOutputFile.close();

  //services->releasePort("progress");
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
