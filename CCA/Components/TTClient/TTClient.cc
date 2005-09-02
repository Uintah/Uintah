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


/*
 *  TTClient.cc:
 *
 *  Written by:
 *   Kosta Damevski
 *   Department of Computer Science
 *   University of Utah
 *   February 2003
 *
 */

#include <sci_defs/qt_defs.h>
#include <CCA/Components/TTClient/TTClient.h>
#include <CCA/Components/Builder/QtUtils.h>
#include <Core/Thread/Time.h>
#include <iostream>

#if HAVE_QT
 #include <qmessagebox.h>
 #include <qinputdialog.h>
 #include <qstring.h>
#endif

using namespace std;
using namespace SCIRun;

extern "C" sci::cca::Component::pointer make_SCIRun_TTClient()
{
  return sci::cca::Component::pointer(new TTClient());
}


TTClient::TTClient() : count(12)
{
}

TTClient::~TTClient()
{
  services->removeProvidesPort("ui");
  services->removeProvidesPort("go");
  services->unregisterUsesPort("tt");
  services->unregisterUsesPort("progress");
}

void TTClient::setServices(const sci::cca::Services::pointer& svc)
{
  services = svc;
  sci::cca::TypeMap::pointer props = svc->createTypeMap();

  ttUIPort *uip = new ttUIPort();
  uip->setParent(this);
  ttUIPort::pointer uiPortPtr(uip);
  svc->addProvidesPort(uiPortPtr, "ui", "sci.cca.ports.UIPort", props);

  ttGoPort *gop = new ttGoPort();
  gop->setParent(this);
  ttGoPort::pointer goPortPtr(gop);
  svc->addProvidesPort(goPortPtr, "go", "sci.cca.ports.GoPort", props);

  svc->registerUsesPort("tt", "sci.cca.ports.TTPort", props);
    svc->registerUsesPort("progress","sci.cca.ports.Progress", props);
}

int ttUIPort::ui() 
{
#if HAVE_QT
    bool ok;
    int res = QInputDialog::getInteger("TTClient", "TTClient iterate count (from [0, 1000]):",
        TTCl->getCount(), 0, 1000, 1, &ok);
    if (ok) {
      TTCl->setCount(res);
    }
#else
    std::cerr << "UI not available, using default (count = " << count << ")." << std::endl;
#endif
  return 0;
}


int ttGoPort::go() 
{
  PP::PingPong::pointer PPptr;
  sci::cca::ports::Progress::pointer pPtr;
  try {
      sci::cca::Port::pointer pp = TTCl->getServices()->getPort("tt");
      PPptr = pidl_cast<PP::PingPong::pointer>(pp);
      sci::cca::Port::pointer progPort = TTCl->getServices()->getPort("progress");
      pPtr = pidl_cast<sci::cca::ports::Progress::pointer>(progPort);
  }
  catch (const sci::cca::CCAException::pointer &e) {
#if HAVE_QT
    QMessageBox::critical(0, "TTClient", e->getNote());
#else
    cout << e->getNote() << endl;
#endif
    return 1;
  }

  const int mi = TTCl->getCount();
  double start = Time::currentSeconds();
  for (int i = 0; i < mi; i++) {
      int retval = PPptr->pingpong(i);
      pPtr->updateProgress(i+1, mi);
      cout << "Pingpong: retval = " << retval << endl;
  }
  double elapsed = Time::currentSeconds() - start;
  ostringstream stm;
  stm << mi << " reps in " << elapsed << " seconds" << std::endl;
  double us = elapsed / mi * 1000 * 1000;
  stm << us << " us/rep" << std::endl;

#if HAVE_QT
  QMessageBox::information(0, "TTClient", stm.str());
#else
  std::cerr << stm.str();
#endif

  TTCl->getServices()->releasePort("tt");
  TTCl->getServices()->releasePort("progress");
  return 0;
}
