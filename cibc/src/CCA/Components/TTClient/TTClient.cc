/*
  For more information, please see: http://software.sci.utah.edu

  The MIT License

  Copyright (c) 2004 Scientific Computing and Imaging Institute,
  University of Utah.

  
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

#include <CCA/Components/TTClient/TTClient.h>
#include <Core/Thread/Time.h>
#include <sci_wx.h>
#include <iostream>

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
}

int ttUIPort::ui()
{
#if HAVE_GUI
  int res = (int) wxGetNumberFromUser( wxT("Ping pong test between TTClient and TableTennis."),
                                       wxT("Iterate count (from [1, 1000]):"),
                                       wxT("TTClient"),
                                       TTCl->getCount(), 1, 1000);
  if (res > 0) {
    TTCl->setCount(res);
  }
#else
  std::cerr << "UI not available, using default (count = " << TTCl->getCount() << ")." << std::endl;
#endif
  return 0;
}


int ttGoPort::go()
{
  sci::cca::Services::pointer services = TTCl->getServices();
#if HAVE_GUI
  sci::cca::ports::GUIService::pointer guiService;
  try {
    guiService = pidl_cast<sci::cca::ports::GUIService::pointer>(services->getPort("cca.GUIService"));
    if (guiService.isNull()) {
      wxMessageBox(wxT("GUIService is not available"), wxT("TTClient"), wxOK|wxICON_ERROR, 0);
      return -2;
    }
  }
  catch (const sci::cca::CCAException::pointer &e) {
    wxMessageBox(e->getNote(), wxT("TTClient"), wxOK|wxICON_ERROR, 0);
  }
  sci::cca::ComponentID::pointer cid = services->getComponentID();
#endif

  PP::PingPong::pointer PPptr;
  try {
    sci::cca::Port::pointer pp = services->getPort("tt");
    PPptr = pidl_cast<PP::PingPong::pointer>(pp);
  }
  catch (const sci::cca::CCAException::pointer &e) {
#if HAVE_GUI
    wxMessageBox(e->getNote(), wxT("TTClient"), wxOK|wxICON_ERROR, 0);
#else
    std::cout << e->getNote() << std::endl;
#endif
    return -1;
  }

  const int mi = TTCl->getCount();
  double start = Time::currentSeconds();
  for (int i = 0; i < mi; i++) {
    int retval = PPptr->pingpong(i);
#if HAVE_GUI
  guiService->updateProgress(cid, int(i / mi));
#endif
    std::cout << "Pingpong: retval = " << retval << std::endl;
  }
  double elapsed = Time::currentSeconds() - start;
  std::ostringstream stm;
  stm << mi << " reps in " << elapsed << " seconds" << std::endl;
  double us = elapsed / mi * 1000 * 1000;
  stm << us << " us/rep" << std::endl;

#if HAVE_GUI
  wxMessageBox(wxT(stm.str().c_str()), wxT("TTClient"), wxOK|wxICON_INFORMATION, 0);
  guiService->updateProgress(cid, 100);
  services->releasePort("cca.GUIService");
#else
  std::cerr << stm.str() << std::endl;
#endif

  services->releasePort("tt");
  services->releasePort("progress");

  return 0;
}
