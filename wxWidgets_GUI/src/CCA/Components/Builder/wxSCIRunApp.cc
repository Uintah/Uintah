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

#include <CCA/Components/Builder/wxSCIRunApp.h>
#include <CCA/Components/Builder/GUIBuilder.h>
#include <CCA/Components/Builder/BuilderWindow.h>
#include <Core/Thread/Guard.h>
#include <Core/Util/Environment.h>

#include <wx/splash.h>

#ifndef DEBUG
#  define DEBUG 0
#endif

#define SPLASH_TIMEOUT 10000

namespace GUIBuilder {

using namespace SCIRun;

Mutex wxSCIRunApp::appLock("GUI application lock");
Semaphore wxSCIRunApp::sem("wxWidgets GUI Thread startup wait", 0);
sci::cca::GUIBuilder::pointer wxSCIRunApp::topBuilder(0);

// Don't automatically create main function (see wx/app.h).
// Initializes global application object.
IMPLEMENT_APP_NO_MAIN(wxSCIRunApp)

bool
wxSCIRunApp::OnInit()
{
  Guard g(&appLock);
  wxApp::OnInit(); // for command line processing (if any)

  // only show if this is the first Builder in this address space
  wxInitAllImageHandlers();

  std::string path(sci_getenv("SCIRUN_SRCDIR"));
  path += "/CCA/Components/Builder/scirun2-splash.png";
  wxBitmap bitmap(wxT(path), wxBITMAP_TYPE_PNG);
  if (bitmap.Ok()) {
    wxSplashScreen splash(bitmap, wxSPLASH_TIMEOUT|wxSPLASH_CENTRE_ON_SCREEN, SPLASH_TIMEOUT, 0, wxID_ANY);
    splash.Show(true);
  } else {
    std::cerr << "wxSCIRunApp::OnInit(): bitmap not loaded" << std::endl;
  }
  wxApp::Yield();

  GUIBuilder::setApp(*this);
  semUp();

  BuilderWindow *window = new BuilderWindow(topBuilder, 0);
  window->Show(true);
  SetTopWindow(window);

  return true;
}

void
wxSCIRunApp::AddTopWindow(const sci::cca::GUIBuilder::pointer& bc)
{
#if DEBUG
  std::cerr << "wxSCIRunApp::AddTopWindow(): from thread " << Thread::self()->getThreadName() << std::endl;
#endif

  // set the "main" top level window as parent
  appLock.lock();
  wxWindow *top = GetTopWindow();
  appLock.unlock();

  BuilderWindow *window = new BuilderWindow(bc, top);
  window->Show(true);
}

BuilderWindow*
wxSCIRunApp::GetTopBuilderWindow() const
{
  appLock.lock();
  wxWindow *top = GetTopWindow();
  appLock.unlock();

  BuilderWindow *bw = dynamic_cast<BuilderWindow*>(top);
  return bw;
}

}
