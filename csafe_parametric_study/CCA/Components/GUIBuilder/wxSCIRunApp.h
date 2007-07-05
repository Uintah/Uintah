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

#ifndef CCA_Components_GUIBuilder_wxSCIRunApp_h
#define CCA_Components_GUIBuilder_wxSCIRunApp_h

#include <wx/app.h>
#include <wx/wxprec.h>
#ifndef WX_PRECOMP
 #include <wx/wx.h>
#endif

#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Thread.h>
#include <Core/CCA/spec/cca_sidl.h>

namespace GUIBuilder {

class BuilderWindow;

// wxApp is a singleton class, has private copy ctor, assgn. op. (see wx/app.h, wx/def.h)
class wxSCIRunApp : public wxApp {
public:
  virtual bool OnInit();
  // virtual int OnExit() { return wxApp::OnExit(); }
  BuilderWindow* GetTopBuilderWindow() const;

  static void SetTopBuilder(const sci::cca::GUIBuilder::pointer& bc);

  static void semDown() { sem.down(); }
  static void semUp() { sem.up(); }

private:
  static SCIRun::Mutex appLock;
  static SCIRun::Semaphore sem;

  // keep track of inital Builder component (instantiated from main)
  static sci::cca::GUIBuilder::pointer topBuilder;
};

DECLARE_APP(wxSCIRunApp)

}

#endif
