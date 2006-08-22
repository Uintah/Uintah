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
 * FrameworkProxyDialog.h
 *
 * Written by:
 *  Ayla Khan
 *  Scientific Computing and Imaging Institute
 *  University of Utah
 *  September 2004
 *
 *  Copyright (C) 2004 SCI Institute
 *
 */

#ifndef CCA_Components_GUIBuilder_FrameworkProxyDialog_h
#define CCA_Components_GUIBuilder_FrameworkProxyDialog_h

#include <wx/dialog.h>

class wxStaticText;
class wxComboBox;
class wxTextCtrl;
class wxSpinCtrl;
class wxButton;
class wxRadioBox;

// TODO: save some (NOT password) user entries to file!
// Note: passwords stored in memory

namespace GUIBuilder {

class FrameworkProxyDialog : public wxDialog {
public:
  enum {
    ID_BUTTON_RESET = wxID_HIGHEST,
  };

  FrameworkProxyDialog(wxWindow *parent,
                       wxWindowID id = wxID_ANY,
                       const wxString &title = wxT("Add Framework Proxy"),
                       const wxPoint& pos = wxDefaultPosition,
                       const wxSize& size = wxDefaultSize,
                       long style = wxCAPTION|wxSYSTEM_MENU,
                       const wxString& name = wxT("Add Framework Proxy"),
                       const std::string& defaultProxy = "localhost",
                       const std::string& defaultDomain = "localhost",
                       const std::string& defaultLogin = "localuser");
  virtual ~FrameworkProxyDialog();

  void OnReset(wxCommandEvent& event);

  wxString GetLoader() const;
  wxString GetDomain() const;
  wxString GetLogin() const;
  wxString GetPath() const;
//   wxString GetPassword() const;
  int GetCopiesNumber() const;
  wxString GetMPIWhereString() const;

protected:
  void SetLayout();
  void SetDefaultText();

private:
  wxComboBox* proxyComboBox;
  wxComboBox* domainComboBox;
  wxComboBox* loginComboBox;
  wxComboBox* pathComboBox;
//   wxTextCtrl* passwordTextCtrl;

  wxStaticText* copiesStaticText;
  wxSpinCtrl* copiesSpinCtrl;
  wxRadioBox* mpiWhereRadioBox;
  wxTextCtrl* whereTextCtrl;
  wxButton* helpButton;
  wxButton* resetButton;

  const wxString proxy;
  const wxString domain;
  const wxString login;
  const wxString path;

  static const int TEXT_CTRL_WIDTH = 200;

  DECLARE_EVENT_TABLE();
};

}

#endif
