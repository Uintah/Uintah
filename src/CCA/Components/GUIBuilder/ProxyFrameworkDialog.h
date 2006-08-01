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

#include <sci_wx.h>

class wxStaticText;
class wxComboBox;
class wxTextCtrl;
class wxSpinCtrl;
class wxButton;
class wxRadioBox;

namespace GUIBuilder {

class FrameworkProxyDialog : public wxDialog {
public:
  FrameworkProxyDialog(const std::string& defaultLoader="localhost", const std::string& defaultDomain="localhost", const std::string& defaultLogin);
  virtual ~FrameworkProxyDialog();

protected:
  void SetLayout();

private:
  wxComboBox* loaderComboBox;
  wxComboBox* domainComboBox;
  wxComboBox* loginComboBox;
  wxComboBox* pathComboBox;
  wxTextCtrl* passwdTextCtrl;

  wxSpinCtrl* copiesSpinCtrl;
  //wxRadioButton* radioButtonCPU;
  //wxRadioButton* radioButtonNode;
  wxRadioBox* mpiWhereRadioBox;
  wxTextCtrl* whereTextCtrl;
  wxButton* helpButton;
  wxButton* resetButton;
};

}

#endif
