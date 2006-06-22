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
 * PortIcon.h
 *
 */

#ifndef CCA_Components_GUIBuilder_PortIcon_h
#define CCA_Components_GUIBuilder_PortIcon_h

#include <Core/CCA/spec/cca_sidl.h>
#include <CCA/Components/GUIBuilder/GUIBuilder.h>

class wxWindow;

namespace GUIBuilder {

class NetworkCanvas;
class ComponentIcon;

class PortIcon : public wxWindow {
public:
  PortIcon(const sci::cca::GUIBuilder::pointer& bc, ComponentIcon *parent,
           wxWindowID id, GUIBuilder::PortType pt, const std::string& name);
  virtual ~PortIcon();
  bool Create(wxWindow *parent, wxWindowID id, const wxString &name);

  void OnPaint(wxPaintEvent& event);
  void OnLeftDown(wxMouseEvent& event);
  void OnLeftUp(wxMouseEvent& event);
  void OnMouseMove(wxMouseEvent& event);
  void OnRightClick(wxMouseEvent& event);

  wxColor GetPortColor() { return pColor; }
  const std::string GetPortName() const { return name; }
  GUIBuilder::PortType GetPortType() const { return portType; }

  ComponentIcon* GetParent() const { return parent; }

  const static int PORT_WIDTH = 7;
  const static int PORT_HEIGHT = 10;
  const static int HIGHLIGHT_WIDTH = 3;

protected:
  PortIcon();
  void Init();

  sci::cca::GUIBuilder::pointer builder;
  ComponentIcon* parent;
  GUIBuilder::PortType portType;
  std::string name;
  std::string model;
  std::string type;

  wxRect hRect;
  wxColor hColor;
  wxColor pColor;

private:
  PortIcon(const PortIcon&);
  PortIcon& operator=(const PortIcon&);

  const int ID_MENU_POPUP;

  DECLARE_DYNAMIC_CLASS(PortIcon)
  DECLARE_EVENT_TABLE()
};

}

#endif
