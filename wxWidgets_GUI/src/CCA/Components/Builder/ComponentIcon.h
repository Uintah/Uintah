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


#ifndef ComponentIcon_h
#define ComponentIcon_h

#include <Core/CCA/spec/cca_sidl.h>

#include <map>
//#include <vector>

//class wxRegion;

class wxWindow;
class wxPanel;
class wxGridBagSizer;
class wxButton;
class wxGauge;

namespace GUIBuilder {

class NetworkCanvas;
class PortIcon;

// to replace button?
// class MessageControl : public wxWindow {
// public:
//   MessageControl();
//   ~MessageControl();

// private:
// };

class ComponentIcon : public wxPanel {
public:
  typedef std::map<std::string, PortIcon*> PortMap;

  enum {
    ID_MENU_GO = wxID_HIGHEST,
    ID_MENU_INFO,
  };

  // deal with wxValidator?
  ComponentIcon(const sci::cca::BuilderComponent::pointer& bc, wxWindowID winid, NetworkCanvas* parent, const sci::cca::ComponentID::pointer& compID, int x, int y);
  virtual ~ComponentIcon();

  // draw own border?
  bool Create(wxWindow* parent, wxWindowID winid, const wxPoint& pos = wxDefaultPosition, const wxSize& size = wxDefaultSize, long style = wxNO_BORDER|wxCLIP_CHILDREN /*wxDOUBLE_BORDER|wxRAISED_BORDER|wxCLIP_CHILDREN*/);

  void OnPaint(wxPaintEvent& event);
  // Empty implementation, to prevent flicker - from wxWidgets book
  //void OnEraseBackground(wxEraseEvent& event) {}
  void OnLeftDown(wxMouseEvent& event);
  void OnLeftUp(wxMouseEvent& event);
  void OnMouseMove(wxMouseEvent& event);
  void OnRightClick(wxMouseEvent& event);
  void OnGo(wxCommandEvent& event);

  const wxSize& GetBorderSize() const { return borderSize; }
  //void DrawPorts(wxDC& dc);

  const sci::cca::ComponentID::pointer GetComponentInstance() const { return cid; }
  const std::string GetComponentInstanceName() { return cid->getInstanceName(); }
  PortIcon* GetPortIcon(const std::string& portName) { return ports[portName]; }

  void GetCanvasPosition(wxPoint& p);
  NetworkCanvas* GetCanvas() const { return canvas; }

  // possible to set shape w/ bitmap region?

protected:
  ComponentIcon();
  void Init();
  //void setLayout();

  NetworkCanvas *canvas;

  wxGridBagSizer* gridBagSizer;
  wxStaticText* label;
  wxStaticText* timeLabel;
  wxButton* uiButton;
  wxButton* msgButton;
  wxGauge* progressGauge;
  wxSize borderSize;

  wxMenu *popupMenu;
  wxMenu* menu;

  bool hasUIPort;
  bool hasGoPort;
  bool isSciPort;

  bool isMoving;

  sci::cca::ComponentID::pointer cid;
  sci::cca::BuilderComponent::pointer builder;
  std::string goPortName;

  PortMap ports;
  wxPoint movingStart;

private:
  ComponentIcon(const ComponentIcon&);
  ComponentIcon& operator=(const ComponentIcon&);

  const int ID_MENU_POPUP;
  const int ID_BUTTON_UI;
  const int ID_BUTTON_STATUS;
  const int ID_PROGRESS;

  DECLARE_EVENT_TABLE()
  DECLARE_DYNAMIC_CLASS(ComponentIcon)
  //DECLARE_NO_COPY_CLASS(ComponentIcon)
};

}

#endif
