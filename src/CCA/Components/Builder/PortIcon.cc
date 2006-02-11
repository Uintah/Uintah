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
 * PortIcon.cc
 *
 */

#include <CCA/Components/Builder/PortIcon.h>

#include <wx/region.h>

namespace SCIRun {

BEGIN_EVENT_TABLE(PortIcon, wxRegion)
// from htmlwin.cpp 
//EVT_LEFT_DOWN(wxHtmlWindow::OnMouseDown)
//EVT_LEFT_UP(wxHtmlWindow::OnMouseUp)
//EVT_RIGHT_UP(wxHtmlWindow::OnMouseUp)
//EVT_MOTION(wxHtmlWindow::OnMouseMove)
//EVT_PAINT(wxHtmlWindow::OnPaint)
//#if wxUSE_CLIPBOARD
//EVT_LEFT_DCLICK(wxHtmlWindow::OnDoubleClick)
//EVT_ENTER_WINDOW(wxHtmlWindow::OnMouseEnter)
//EVT_LEAVE_WINDOW(wxHtmlWindow::OnMouseLeave)
//#endif // wxUSE_CLIPBOARD
END_EVENT_TABLE()



// TODO: filter menu for type, find a better way to show port type
PortIcon::PortIcon()
{
}

PortIcon::~PortIcon()
{
}

void PortIcon::drawPort(QPainter &p)
{
//p.setPen(pColor);
//p.setBrush(pColor);
//p.drawRect(pRect);
//p.setPen(iColor);
//p.setBrush(iColor);
//p.drawRect(iRect);
}

QPoint PortIcon::portPoint()
{
}

// better way to figure out how to map color to port type?
//void PortIcon::portColorMap()
//{
//    colorMap = sci::cca::TypeMap::pointer(new TypeMap);
//
//    // using named colors
//    // Qt can use these for either Unix or Windows systems
//    // see Qt documentation for QColor
//    colorMap->putString(std::string("default"),
//                        std::string("yellow"));
//    colorMap->putString(std::string("highlight"),
//                        std::string("white"));
//    colorMap->putString(std::string("StringPort"),
//                        std::string("cadetblue1"));
//    colorMap->putString(std::string("ZListPort"),
//                        std::string("goldenrod4"));
//    colorMap->putString(std::string("LinSolverPort"),
//                        std::string("darkorange"));
//    colorMap->putString(std::string("PDEdescriptionPort"),
//                        std::string("tomato"));
//    colorMap->putString(std::string("MeshPort"),
//                        std::string("magenta"));
//    colorMap->putString(std::string("ViewPort"),
//                        std::string("darkseagreen"));
//    colorMap->putString(std::string("FEMmatrixPort"),
//                        std::string("gray65"));
//}

}
