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
 *  GuiGeom.h: Interface to TCL variables for Geom stuff
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Geom_GuiGeom_h
#define SCI_Geom_GuiGeom_h 1

#include <Core/GuiInterface/GuiVar.h>

namespace SCIRun {
  class Color;

  class GuiColor : public GuiVar {
    GuiDouble r;
    GuiDouble g;
    GuiDouble b;
  public:
    GuiColor(GuiContext* ctx);
    ~GuiColor();

    Color get();
    void set(const Color&);
  };

  class Material;
  class GuiMaterial : public GuiVar {
    GuiColor ambient;
    GuiColor diffuse;
    GuiColor specular;
    GuiDouble shininess;
    GuiColor emission;
    GuiDouble reflectivity;
    GuiDouble transparency;
    GuiDouble refraction_index;
  public:
    GuiMaterial(GuiContext* ctx);
    ~GuiMaterial();
   
    Material get();
    void set(const Material&);
  };
} // End namespace SCIRun


#endif // ifndef SCI_Geom_GuiGeom_h
