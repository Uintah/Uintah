/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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

  class SCICORESHARE GuiColor : public GuiVar {
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
  class SCICORESHARE GuiMaterial : public GuiVar {
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
