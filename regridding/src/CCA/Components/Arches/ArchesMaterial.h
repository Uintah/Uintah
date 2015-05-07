/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef __ARCHES_MATERIAL_H__
#define __ARCHES_MATERIAL_H__

#include <Core/Grid/Material.h>


namespace Uintah {

      
/**************************************
     
CLASS
   ArchesMaterial

   Short description...

GENERAL INFORMATION

   ArchesMaterial.h

   Rajesh Rawat
   Department of Chemical and Fuels Engineering
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

   
KEYWORDS
   MPM_Material

DESCRIPTION
   Long description...

WARNING

****************************************/
  class Burn;
  class ArchesMaterial : public Material {
  public:
    ArchesMaterial();
    
    ~ArchesMaterial();
    
    Burn* getBurnModel() {
      return 0;
    }
  private:

    // Prevent copying of this class
    // copy constructor
    ArchesMaterial(const ArchesMaterial &archesmm);
    ArchesMaterial& operator=(const ArchesMaterial &archesmm);
  };

} // End namespace Uintah

#endif // __ARCHES_MATERIAL_H__
