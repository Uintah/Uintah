/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

//----- RadiationVariables.h -----------------------------------------------

#ifndef Uintah_Components_Models_RadiationVariables_h
#define Uintah_Components_Models_RadiationVariables_h

#include <Core/Grid/Variables/CCVariable.h>

/**************************************
CLASS
   RadiationVariables
   
   Class RadiationVariables creates and stores the 
   Variables that are used in Components/Models/Radiation

GENERAL INFORMATION
   RadiationVariables.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date:   April 11, 2005
   
   C-SAFE 
   
   
KEYWORDS

DESCRIPTION
   This routine is based on ArchesVariables, written by Rajesh Rawat (2000) 
   in the Components/Arches directory.   

WARNING
   none

************************************************************************/

namespace Uintah {

  class RadiationVariables {
    public:
      RadiationVariables();
      ~RadiationVariables();

      CCVariable<double> ABSKG;
      CCVariable<int> cellType;
      CCVariable<double> cenint;
      CCVariable<double> co2;
      CCVariable<double> h2o;
      CCVariable<double> mixfrac;
      CCVariable<double> cp;
      CCVariable<double> ESRCG;
      CCVariable<double> qfluxe;
      CCVariable<double> qfluxw;
      CCVariable<double> qfluxn;
      CCVariable<double> qfluxs;
      CCVariable<double> qfluxt;
      CCVariable<double> qfluxb;
      CCVariable<double> sootVF;
      CCVariable<double> shgamma;
      CCVariable<double> src;
      CCVariable<double> temperature;

    }; // End class RadiationVariables
   
} // End namespace Uintah

#endif
