/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//----- RadiationConstVariables.h -----------------------------------------------

#ifndef Uintah_Components_Models_RadiationConstVariables_h
#define Uintah_Components_Models_RadiationConstVariables_h

#include <Core/Grid/Variables/CCVariable.h>

/**************************************
CLASS
   RadiationConstVariables
   
   Class RadiationConstVariables creates and stores the ConstVariables 
   that are used in Components/Models/Radiation 

GENERAL INFORMATION
   RadiationConstVariables.h - declaration of the class
   
   Author: Seshadri Kumar (skumar@crsim.utah.edu)
   
   Creation Date: April 12, 2005

   C-SAFE 
   
   
KEYWORDS

DESCRIPTION
   This routine is based on ArchesConstVariables, written by Rajesh Rawat 
   (2000) in the Components/Arches directory.      

WARNING
   none

************************************************************************/

namespace Uintah {

  class RadiationConstVariables {
    public:
      RadiationConstVariables();
      ~RadiationConstVariables();
      constCCVariable<double> ABSKG;
      constCCVariable<double> ESRCG;
      constCCVariable<double> shgamma;
      constCCVariable<double> co2;
      constCCVariable<double> oldCO2;
      constCCVariable<int> cellType;
      constCCVariable<double> density;
      constCCVariable<double> iceDensity;
      constCCVariable<double> h2o;
      constCCVariable<double> oldH2O;
      constCCVariable<double> mixfrac;
      constCCVariable<double> sootVF;
      constCCVariable<double> temperature;
      constCCVariable<double> iceTemp;

    }; // End class RadiationConstVariables
} // End namespace Uintah

#endif
