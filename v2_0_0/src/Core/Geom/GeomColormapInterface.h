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
 *  GeomColormapInterface.h - interface to colormap class
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   February 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#ifndef SCI_Geom_GeomColormapInterface_h
#define SCI_Geom_GeomColormapInterface_h 1

#include <Core/Geom/Material.h>

namespace SCIRun {

class SCICORESHARE GeomColormapInterface {
public:
    virtual MaterialHandle& lookup2(double value)=0;
    virtual double getMin() const = 0;
    virtual double getMax() const = 0;
};

} // End namespace SCIRun



#endif
