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
 *  TrigTable.h: Faster ways to do trig...
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   May 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#ifndef SCI_Math_TrigTable_h
#define SCI_Math_TrigTable_h 1

#include <Core/share/share.h>

class SCICORESHARE SinCosTable {
    double* sindata;
    double* cosdata;
    int n;
public:
    SinCosTable(int n, double min, double max, double scale=1.0);
    ~SinCosTable();
    double sin(int) const;
    double cos(int) const;
};

#endif
