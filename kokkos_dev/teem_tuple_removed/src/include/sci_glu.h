/*  The contents of this file are subject to the University of Utah Public
 *  License (the "License"); you may not use this file except in compliance
 *  with the License.
 *  
 *  Software distributed under the License is distributed on an "AS IS"
 *  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
 *  License for the specific language governing rights and limitations under
 *  the License.
 *  
 *  The Original Source Code is SCIRun, released March 12, 2001.
 *  
 *  The Original Source Code was developed by the University of Utah.
 *  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
 *  University of Utah. All Rights Reserved.
 *  
 *    File   : sci_glu.h
 *    Author : Martin Cole
 *    Date   : Thu Apr  3 10:29:23 2003
 *
 *    This is required if you wish to use the gl.h provided with the nvidia
 *    drivers.
 */

#if !defined(SCI_GLU_H)
#define SCI_GLU_H

#include <GL/gl.h>

#ifndef GLAPIENTRY
  #define GLAPIENTRY
#endif

#ifndef GLAPI
  #define GLAPI
#endif

#include <GL/glu.h>

#endif  /* #define SCI_GLU_H */
