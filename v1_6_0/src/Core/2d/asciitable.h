/* asciitable.h */

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

#ifndef SCIRun_Core_2d_asciitable_h
#define SCIRun_Core_2d_asciitable_h 1

#define TABLE_PIXEL_WIDTH 510
#define TABLE_PIXEL_HEIGHT 510
#define TABLE_CHAR_WIDTH 10
#define TABLE_CHAR_HEIGHT 10

#define CHAR_PIXEL_WIDTH 51 /*TABLE_PIXEL_WIDTH/TABLE_CHAR_WIDTH*/
#define CHAR_PIXEL_HEIGHT 51 /*TABLE_PIXEL_HEIGHT/TABLE_CHAR_HEIGHT*/

#ifdef __cplusplus
#define EXTERN extern "C"
#else
#define EXTERN extern
#endif

EXTERN unsigned char font[];

#endif
