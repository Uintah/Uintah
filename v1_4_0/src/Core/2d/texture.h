/* texture.h */

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

#ifndef SCIRun_Core_2d_texture_h
#define SCIRun_Core_2d_texture_h 1

typedef struct {
  float wslop;
  float hslop;
  int p2width;
  int p2height;
  int id;
  unsigned char* pixels;
} Texture;

Texture* CreateTexture(int width, int height, int id, unsigned char* data);
void UseTexture(Texture* tex);

#endif
