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

#ifndef VERTEXPROGRAM_ARB
#define VERTEXPROGRAM_ARB

namespace Volume {

class VertexProgramARB
{
public:
  VertexProgramARB (const char* str, bool isFileName = false);
  ~VertexProgramARB ();
  void init( const char* str, bool isFileName );
  
  void create ();
  bool valid ();
//  void update ();
  void destroy ();

  void bind ();
  void release ();
  void enable ();
  void disable ();
  void makeCurrent ();

  void setLocalParam(int, float, float, float, float);
  
protected:
  unsigned int mId;
  unsigned char* mBuffer;
  unsigned int mLength;
  char* mFile;
};

}// end namespace Volume

#endif // VERTEXPROGRAM_ARB
