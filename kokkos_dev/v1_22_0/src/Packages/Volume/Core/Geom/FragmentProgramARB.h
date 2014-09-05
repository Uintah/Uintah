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

#ifndef FRAGMENTPROGRAM_ARB
#define FRAGMENTPROGRAM_ARB


namespace Volume {

#if defined(GL_ARB_fragment_program) && defined(__APPLE__)
class FragmentProgramARB
{
public:
  FragmentProgramARB (const char* str, bool isFileName = false);
  ~FragmentProgramARB ();
  void init( const char* str, bool isFileName );

  
  bool created();
  void create ();
//  void update ();
  void destroy ();

  void bind ();
  void release ();
  void enable ();
  void disable ();
  void makeCurrent ();
  
protected:
  unsigned int mId;
  unsigned char* mBuffer;
  unsigned int mLength;
  char* mFile;
};
#endif

}// end namespace Volume
#endif
