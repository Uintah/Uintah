
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

#include <sci_defs.h>

#if defined(HAVE_GLEW)
#include <GL/glew.h>
#else
#include <GL/gl.h>
#endif

#include <Packages/Volume/Core/Geom/FragmentProgramARB.h>

#include <stdlib.h>
#include <iostream>
using namespace Volume;
using std::cerr;
using std::endl;

#if defined(GL_ARB_fragment_program)  && defined(__APPLE__)
FragmentProgramARB::FragmentProgramARB (const char* str, bool isFileName)
  : mId(0), mBuffer(0), mLength(0), mFile(0)
{
  init( str, isFileName);
}

void
FragmentProgramARB::init( const char* str, bool isFileName )
{
  if( isFileName ) {
    if(mFile) delete mFile;
    mFile = new char[strlen(str)];
    strcpy(mFile, str);
    
    FILE *fp;

    if (!(fp = fopen(str,"rb")))
    {
      cerr << "FragProgARB::constructor error: " << str << " could not be read " << endl;
      return;
    }
    
    fseek(fp, 0, SEEK_END);
    mLength = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    
    if(mBuffer) delete mBuffer;
    mBuffer = new unsigned char[mLength+1];
    
    fread( mBuffer, 1, mLength, fp);
    mBuffer[mLength] = '\0'; // make it a regular C string
    fclose(fp);
  } else {
    mLength = strlen(str);
    mBuffer = new unsigned char[mLength+2];
    strcpy((char*)mBuffer, str);
  }
  
}

FragmentProgramARB::~FragmentProgramARB ()
{
  delete [] mBuffer;
}

bool
FragmentProgramARB::created() {return bool(mId);}

void
FragmentProgramARB::create ()
{
  glGenProgramsARB(1, &mId);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
  glProgramStringARB(GL_FRAGMENT_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
		     mLength, mBuffer);

  if (glGetError() != GL_NO_ERROR)
  {
	cerr << "Fragment program error" << endl;
	return;
  }
}

void
FragmentProgramARB::destroy ()
{
  glDeleteProgramsARB(1, &mId);
  mId = 0;
}

void
FragmentProgramARB::bind ()
{
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
}

void
FragmentProgramARB::release ()
{
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, 0);
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::enable ()
{
  glEnable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::disable ()
{
  glDisable(GL_FRAGMENT_PROGRAM_ARB);
}

void
FragmentProgramARB::makeCurrent ()
{
  glBindProgramARB(GL_FRAGMENT_PROGRAM_ARB, mId);
}

#endif

