
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

#include <sci_gl.h>

#include <Packages/Volume/Core/Geom/VertexProgramARB.h>

#include <stdlib.h>
#include <iostream>
#include <string>
using namespace Volume;
using std::cerr;
using std::endl;
using std::string;

VertexProgramARB::VertexProgramARB (const char* str, bool isFileName)
  : mId(0), mBuffer(0), mLength(0), mFile(0)
{
  init( str, isFileName);
}

void
VertexProgramARB::init( const char* str, bool isFileName )
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

VertexProgramARB::~VertexProgramARB ()
{
  delete [] mBuffer;
}

bool
VertexProgramARB::valid()
{
  return glIsProgramARB(mId);
}

void
VertexProgramARB::create ()
{
  glGenProgramsARB(1, &mId);
  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, mId);
  glProgramStringARB(GL_VERTEX_PROGRAM_ARB, GL_PROGRAM_FORMAT_ASCII_ARB,
		     mLength, mBuffer);
  if (glGetError() != GL_NO_ERROR)
  {
    string mProgram((const char*)mBuffer);
    int position;
    glGetIntegerv(GL_PROGRAM_ERROR_POSITION_ARB, &position);
    int start = position;
    for (; start > 0 && mProgram[start] != '\n'; start--);
    if (mProgram[start] == '\n') start++;
    uint end = position;
    for (; end < mProgram.length()-1 && mProgram[end] != '\n'; end++);
    if (mProgram[end] == '\n') end--;
    int ss = start;
    int l = 1;
    for (; ss >= 0; ss--) { if (mProgram[ss] == '\n') l++; }
    string line((char*)(mProgram.c_str()+start), end-start+1);
    string underline = line;
    for (uint i=0; i<end-start+1; i++) underline[i] = '-';
    underline[position-start] = '#';
    glBindProgramARB(GL_VERTEX_PROGRAM_ARB, 0);
    cerr << "Vertex program error at line " << l << ", character "
         << position-start << endl << endl << line << endl
         << underline << endl;
  }
}

void
VertexProgramARB::destroy ()
{
  glDeleteProgramsARB(1, &mId);
  mId = 0;
}

void
VertexProgramARB::bind ()
{
  glEnable(GL_VERTEX_PROGRAM_ARB);
  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, mId);
}

void
VertexProgramARB::release ()
{
  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, 0);
  glDisable(GL_VERTEX_PROGRAM_ARB);
}

void
VertexProgramARB::enable ()
{
  glEnable(GL_VERTEX_PROGRAM_ARB);
}

void
VertexProgramARB::disable ()
{
  glDisable(GL_VERTEX_PROGRAM_ARB);
}

void
VertexProgramARB::makeCurrent ()
{
  glBindProgramARB(GL_VERTEX_PROGRAM_ARB, mId);
}

void
VertexProgramARB::setLocalParam(int i, float x, float y, float z, float w)
{
  glProgramLocalParameter4fARB(GL_VERTEX_PROGRAM_ARB, i, x, y, z, w);
}
