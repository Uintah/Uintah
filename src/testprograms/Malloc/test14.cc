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


#include <stdlib.h>
#include <iostream>
using std::cerr;
#include <Core/Malloc/Allocator.h>

struct X {
    int crap;
    X();
    ~X();
};

X::X()
{
    static int count;
    crap=count++;
}

X::~X()
{
    cerr << "~X() : crap=" << crap << "\n";
}


void* operator new(size_t, SCIRun::Allocator*, const char*);
#ifdef _BOOL
void* operator new[](size_t, SCIRun::Allocator*, const char*);
#endif

int
main(char **, int )
{
#if 0
    char* p=new char[100];
    delete[] p;
    X* pp=new X;
    delete pp;
    pp=new X[5];
    delete[] pp;
    pp=new (default_allocator, "1") X;
    delete pp;
    pp=new (default_allocator, "2") X[5];
    delete[] pp;
#endif
    return 0;
}
