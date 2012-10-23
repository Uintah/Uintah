/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <cstdlib>
#include <cstdio>
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


#if 0

void* operator new(size_t, SCIRun::Allocator*, const char*);
#ifdef _BOOL
void* operator new[](size_t, SCIRun::Allocator*, const char*);
#endif

#endif

int
main(int,char **)
{
  printf( "Warning... test14 has been disabled.\n" );

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
