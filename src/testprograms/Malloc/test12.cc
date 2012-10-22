/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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
#ifndef _WIN32
#include <strings.h>
#endif
#if defined(__sun)
#include <cstring>
#define bcopy(src,dest,n) memcpy(dest,src,n)
#elif defined(__linux) || defined(__digital__) || defined __sgi || defined __APPLE__ || defined _WIN32
#include <cstring>
#else
#error "Need bcopy idfdef for this architecture"
#endif

int
main(int,char **)
{
    for(int i=0;i<50;i++){
	int n=rand()%1000+1;
	void* p=malloc(10);
	for(int j=0;j<n;j++){
	    int s=rand()%(65536-8);
	    p=realloc(p, s);
	    memset(p, 0, s);
	}
	free(p);
    }
    return 0;
}
