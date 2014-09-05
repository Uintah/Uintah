/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



#include <stdlib.h>
#include <stdio.h>
#include <strings.h>
#if defined(__sun)
#include <string.h>
#define bcopy(src,dest,n) memcpy(dest,src,n)
#elif defined(__linux) || defined(__digital__) || defined __sgi || defined __APPLE__
#include <string.h>
#else
#error "Need bcopy idfdef for this architecture"
#endif

int
main(char **, int )
{
    unsigned long int tot=0;
    unsigned long int bytes=0;
    for(int i=0;i<100;i++){
	int n=rand()%40+1;
	void* p[40];
	int j;
	for(j=0;j<n;j++){
	    int s=(rand()|(rand()<<15))%(2*1024*1024);
	    p[j]=malloc(s);
	    bytes+=s;
	    bzero(p[j], s);
	}
	for(j=0;j<n;j++){
	    free(p[j]);
	}
	tot+=n;
    }
    fprintf(stderr, "%lu allocations (%lu bytes)\n", tot, bytes);
    return 0;
}
