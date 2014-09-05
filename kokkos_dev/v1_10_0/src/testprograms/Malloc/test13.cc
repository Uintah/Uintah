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
