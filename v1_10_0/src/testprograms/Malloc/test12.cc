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
    for(int i=0;i<50;i++){
	int n=rand()%1000+1;
	void* p=malloc(10);
	for(int j=0;j<n;j++){
	    int s=rand()%(65536-8);
	    p=realloc(p, s);
	    bzero(p, s);
	}
	free(p);
    }
    return 0;
}
