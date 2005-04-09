
/*
 * Dead-simple utility to dlopen a bunch of libraries and print
 * out any errors - like missing symbols.  Useful for figuring
 * out template instantiations under linux
 */

#include <dlfcn.h>
#include <stdio.h>

main(int argc, char* argv[])
{
    for(int i=1;i<argc;i++){
	void* handle=dlopen(argv[i], RTLD_NOW);
	if(!handle){
	    fprintf(stderr, "Error opening %s:\n%s\n", argv[i], dlerror());
	}
    }
}

	
