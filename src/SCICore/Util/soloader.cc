// soloader.cpp written by Chris Moulding 11/98

#include <iostream>
using std::cerr;
using std::endl;
#include <SCICore/Util/soloader.h>

LIBRARY_HANDLE explicitlyopenedhandles[3000];
int numberofopenedhandles = 0;

void* GetLibrarySymbolAddress(const char* libname, const char* symbolname)
{
	void* SymbolAddress = 0;
	LIBRARY_HANDLE LibraryHandle = 0;

#ifdef _WIN32
	LibraryHandle = LoadLibrary(libname);
#else
	LibraryHandle = dlopen(libname, RTLD_LAZY);
#endif

	if (LibraryHandle == 0)
	{
	    //cerr << "ERROR: The library \"" << libname << "\" could not be found, or is corrupt." << endl;
		return 0;
	}

#ifdef _WIN32
	SymbolAddress = GetProcAddress(LibraryHandle,symbolname);
#else
	SymbolAddress = dlsym(LibraryHandle,symbolname);
#endif

	if (SymbolAddress == 0)
	{
	    //cerr << "ERROR: The symbol \"" << symbolname << "\" could not be found in the library \"" << libname << "\"." << endl;
		return 0;
	}

	explicitlyopenedhandles[numberofopenedhandles++]=LibraryHandle;

	return SymbolAddress;
}

void* GetHandleSymbolAddress(LIBRARY_HANDLE handle, const char* symbolname)
{
    void* SymbolAddress = 0;
     
#ifdef _WIN32
	SymbolAddress = GetProcAddress(handle,symbolname);
#else
	SymbolAddress = dlsym(handle,symbolname);
#endif

	if (SymbolAddress == 0)
	{
	    //cerr << "ERROR: The symbol \"" << symbolname << "\" could not be found using the handle \"" << handle << "\"." << endl;
		return 0;
	}

	explicitlyopenedhandles[numberofopenedhandles++]=handle;

	return SymbolAddress;
}

LIBRARY_HANDLE GetLibraryHandle(const char* libname)
{
	LIBRARY_HANDLE LibraryHandle = 0;

#ifdef _WIN32
	LibraryHandle = LoadLibrary(libname);
#else
	LibraryHandle = dlopen(libname, RTLD_LAZY);
#endif

	if (LibraryHandle == 0)
	{
	    //cerr << "ERROR: The library \"" << libname << "\" could not be found, or is corrupt." << endl;
		return 0;
	}

   	explicitlyopenedhandles[numberofopenedhandles++]=LibraryHandle;

	return LibraryHandle;
}

void CloseLibraries()
{
	for (int i=0; i<numberofopenedhandles; i++)
#ifdef _WIN32
		FreeLibrary(explicitlyopenedhandles[i]);
#else
		dlclose(explicitlyopenedhandles[i]);
#endif
	
}









