The sub.mk file in the folder has been modified to generate a shared library instead of a executable, after configuring and building, the library gets placed in the lib directory under the main build folder.

The library has the following name,
libPackages_Uintah_StandAlone_tools_uda2vis

As of now the name and location of this library has been hardcoded into the file avtudaReaderMTMDFileFormat.C and needs to be changed by the user.

Also, the location of the header file 'particleData.h' (in the folder uda2vis) has been hardcoded into avtudaReaderMTMDFileFormat.h and needs to be changed by the user accordingly. 

The above issues and other's would be fixed in subsequent releases. 
