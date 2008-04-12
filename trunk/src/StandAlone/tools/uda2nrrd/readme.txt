The sub.mk file in the folder has been modified to generate a shared library instead of a executable, after building configuring and building, the library gets placed in the lib directory under the main build folder.

The library has the following name,
libPackages_Uintah_StandAlone_tools_uda2nrrd

As of now the name and location of this library has been hardcoded into the file avtudaReaderFileFormat.C (in the folder UdaReader) and needs to be changed by the user.

Also, the location of the header file 'particleData.h' (in the folder uda2nrrd) has been hardcoded into avtudaReaderFileFormat.h (in the folder UdaReader) and needs to be changed by the user accordingly. 

The above issues and other's would be fixed in subsequent releases. 