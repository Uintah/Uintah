To build the files, you need to generate the makefile. This can be accomplished by pointing the xml2make utility in the VisIt's bin directory to the xml file in the current directory,

After downloading the directory from svn, type the following at the command prompt,

xml2makefile -public -clobber udaReaderMTMD.xml

This would generate a makefile in the corresponding directory and calling 'make' thereafter would generate the plugin(s) and would place them appropriately into VisIt's plugin directory.

To start reading uda files, place a test.uda file inside the uda archive and point to that file when running VisIt.
