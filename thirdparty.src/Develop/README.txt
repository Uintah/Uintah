Most Important Notes
--------------------

- Use the 'make_tarball.sh' script to create the Thirdparty tarball.


Full Info
---------

This directory contains files used in the development and packaging of
the Thirdparty software for distribution.  When the Thirdparty tarball
is created (via the make_tarball.sh script) this directory is not
exported. 

  NOTE: If you grabbed the Thirdparty from SVN, then you will have
  this directory, but most likely will not use it.

Items
-----

* Hacks - a directory with information on various hacks that are
          sometimes necessary to get the Thirdparty to build on
          various platforms.  eg: blt-patch and ImageMagick.txt.

* make_tarball.sh - Creates the 'release' tarball (which excludes this
          directory).

* README.txt - This file.
         
* TODO.txt - List of things that need to be done with the Thirdparty.


Generating Patch Files
----------------------

Use "diff -p orig new".  Save the diff into a patch file.  You may
need to edit the diff output to have the correct path to the orig
file.  The path should only have the library name down to the file.
It should NOT begin with a / or have Thirdparty src/ path in it.  Eg:  

*** tcl8.3.4/unix/configure	Fri Oct 19 18:24:15 2001

