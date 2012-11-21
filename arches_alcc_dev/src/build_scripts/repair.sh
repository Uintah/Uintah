#! /bin/sh

#
# repair.sh [-quiet] <bad_include>
#
# The purpose of this script is to find .d (Makefile dependency files)
# that have bad (old) include files in them and to delete the .d and
# corresponding .o file.  This is useful/necessary when you see an error
# like:
#
#    No rule to make target `../src/Dataflow/Modules/Render/SCIBaWGL.h'...
#
# In this case, you would run (from your build directory):
#
#    ../src/build_scripts/repair.sh SCIBaWGL.h
#

usage()
{
  echo ""
  echo "Usage: $0 [-quiet] <bad_include>"
  echo ""
  echo "  -h[elp]     - Print this help."
  echo "  -quiet      - Don't print out a warning if no files found."
  echo "  bad_include - The name of the bad file (or some unique text"
  echo "                that is grep'd for in the .d files.)"
  echo ""
  echo "This script will run through all the .d (Makefile Dependency)"
  echo "files in the current directory and below.  If any of those files"
  echo "have the 'bad_include' in them, it will delete the .d and"
  echo "corresponding .o file."
  echo ""
  echo "If, for example, you updated to the newest thirdparty (version 1.22)"
  echo "and are now getting:"
  echo ""
  echo "  gmake: No rule to make target ???, needed by ___.o"
  echo ""
  echo "You could run this script like this:"
  echo ""
  echo "cd .../SCIRun/<bin>"
  echo "../src/build_scripts/repair.sh Thirdparty/1.20"
  echo ""
  echo "And then run your 'gmake' again."
  echo ""
  echo "Example 2:"
  echo ""
  echo "If the error is something like:"
  echo ""
  echo "    No rule to make target `../src/Dataflow/Modules/Render/SCIBaWGL.h',"
  echo "       needed by `Dataflow/Modules/Render/OpenGL.o'.  Stop."
  echo ""
  echo "Then you would use:"
  echo ""
  echo "../src/build_scripts/repair.sh SCIBaWGL.h"
  echo ""
  exit
}

if test $# = 0; then
    echo ""
    echo "Bad number of arguments..."
    usage
fi

if test $1 = "-h" || test $1 = "-help" || test $1 = "--help"; then
    usage
fi

if test $1 = "-quiet"; then
    be_quiet=true
    shift
fi

if test $# = 0; then
    echo ""
    echo "Bad number of arguments..."
    usage
fi

bad_inc=$1

extension=d
if test `uname` == "AIX"; then
  extension=u
fi

files=`find . -name "*".$extension -o -name "depend.mk" | xargs grep -l $bad_inc`

file_found=no

for file in $files; do
   file_found=yes

   filename=`echo $file | sed "s%.*/%%"`

   if test "$filename" = "depend.mk"; then
     c_files=`grep $bad_inc $file | cut -f1 -d":"`
   else
     c_files=$file
   fi

   for cfile in $c_files; do
     base=`echo $cfile | sed "s%\.$extension%%" | sed "s%\.o%%"`
     echo "rm -rf $base.o"
           rm -rf $base.o
     if test "$filename" != "depend.mk"; then
        echo "rm -rf $base.$extension"
              rm -rf $base.$extension
     else
        # remove the individual bad line from the __depend.mk__ file.
        rm -rf $file.temp
        grep -v $bad_inc $file > $file.temp
        mv $file.temp $file
     fi 
   done
done

if test "$be_quiet" != "true" -a $file_found = "no"; then
   echo ""
   echo "No matching files found. (Perhaps your 'bad_include' was incorrect?)"
   echo "If you continue to have problems with this script, please contact"
   echo "J. Davison de St. Germain (dav@sci.utah.edu)."
   echo ""
fi



