#!/usr/bin/env python

from os import stat, system, mkdir, path, getcwd, chdir
from sys import exit

#pass the ups directory and filename, and a list of changes
# change will be the entire tag change, i.e.,
# ["<patches> [2,2,2] </patches>",'<burn type = "null" />'] 
# would be a list of changes

def modUPS(inputsDir, filename, changes):
    
    org_filename = "%s/%s" % (inputsDir, filename)
    mod_filename  = "%s/tmp/%s" % (inputsDir, filename)

    #__________________________________
    #  bulletproofing
    # see if filename exists in directory and create tmp
    try:
      stat(inputsDir)
    except Exception:
      print "(%s) does not exist" % inputsDir
      exit(1)
  
    try:
      stat(org_filename)
    except Exception:
      print "(%s) does not exist" % org_filename
      exit(1)
    try:
      stat("%s/tmp" % inputsDir)
    except Exception:
      mkdir("%s/tmp" % inputsDir)
      
    #__________________________________
    # append numbers to the end of  filename.  You need to return a unique filename
    # if a base ups file is modified more than once from multiple modUPS calls.
    # go through loop until stat fails
    append = 1
    try:
      while 1:
        appendedFilename = "%s.%d" % (mod_filename,append)
	stat(appendedFilename)
	append = append + 1
    except Exception:
      mod_filename = "%s.%d" % (mod_filename, append)
      filename = "%s.%d" % (filename, append)
    
    
    #__________________________________
    # copy filename to tmp
    command = "cp %s %s" % (org_filename, mod_filename)
    system(command)
    
    #__________________________________

    for change in changes:
      addToScript = 1
      system("rm -f sedscript")
      sedreplacestring = ""
      sedscript = "s/"
      
      for ch in change:
        if ch == '=' or ch == '>' or ch == ' ':
          addToScript = 0
        if addToScript == 1:
          sedscript = sedscript + ch
        if ch == '/':
          sedreplacestring = sedreplacestring + "\/"
        else:
          sedreplacestring = sedreplacestring + ch
          
      sedscript = sedscript + ".*>/" + sedreplacestring + "/"
      
      command   = "echo \'%s\' > sedscript" % sedscript
      system(command)
      
      # be careful not all sed options (-i) are portable between OSs
      command = "sed -i.bak -f sedscript %s" % (mod_filename)
      system(command)
      
      command = "rm -f %s.bak" % (mod_filename)
      system(command)

    return "tmp/%s" % filename
    
    
    
    
#______________________________________________________________________
#  This script uses xmlstarlet to modify the ups file and is designed to 
#  change the same xml tag that is in multiple locations.  Example:
#
#
#   RMCRT_DO_perf_GPU_ups = modUPS2( the_dir, \
#                               "RMCRT_DO_perf.ups", \
#                               ["/Uintah_specification/Grid/Level/Box[@label=0]/resolution :[12,12,12]",
#                                "/Uintah_specification/Grid/Level/Box[@label=1]/resolution :[24,24,24]",
#                                "/Uintah_specification/Grid/Level/Box[@label=2]/resolution :[48,48,48]"
#                               ] )
#  This script depends on xmlstarlet.  
#  Use:
#       xmlstarlet el -v <ups>
#  to see the path to the xml node

def modUPS2( inputsDir, filename, changes):
    
    org_filename = "%s/%s" % (inputsDir, filename)
    mod_filename  = "%s/tmp/%s" % (inputsDir, filename)
    tmp_filename  = "%s/tmp/%s.tmp" % (inputsDir, filename)

    #__________________________________
    #  bulletproofing
    # see if filename exists in directory and create tmp
    rc = system("which xmlstarlet > /dev/null 2>&1")
    
    if rc == 256:
      print( "__________________________________")
      print( "ERROR:modUPS.py " )
      print( "      The command (xmlstarlet) was not found and the file %s was not modified" %filename )
      print( "__________________________________")
      return
    
    try:
      stat(inputsDir)
    except Exception:
      print "(%s) does not exist" % inputsDir
      exit(1)
  
    try:
      stat(org_filename)
    except Exception:
      print "(%s) does not exist" % org_filename
      exit(1)
      
    try:
      stat("%s/tmp" % inputsDir)
    except Exception:
      mkdir("%s/tmp" % inputsDir)

    #__________________________________
    # copy filename to tmp/
    command = "cp %s %s" % (org_filename, mod_filename)
    system(command)
    
    #__________________________________
    #  Apply changes to ups file
    for change in changes:
    
      option = change.split(":")
     
      command = "xmlstarlet ed -u %s -v %s  %s > %s" % (option[0], option[1], mod_filename, tmp_filename )
      system(command)
      
      command = "mv %s %s" % (tmp_filename, mod_filename)
      system(command)

    return "tmp/%s" % filename    
