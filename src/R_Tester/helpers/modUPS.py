#!/usr/bin/env python

from os import stat, system, mkdir, path, getcwd, chdir
from sys import exit

#______________________________________________________________________
def bulletProofInputs(inputsDir, org_filename):
  try:
    stat(inputsDir)
  except Exception:
    print " ERROR: modUPS: the inputs directory (%s) does not exist" % inputsDir
    exit(1)

  try:
    stat(org_filename)
  except Exception:
    print " ERROR: modUPS: the orginal ups file (%s) does not exist" % org_filename
    exit(1)


#______________________________________________________________________
# Append a number to the end of the filename.  You need to return a unique filename
# if a base ups file is modified more than once from multiple modUPS calls.
# go through loop until stat fails
def getUniqueFilename(org_filename, inputsDir):

  #__________________________________
  #  create inputsDir/tmp directory
  tmpDir = "%s/tmp" % inputsDir
  try:
    stat( tmpDir  )
  except Exception:
    mkdir(tmpDir )

  #__________________________________
  # Append number to filename
  basename = path.basename(org_filename)
  count = 1
  appendedFilename = "null"

  try:
    while 1:
      appendedFilename = "%s.%d" % (basename,count)
      stat(appendedFilename)
      count = count + 1

  except Exception:
    mod_filename = "%s/%s" % (tmpDir, appendedFilename)
  #__________________________________
  # copy filename to tmpDir
  command = "cp %s %s" % (org_filename, mod_filename)
  system(command)

  return mod_filename

#______________________________________________________________________
# pass the ups directory and filename, and a list of changes
# change will be the entire tag change, i.e.,
# ["<patches> [2,2,2] </patches>",'<burn type = "null" />']
# would be a list of changes

def modUPS(inputsDir, filename, changes):

    org_filename = "%s/%s" % (inputsDir, filename)


    bulletProofInputs(inputsDir, org_filename)

    mod_filename = getUniqueFilename(org_filename, inputsDir)

    #__________________________________
    #  Apply changes using sed
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

    #return the relative path to the inputsDir
    relPath = path.relpath(mod_filename, inputsDir)

    return relPath


#______________________________________________________________________
#  This script uses xmlstarlet to modify the ups file and is designed to
#  change the same xml tag that is in multiple locations.  It will also delete xml tags.
#
#   usage:
#   mod_ups = modUPS2( inputDirectory, <ups file>, [ (<operation>, "xmlpath : value" )]
#
#     where: operation is either "update" or  "delete"
#
#   Example:
#   RMCRT_DO_perf_GPU_ups = modUPS2( the_dir, \
#                               "RMCRT_DO_perf.ups", \
#                               [( "update", "/Uintah_specification/Grid/Level/Box[@label=0]/resolution :[32,32,32]" ),
#                                ( "update", "/Uintah_specification/Grid/Level/Box[@label=0]/patches    :[2,2,2]"    ),
#                                ( "update", "/Uintah_specification/Grid/Level/Box[@label=1]/resolution :[64,64,64]" ),
#                                ( "update", "/Uintah_specification/Grid/Level/Box[@label=1]/patches    :[4,4,4]"    ),
#                                ( "update", "Uintah_specification/RMCRT/nDivQRays                      : 100"       )
#                               ] )
#  This script depends on xmlstarlet.
#  Use:
#       xmlstarlet el -v <ups>
#  to see the path to the xml node

def modUPS2( inputsDir, filename, changes):

    org_filename = "%s/%s" % (inputsDir, filename)
    tmp_filename  = "%s/tmp/%s.tmp" % (inputsDir, filename)

    bulletProofInputs(inputsDir, org_filename)

    mod_filename = getUniqueFilename(org_filename, inputsDir)

    #__________________________________
    #  bulletproofing
    rc = system("which xmlstarlet > /dev/null 2>&1")

    if rc == 256:
      print "__________________________________"
      print "  ERROR:modUPS.py "
      print "      The command (xmlstarlet) was not found and the file %s was not modified" % filename
      print "__________________________________"
      return

    #__________________________________
    #  Apply changes to ups file
    for change in changes:
      operation = change[0].upper()

      option = change[1].split(":")

      if operation == "UPDATE":
        command = "xmlstarlet edit --update %s --value %s  %s > %s" % (option[0], option[1], mod_filename, tmp_filename )

      elif operation == "DELETE":
        command = "xmlstarlet edit --delete %s %s > %s "% (option[0], mod_filename, tmp_filename )

      else:
        print "__________________________________"
        print "  ERROR:modUPS.py "
        print "      The operation (%s) was not found and the file %s was not modified" % (operation, filename )
        print "__________________________________"
        return
      system(command)

      command = "mv %s %s" % (tmp_filename, mod_filename)
      system(command)

    #return the relative path to the inputsDir
    relPath = path.relpath(mod_filename, inputsDir)

    return relPath
