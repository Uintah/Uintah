#!/usr/bin/python

from os import stat, system, mkdir, path, getcwd, chdir
from sys import exit

#pass the ups directory and filename, and a list of changes
# change will be the entire tag change, i.e.,
# ["<patches> [2,2,2] </patches>",'<burn type = "null" />'] 
# would be a list of changes

def modUPS(directory, filename, changes):
    realdir = path.normpath(path.join(getcwd(), directory))
    tmpdir = path.normpath(path.join(realdir, "tmp"))
    origfilename = "%s/%s" % (realdir, filename)
    newfilename = "%s/tmp/MOD-%s" % (realdir, filename)
    tempfilename = "%s/tmp/%s.tmp" % (realdir, filename)

    # see if filename exists in directory and create tmp
    try:
      stat(realdir)
    except Exception:
      print "%s does not exist" % realdir
      exit(1)
  
    try:
      stat(origfilename)
    except Exception:
      print "%s does not exist" % origfilename
      exit(1)
    try:
      stat("%s/tmp" % realdir)
    except Exception:
      mkdir("%s/tmp" % realdir)

    # append numbers to the end of tmp file
    # go through loop until stat fails
    append = 1
    try:
      while 1:
        appendedFilename = "%s.%d" % (newfilename,append)
	stat(appendedFilename)
	append = append + 1
    except Exception:
      newfilename = "%s.%d" % (newfilename, append)
      filename = "%s.%d" % (filename, append)

    # copy filename to tmp
    command = "cp %s %s" % (origfilename, newfilename)
    system(command)
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
      command = "echo \'%s\' > sedscript" % sedscript
      system(command)

      command = "sed -f sedscript %s > %s" % (newfilename, tempfilename)
      system(command)
      command = "mv %s %s" % (tempfilename, newfilename)
      system(command)
    #system("rm -f sedscript")
    system("rm -rf tempfilename")
    return "tmp/MOD-%s" % filename
