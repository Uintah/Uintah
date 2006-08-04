#!/bin/tcsh

# Set argument defaults
set ifname = ""
set working_dir = ""
set purge = "false"
set arguments = ""
set args_set = "false"
set obname = "particle"
set util_dir = "/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/tex-utils"
set nrrdstack_dir = "/home/sci/bigler/pub/irix/bin"

# Parse command line
set i = 1
while ($i <= $#)
  set arg=$argv[$i]

  switch ($arg)
  case "-i":
    @ i++
    set ifname = $argv[$i]
    breaksw
  case "-dir":
    @ i++
    set working_dir = $argv[$i]
    breaksw
  case "-purge":
    set purge = "true"
    breaksw
  case "-args":
    @ i++
    set arguments = "$argv[$i-$#]"
    set args_set = "true"

    # Search for "-o <basename>" in argument string
    while ($i <= $#)
      set gen_arg = $argv[$i]

      switch ($gen_arg)
      case "-o":
        @ i++
        set obname = $argv[$i]
      endsw

      @ i++
    end
    breaksw
  case "--help":
    echo "usage:  tex-gen [options] -i <filename> -args <string>"
    echo "options:"
    echo "  -dir <directory>   set working directory (.)"
    echo "  -purge             remove texture subsets (false)"
    echo ""

    # Display genpttex arguments
    echo "genpttex arguments:"
    $util_dir/genpttex --help
    exit 0
    breaksw
  default:
    echo "error:  unrecognized option:  $arg"
    exit 1
  endsw

  @ i++
end

if ($ifname == "") then
  echo "error:  input filename not specified"
  exit 1
endif

if ($args_set != "true") then
  echo "error:  genpttex argument string not specified"
  exit 1
endif

# Print configuration
echo "Input filename:   $ifname"
echo "Output basename:  $obname"
echo "Argument string:  $arguments"
echo ""

# Save configuration
echo "Input filename:   $ifname" > tex-generate.status
echo "Output basename:  $obname" >> tex-generate.status
echo "Argument string:  $arguments" >> tex-generate.status
echo "" >> tex-generate.status

# Change working directory
if ($working_dir != "") then
  echo "Changing working directory:"
  echo "  cd $working_dir"

  cd $working_dir
  if ($status != "0") then
    echo "error:  failed to change working directory to $working_dir"
    exit 1
  endif

  echo ""
endif

# Save start time
date >> tex-generate.status
if ($status != "0") then
  echo "error:  failed to save start time"
  exit 1
endif

# Generate textures
echo "Generating textures:"
echo "  genpttex -i $ifname $arguments"

$util_dir/genpttex -i $ifname $arguments
if ($status != "0") then
  echo "error:  failed to generate textures"
  exit 1
endif

echo ""

# Save end time
date >> tex-generate.status
if ($status != "0") then
  echo "error:  failed to save end time"
  exit 1
endif

# Stack textures
echo "Stacking textures:"
echo "  nrrdstack $obname-all.nrrd $obname???????.nrrd"
echo ""

$nrrdstack_dir/nrrdstack $obname-all.nrrd $obname???????.nrrd
if ($status != "0") then
  echo "error:  failed to stack textures"
  exit 1
endif

echo ""

# Purge texture subsets
if ($purge == "true") then
  echo "Purging texture subsets:"
  echo "  rm -rf $obname???????.nrrd"

  rm -rf $obname???????.nrrd
  if ($status != "0") then
    echo "error:  failed to purge texture subsets"
    exit 1
  endif

  echo ""
endif
