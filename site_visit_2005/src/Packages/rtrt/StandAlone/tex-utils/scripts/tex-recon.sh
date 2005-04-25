#!/bin/tcsh

# Set argument defaults
set xres = "16"
set yres = "16"
set tilex = ""
set tiley = ""
set arguments = ""
set args_set = "false"
set util_dir = "/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/tex-utils"

# Parse command line
set i = 1
while ($i <= $#)
  set arg=$argv[$i]

  switch ($arg)
  case "-res":
    @ i++
    set xres = $argv[$i]
    @ i++
    set yres = $argv[$i]
    breaksw
  case "-tile":
    @ i++
    set tilex = $argv[$i]
    @ i++
    set tiley = $argv[$i]
    breaksw
  case "-args":
    @ i++
    set arguments = "$argv[$i-$#]"
    set args_set = "true"
    set i = $#
    breaksw
  case "--help":
    echo "usage:  tex-recon [options]"
    echo "options:"
    echo "  -res <int> <int>    set texture width and height (16, 16)"
    echo "  -tile <int> <int>   display reconstructed results (true)"
    echo "  -args <string>      set argument string for error utility (null)"
    echo ""

    # Display recon-pca arguments
    echo "recon-pca arguments:"
    $util_dir/recon-pca --help
    echo ""
 
    breaksw
  default:
    echo "error:  unrecognized option:  $arg"
    exit 1
  endsw

  @ i++
end

# Create temporary files
unu save -f nrrd -i pca-coeff-q.nrrd -o pca-coeff-q.nhdr
if ($status != "0") then
  echo "failed to create temporary files"
  exit 1
endif

# Reconstruct textures
echo "Reconstructing textures:"
echo "  recon-pca $arguments"
echo ""

$util_dir/recon-pca $arguments
if ($status != "0") then
  echo "pca-error failed"
  exit 1
endif

# Remove temporary files
rm -f pca-coeff-q.{nhdr,raw}
if ($status != "0") then
  echo "failed to remove temporary files"
  exit 1
endif

# Split axis[0]
unu axsplit -a 0 -s $xres $yres -i recon.nhdr -o recon.nrrd
if ($status != "0") then
  echo "failed to split axis[0]"
  exit 1
endif

rm -f recon.{nhdr,raw}
if ($status != "0") then
  echo "failed to remove temporary files"
  exit 1
endif

# Display results
if ($tilex != "" && $tiley != "") then
  unu axsplit -i recon.nrrd -a 2 -s $tilex $tiley \
    | unu permute -p 0 2 1 3 \
    | unu axmerge -a 0 \
    | unu axmerge -a 1 \
    | XV
  if ($status != "0") then
    echo "failed to display results"
    exit 1
  endif
endif

echo ""
