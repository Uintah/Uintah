#!/bin/tcsh

# Set argument defaults
set ibname = ""
set method = ""
set arguments = ""
set args_set = "false"
set util_dir = "/home/sci/cgribble/SCIRun/irix.64/Packages/rtrt/StandAlone/tex-utils"

# Parse command line
set i = 1
while ($i <= $#)
  set arg=$argv[$i]

  switch ($arg)
  case "-i":
    @ i++
    set ibname = $argv[$i]
    breaksw
  case "-c":
    @ i++
    set method = $argv[$i]
    breaksw
  case "-args":
    @ i++
    set arguments = "$argv[$i-$#]"
    set args_set = "true"
    set i = $#
    breaksw
  case "--help":
    echo "usage:  tex-error -i <basename> -c <method> [options]"
    echo "options:"
    echo "  -args <string>   set argument string for error utility (null)"
    echo ""

    # Display pca-error arguments
    echo "pca-error arguments:"
    $util_dir/pca-error --help
    echo ""

    # Display vq-error
    echo "vq-error arguments:"
    $util_dir/vq-error --help
    echo ""
 
    breaksw
  default:
    echo "error:  unrecognized option:  $arg"
    exit 1
  endsw

  @ i++
end

# Validate arguments
if ($ibname == "") then
  echo "error:  input basename not specified"
  exit 1
endif

if ($method == "") then
  echo "error:  compression method not specified"
  exit 1
else if ($method != "pca" && $method != "vq") then
  echo "error:  unrecognized compression method:  $method"
  exit 1
endif

# Print configuration
echo "Input basename:      $ibname"
echo "Compression method:  $method"
if ($arguments == "") then
echo "Argument string:     (null)"
else
echo "Argument string:     $arguments"
endif
echo ""


# Create temporary files
unu save -f nrrd -i $ibname.nrrd -o $ibname.nhdr
if ($status != "0") then
  echo "failed to create temporary files"
  exit 1
endif

# Measure error
echo "Measuring error:"
if ($method == "pca") then

  # Create temporary files
  unu save -f nrrd -i pca-coeff-q.nrrd -o pca-coeff-q.nhdr
  if ($status != "0") then
    echo "failed to create temporary files"
    exit 1
  endif

  # Measure PCA error
  echo "  pca-error -v $ibname.nrrd $arguments"
  echo ""

  $util_dir/pca-error -v $ibname.nrrd $arguments
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

else if ($method == "vq") then

  # Measure VQ error
  echo "  vq-error -v $ibname.nrrd $arguments"
  echo ""

  $util_dir/vq-error -v $ibname.nrrd $arguments
  if ($status != "0") then
    echo "vq-error failed"
    exit 1
  endif

endif
echo ""

# Remove temporary files
rm -f $ibname.{nhdr,raw}
if ($status != "0") then
  echo "failed to remove temporary files"
  exit 1
endif
