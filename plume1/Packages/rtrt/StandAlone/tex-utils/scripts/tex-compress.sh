#!/bin/tcsh

# Set argument defaults
set ibname = ""
set method = ""
set arguments = ""
set args_set = "false"
set gamma = "false"
set xres = "16"
set yres = "16"
set obname = ""
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

    # Search for "-o <basename>" in argument string
    while ($i <= $#)
      set compress_arg = $argv[$i]

      switch ($compress_arg)
      case "-o":
        @ i++
        set obname = $argv[$i]
        breaksw
      endsw

      @ i++
    end

    breaksw
  case "-res":
    @ i++
    set xres = $argv[$i]
    @ i++
    set yres = $argv[$i]
    breaksw
  case "-gamma":
    set gamma = "true"
    breaksw
  case "--help":
    echo "usage:  tex-compress [options] -i <basename> -c <method> -args <string>"
    echo "options:"
    echo "  -res <int> <int>   set texture width and height (16, 16)"
    echo "  -gamma             gamma correct and quantize textures (false)"
    echo ""

    # Display batch-pca arguments
    echo "batch-pca arguments:"
    $util_dir/batch-pca --help
    echo ""
 
    # Display pnn-vq arguments
    echo "pnn-vq arguments:"
    $util_dir/pnn-vq --help
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

if ($args_set != "true") then
  echo "error:  texture compression argument string not specified"
  exit 1
endif

if ($method == "") then
  echo "error:  compression method not specified"
  exit 1
else if ($method != "pca" && $method != "vq") then
  echo "error:  unrecognized compression method:  $method"
  exit 1
endif

if ($obname == "") then
  set obname = $method
endif

# Print configuration
echo "Input basename:      $ibname"
echo "Output basename:     $obname"
echo "Compression method:  $method"
echo "Gamma/quantize:      $gamma"
echo "Argument string:     $arguments"
echo ""

# Save configuration
echo "Input basename:      $ibname" > tex-compress.$method-status
echo "Output basename:     $obname" >> tex-compress.$method-status
echo "Compression method:  $method" >> tex-compress.$method-status
echo "Gamma/quantize:      $gamma" >> tex-compress.$method-status
echo "Argument string:     $arguments" >> tex-compress.$method-status
echo "" >> tex-compress.$method-status

if ($gamma == "true") then
  # Gamma correct input textures
  echo "Gamma correcting and quantizing input textures:"
  echo "  unu gamma -g 2 -i $ibname.nrrd | unu quantize -b 8 -o $ibname-q.nrrd"

  unu gamma -g 2 -i $ibname.nrrd | unu quantize -b 8 -o $ibname-q.nrrd
  if ($status != "0") then
    echo "failed to gamma correct and quantize textures"
    exit 1
  endif
  
  echo ""

endif

# Compress textures 
echo "Compressing textures:"
if ($method == "pca") then

  # Create temporary files
  unu save -f nrrd -i $ibname-q.nrrd -o $ibname-q.nhdr
  if ($status != "0") then
    echo "failed to create temporary files"
    exit 1
  endif

  # Save start time
  date >> tex-compress.$method-status
  if ($status != "0") then
    echo "error:  failed to save start time"
    exit 1
  endif

  # PCA compression
  echo "  batch-pca -i $ibname-q.nrrd $arguments"
  echo ""

  $util_dir/batch-pca -i $ibname-q.nrrd $arguments
  if ($status != "0") then
    echo "batch-pca failed"
    exit 1
  endif

  echo ""

  # Save end time
  date >> tex-compress.$method-status
  if ($status != "0") then
    echo "error:  failed to save start time"
    exit 1
  endif

  # Remove temporary files
  rm -rf $ibname-q.{nhdr,raw}
  if ($status != "0") then
    echo "failed to remove temporary files"
    exit 1
  endif

  # Create coefficient nrrd
  unu save -f nrrd -i $obname-coeff.nhdr -o $obname-coeff.nrrd
  if ($status != "0") then
    echo "failed to create coefficient nrrd"
    exit 1
  endif

  rm -rf $obname-coeff.{nhdr,raw}
  if ($status != "0") then
    echo "failed to remove temporary coefficient files"
    exit 1
  endif

  # Quantize results
  echo "Quantizing compressed textures:"
  echo "  unu quantize -b 8 -i $obname-basis.nrrd -o $obname-basis-q.nrrd"

  unu quantize -b 8 -i $obname-basis.nrrd -o $obname-basis-q.nrrd
  if ($status != "0") then
    echo "failed to quantize basis vectors"
    exit 1
  endif

  echo "  unu quantize -b 8 -i $obname-coeff.nrrd -o $obname-coeff-q.nhdr"

  unu quantize -b 8 -i $obname-coeff.nrrd -o $obname-coeff-q.nrrd
  if ($status != "0") then
    echo "failed to quantize coefficents"
    exit 1
  endif

  echo "  unu quantize -b 8 -min 0 -max 255 -i $obname-mean.nrrd -o $obname-mean-q.nrrd"

  unu quantize -b 8 -min 0 -max 255 -i $obname-mean.nrrd -o $obname-mean-q.nrrd
  if ($status != "0") then
    echo "failed to quantize mean vector"
    exit 1
  endif

else if ($method == "vq") then

  # Save start time
  date >> tex-compress.$method-status
  if ($status != "0") then
    echo "error:  failed to save start time"
    exit 1
  endif

  # VQ compression
  echo "  pnn-vq -i $ibname-q.nrrd $arguments"
  echo ""

  $util_dir/pnn-vq -i $ibname-q.nrrd $arguments
  if ($status != "0") then
    echo "pnn-vq failed"
    exit 1
  endif

  echo ""

  # Save end time
  date >> tex-compress.$method-status
  if ($status != "0") then
    echo "error:  failed to save start time"
    exit 1
  endif

  # Quantize results
  echo "Quantizing codebook:"
  echo "  unu quantize -b 8 -i $obname-cb.nrrd -o $obname-cb-q.nrrd"

  unu quantize -b 8 -i $obname-cb.nrrd -o $obname-cb-q.nrrd
  if ($status != "0") then
    echo "failed to quantize codebook"
    exit 1
  endif

  echo ""

  # Split axis
  echo "Splitting codebook axis[0]:"
  echo "  unu axsplit -a 0 -s $xres $yres -i $obname-cb-q.nrrd -o $obname-cb-q.nrrd"

  unu axsplit -a 0 -s $xres $yres -i $obname-cb-q.nrrd -o $obname-cb-q.nrrd
  if ($status != "0") then
    echo "failed to split codebook axis[0]"
    exit 1
  endif

endif
echo ""
