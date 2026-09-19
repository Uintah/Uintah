#!/bin/bash
#______________________________________________________________________
#
# The MIT License
#
# Copyright (c) 1997-2026 The University of Utah
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
#/
#______________________________________________________________________


#______________________________________________________________________
# Usage:
#  ppm_To_mpg.sh <optional file name base>
#
#   This script 'glues' together a series of image files into a movie.
#   The user can resize the movie and/or add labels to the header/footer.
#   The image files must be named:
#    <basename><separator><NNN>.<ext>
#   where ext can be any image extension, <separator> can be any run of
#   non-digit characters (e.g. "." or "_t"), and NNN is the image number:
#   consecutive, but not required to start at 0 or be padded with 0.
#
# Pseudocode:
#   verify ffmpeg/avconv and display/convert/composite are on PATH
#   find files matching <basename>*.<ext>
#   pick the file with the smallest trailing frame number as firstFrame
#   orgExt := firstFrame's extension
#
#   prompt for: backup y/n, output size, playback fps, movie format
#   confirm with user, looping until accepted
#
#   prompt for: optional top/bottom title text
#     if titles wanted:
#       render titles onto firstFrame and show a preview (`display`)
#       loop until user accepts the preview
#
#   if backup requested:
#     copy all files in cwd into orgs/
#
#   rename pass (validates + renumbers to 0-based):
#     for each candidate file, sorted by its original frame number:
#       if original number != expected next number: error, abort
#       copy it to <0-based index>.<orgExt>
#
#   if orgExt != target extension:
#     batch-convert every renamed frame to the target extension (one
#     mogrify call for all frames), then drop the orgExt copies
#
#   if titles wanted:
#     batch-annotate the top title onto every frame (one mogrify call)
#     batch-annotate the bottom title onto every frame (one mogrify call)
#
#   if resize requested:
#     batch-resize every frame to the chosen size (one mogrify call)
#
#   remove any pre-existing output movie file
#   run ffmpeg/avconv over the numbered frames to build the movie
#
#   prompt: keep the individual frames? if no, delete them
#
#__________________________________

set -u
shopt -s nullglob

#  Print the trailing run of digits in a filename (before its extension)
#  as a base-10 integer, e.g. "A_t00062.png" -> 62, "movie.7.ppm" -> 7.
#  Returns non-zero if the filename has no trailing digits.
extract_frame_num() {

  local base="${1%.*}"

  if [[ "$base" =~ ([0-9]+)$ ]]; then
    echo $((10#${BASH_REMATCH[1]}))
    return 0
  fi
  return 1
}

#__________________________________
#  Does ffmpeg or avconv exist?
FFMPEG="none"
if command -v ffmpeg >/dev/null 2>&1; then
  FFMPEG="ffmpeg"
elif command -v avconv >/dev/null 2>&1; then
  FFMPEG="avconv"
fi

if [ "$FFMPEG" = "none" ]; then
  echo " ERROR: Could not find one of the commands"
  echo "    ffmpeg  or avconv"
  echo "  ... now exiting"
  exit 0
fi

#__________________________________
#  Do the other commands exist
for cmd in display convert composite; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo " ERROR: Could not find the command: $cmd"
    exit 0
  fi
done

#__________________________________
#   check if there was a file basename specified
imageName="movie"
if [ $# -gt 0 ]; then
  imageName="$1"
  echo " Using $imageName as the filename base."
fi

candidates=("$imageName"*.*)
if [ ${#candidates[@]} -eq 0 ]; then
  echo " ERROR: Could not find any frames named $imageName*.<ext>"
  exit 0
fi

#__________________________________
#  frame numbers need not be zero-padded or start at 0, so scan every
#  candidate and keep the one with the smallest trailing number to use
#  as the sample frame.

firstFrame=""
firstNum=""

for f in "${candidates[@]}"; do
  num=$(extract_frame_num "$f") || continue
  if [ -z "$firstNum" ] || [ "$num" -lt "$firstNum" ]; then
    firstNum=$num
    firstFrame="$f"
  fi
done

if [ -z "$firstFrame" ]; then
  echo " ERROR: Could not find any frames named $imageName*.<ext> with a frame number"
  exit 0
fi
orgExt="${firstFrame##*.}"

ans="n"

#__________________________________
# Defaults
EXT="jpeg"
addLabels="false"
doResize="true"
FONT_DEFAULT="-font helvetica -pointsize 14"   # for the labels

while [ "$ans" = "n" ] || [ "$ans" = "N" ]; do
  #__________________________________
  #  User inputs
  echo "Would you like to make a backup of your images?[n]"
  read -r backup

  size_default=$(identify -verbose "$firstFrame" | grep Geometry | cut -d ":" -f2)
  echo "Enter the size of the movie [$size_default] "
  read -r size

  echo "Play back speed, frames per second [5]"
  read -r playBackSpeed

  echo "Enter movie format [mpeg].  Type ffmpeg -formats for options"
  read -r movieFormat

  #__________________________________
  #  apply defaults
  if [ -z "$backup" ]; then
    backup="n"
  fi
  if [ -z "$size" ]; then
    size="$size_default"
    doResize="false"
  fi
  if [ -z "$playBackSpeed" ]; then
    playBackSpeed="5"
  fi
  if [ -z "$movieFormat" ]; then
    movieFormat="mpeg"
  fi

  echo "-------------------"
  echo "backup images..........$backup"
  echo "movie size.............$size"
  echo "movie playback speed...$playBackSpeed"
  echo "movie filetype.........$movieFormat"
  echo "-------------------"
  echo ""
  echo "Is this correct? [y]"
  read -r ans
done

#__________________________________
#  Add Labels and show the user a sample image
echo ""
echo "Do you want to add titles to movie? [n]"
read -r ans
redo=""
N_title=""
S_title=""

while [ "$ans" = "y" ] || [ "$ans" = "Y" ] || [ "$redo" = "n" ]; do
  addLabels="true"

  echo "Enter the title for the top of the image"
  read -r N_title
  echo "Enter the title for the bottom of the movie"
  read -r S_title

  echo "white or black font color (w/b) [w]"
  read -r fontColor

  if [ -z "$fontColor" ]; then
    fontColor="white"
    bkgrdColor="black"
  else
    fontColor="black"
    bkgrdColor="white"
  fi

  FONT="$FONT_DEFAULT -fill $fontColor -background $bkgrdColor"

  num=$(extract_frame_num "$firstFrame")
  convert "$firstFrame" "$num.$EXT"

  #__________________________________
  # generate the labels
  if [ -n "$N_title" ]; then
    convert "$num.$EXT" $FONT -gravity north -annotate +0+5 "$N_title" "test.0.$EXT"
  fi

  cp "test.0.$EXT" "test.a.$EXT"

  if [ -n "$S_title" ]; then
    convert "test.0.$EXT" $FONT -gravity south -annotate +0+5 "$S_title" "test.a.$EXT"
  fi

  echo "Close the popup window to continue"

  display "test.a.$EXT"

  echo "Is this correct? [y]"
  ans="n"
  read -r redo
  rm -f test.*."$EXT"
done

#___________________________________________________
# Now do the work
if [ "$backup" = "y" ] || [ "$backup" = "Y" ]; then
  mkdir -p orgs
  echo "copying images to orgs/"
  cp * orgs/.
fi

#__________________________________
#  rename
echo "Now renaming files $orgExt files into $EXT files "

#  pair each frame with its (possibly non-zero-based, non-padded) frame
#  number, then sort numerically -- the original file's number is only
#  used to check consecutiveness; the renamed copy is always numbered
#  from 0 so ffmpeg's %d pattern works regardless of the original range.
pairs=()
for f in "$imageName"*."$orgExt"; do
  num=$(extract_frame_num "$f") || continue
  pairs+=("$num $f")
done

count=0
expectedNum=""
while read -r num i; do
  if [ -z "$expectedNum" ]; then
    expectedNum=$num
  fi

  echo " Now renaming $i to $count.$orgExt"
  cp "$i" "$count.$orgExt"

  if [ "$num" != "$expectedNum" ]; then
    echo " ERROR: the images are not consecutively numbered"
    echo " Image number is: $num but it should be $expectedNum"
    rm -f [0-9]*."$EXT" [0-9]*."$orgExt"
    exit 0
  fi

  expectedNum=$((expectedNum + 1))
  count=$((count + 1))
done < <(printf '%s\n' "${pairs[@]}" | sort -n)

#__________________________________
#  convert files to $EXT format -- batched into a single ImageMagick
#  invocation across all frames instead of one convert per frame, since
#  each invocation pays a fixed startup cost that dominates for many
#  small/fast frames.
n=$(identify -verbose "$firstFrame" | grep -ci "$EXT")

if [ "$n" -eq 0 ]; then
  echo "Now converting $orgExt files into $EXT files "
  mogrify -format "$EXT" [0-9]*."$orgExt"
  rm -f [0-9]*."$orgExt"
fi

#__________________________________
# add labels -- the title text is identical on every frame, so annotate
# all frames in two batched mogrify calls instead of looping per frame.
if [ "$addLabels" = "true" ]; then
  echo "Now adding labels to all frames"

  if [ -n "$N_title" ]; then
    mogrify $FONT -gravity north -annotate +0+5 "$N_title" [0-9]*."$EXT"
  fi

  if [ -n "$S_title" ]; then
    mogrify $FONT -gravity south -annotate +0+5 "$S_title" [0-9]*."$EXT"
  fi
fi

#__________________________________
# do Resize -- batched across all frames in a single call
if [ "$doResize" = "true" ]; then
  echo "Now resizing $EXT files"
  mogrify -resize "$size" [0-9]*."$EXT"
fi

#__________________________________
# make the movies
echo "___________________________________"
echo "Now making the movie"

rm -f "$imageName.$movieFormat"
CMD="$FFMPEG -r $playBackSpeed -i %d.$EXT -r 30 -q 1 $imageName.$movieFormat"
echo "$CMD"
$CMD

#__________________________________
#  Backup modified images
echo "__________________________________"
echo "Do you want to keep the individual titled frames as $EXT? [n]"
read -r ans

for T in [0-9]*."$EXT"; do
  if [ "$ans" != "y" ]; then
    rm -f "$T"
  fi
done

exit 0
