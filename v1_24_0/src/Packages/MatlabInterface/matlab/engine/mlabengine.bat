#! /bin/sh

#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# 
# This script starts or stops the matlab engine.  
#
# Usage: mlabengine.bat <start | stop>
#
# A byproduct of running this script is the creation of the
# 'transport.mexsg' file.  This file is actually created
# in the 'mlabengine.m' foutine.
#

if test "$1" = "stop"; then
 echo 
 echo "Stopping the matlab engine.  THIS MAY TAKE A FEW SECONDS"
 echo 'mlabengine(5,[],'\''stop'\'');' | matlab -nosplash &
elif test "$1" = "start"; then
 echo "Starting the matlab engine.  THIS WILL TAKE A FEW SECONDS"
 echo "  You can continue to use this shell, but Matlab output will"
 echo "  periodically appear in this window."
 echo 'mlabengine(5,'\''127.0.0.1:5517'\'');' | matlab -nosplash &
else
 echo "Usage: $0 <start | stop>"
fi

# to check the system from *.c file
# main() {system("echo 'mlabengine(5,'\\''gauss:5517'\\'');'|matlab -nosplash &");}
