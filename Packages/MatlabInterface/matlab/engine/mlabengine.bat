#! /bin/sh

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
