
if [ $1 = "stop" ]
then
 echo STOP the matlab engine
 echo 'mlabengine(5,[],'\''stop'\'');' | matlab -nosplash &
fi

if [ $1 = "start" ]
then
 echo START the matlab engine
 echo 'mlabengine(5,'\''127.0.0.1:5517'\'');' | matlab -nosplash &
fi

# to check the system from *.c file
# main() {system("echo 'mlabengine(5,'\\''gauss:5517'\\'');'|matlab -nosplash &");}
