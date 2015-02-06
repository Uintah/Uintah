#! /bin/sh

cd /usr/local

#rsync -aP --delete --no-g --no-p --no-t blaze.sci.utah.edu:/usr/local/home/csafe-tester/Linux/TestData .
rsync -aPz --delete --exclude CheckPoints blaze.sci.utah.edu:/usr/local/home/csafe-tester/Linux/TestData .

rsync -aPz --delete blaze.sci.utah.edu:/usr/local/home/csafe-tester/CheckPoints TestData

#chmod -R ugo+rw TestData

exit
