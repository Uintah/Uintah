#! /bin/sh

cd /tmp

svn checkout https://gforge.sci.utah.edu/svn/uintah/trunk/doc uintah/doc

cd uintah/doc

./runLatex


cp *Guide/*.pdf /var/www/uintah/htdocs

chown www-data.root /var/www/uintah/htdocs/*.pdf
chmod go+r /var/www/uintah/htdocs/*.pdf

exit
