#! /bin/sh

cd /tmp

if [ -d uintah ] ; then
	rm -rf uintah
fi

svn checkout https://gforge.sci.utah.edu/svn/uintah/trunk uintah

cd uintah/doc

./runLatex

cp *Guide/*.pdf /var/www/uintah/htdocs

chown www-data.root /var/www/uintah/htdocs/*.pdf
chmod go+r /var/www/uintah/htdocs/*.pdf


cd /tmp/uintah

cd src/scripts/doxygen

sed -e 's/OUTPUT_DIRECTORY/#OUTPUT_DIRECTORY/g' < doxygen_config > doxy_config

cd /tmp/uintah

doxygen src/scripts/doxygen/doxy_config

if [ -d /var/www/uintah/htdocs/doxygen ]; then
	rm -rf /var/www/uintah/htdocs/doxygen
fi

mv html /var/www/uintah/htdocs/doxygen

chown -R www-data.root /var/www/uintah/htdocs/doxygen
chmod go+r /var/www/uintah/htdocs/doxygen


exit
