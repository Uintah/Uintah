#! /bin/sh

cd /tmp

if [ -d uintah ] ; then
	rm -rf uintah
fi

#SVN_SOURCE=https://gforge.sci.utah.edu/svn/uintah/trunk 
SVN_SOURCE=file:///usr/local/uintah_subversion/trunk
#svn checkout https://gforge.sci.utah.edu/svn/uintah/trunk uintah
svn -q checkout $SVN_SOURCE uintah

cd uintah/doc

./runLatex

cp *Guide/*.pdf /var/www/uintah/htdocs

chown www-data.root /var/www/uintah/htdocs/*.pdf
chmod go+r /var/www/uintah/htdocs/*.pdf


cd /tmp/uintah

cd src/scripts/doxygen

sed -e 's/#OUTPUT_DIRECTORY/OUTPUT_DIRECTORY/g' < doxygen_config > doxy_config.temp
sed -e 's#/usr/sci/projects/Uintah/www/dist/doxygen/uintah#/var/www/uintah/htdocs/uintah_doxygen/#g' < doxy_config.temp > doxy_config

cd /tmp/uintah/src

doxygen scripts/doxygen/doxy_config

#if [ -d /var/www/uintah/htdocs/uintah_doxygen ]; then
#	rm -rf /var/www/uintah/htdocs/uintah_doxygen
#        mkdir /var/www/uintah/htdocs/uintah_doxygen
#else
#        mkdir /var/www/uintah/htdocs/uintah_doxygen
#fi

#mv html /var/www/uintah/htdocs/uintah_doxygen/

chown -R www-data.root /var/www/uintah/htdocs/uintah_doxygen
chmod go+r /var/www/uintah/htdocs/uintah_doxygen


exit
