# usage: ./remove_xml_block blockToRemove
# example: ./remove_xml_block Kolmogorov  - will remove all LINES that contains the word Kolmogorov
echo Removing all lines that contain $1
sed -i '/'$1'/d' *.ups