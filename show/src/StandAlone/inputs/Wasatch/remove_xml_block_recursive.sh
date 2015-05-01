# usage: ./remove_xml_block blockToRemove
# example: ./remove_xml_block Kolmogorov  - will remove all LINES that contains the word Kolmogorov
echo Removing all lines that contain $1
find ./ -type f -name "*.ups" -exec sed -i .sedtmp '/'$1'/d' {} +;
find ./ -type f -name "*.sedtmp" -exec rm {} +;