#!/bin/sh
# Make the html version of the users' guide
#
######################################################################

echo "Making the HTML files from srug sources."

# THis version has navigation and section numbers

latex2html -split 4 -no_white -link 1 -bottom_navigation \
-html_version 3.2,math  -show_section_numbers -local_icons usersguide

# THis version has section numbers but no navigation icons

#latex2html -split 3 -no_white -link 3  -show_section_numbers \
#-math -html_version 3.2,math usersguide 
