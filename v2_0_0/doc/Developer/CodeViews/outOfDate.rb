#
#  The contents of this file are subject to the University of Utah Public
#  License (the "License"); you may not use this file except in compliance
#  with the License.
#
#  Software distributed under the License is distributed on an "AS IS"
#  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
#  License for the specific language governing rights and limitations under
#  the License.
#
#  The Original Source Code is SCIRun, released March 12, 2001.
#
#  The Original Source Code was developed by the University of Utah.
#  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
#  University of Utah. All Rights Reserved.
#

# This program determines if the doxygen generated documents are out
# of date with respect to Doxyconfig and .c, .cc, and .h files in the
# src side of the tree. Returns 0 if out of date and 1 if not.

require 'find'

begin
  timestamp = File.mtime("html")
  exit(0) if File.mtime("Doxyconfig") > timestamp
  sourceMatch = /\.(c|cc|h)$/
  Find.find("../../../src") do |f|
    exit(0) if (f =~ sourceMatch) and (File.mtime(f) > timestamp)
  end
rescue
  exit(0)
end

exit(1)
