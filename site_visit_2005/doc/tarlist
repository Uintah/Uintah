# -*- ruby -*-
#
#  For more information, please see: http://software.sci.utah.edu
# 
#  The MIT License
# 
#  Copyright (c) 2004 Scientific Computing and Imaging Institute,
#  University of Utah.
# 
#  License for the specific language governing rights and limitations under
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and associated documentation files (the "Software"),
#  to deal in the Software without restriction, including without limitation
#  the rights to use, copy, modify, merge, publish, distribute, sublicense,
#  and/or sell copies of the Software, and to permit persons to whom the
#  Software is furnished to do so, subject to the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Software.
# 
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
#  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
#  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#  DEALINGS IN THE SOFTWARE.
#

# Print to stdout a list of files to be included in the tar archive.
# This script is used as a filter by doc/Makefile.

# To do: 
# Consider a per-directory 'doc-include' file which lists
# files to be included in the doc distribution.  Then the
# responsibility of this file would be to slurp up the contents of all
# those files and produce a list of files to be included in the doc
# distro.

Dir.chdir("..")
root=File.basename(Dir.pwd())
Dir.chdir("..")

# First do individual files and directories (non-wildcards)
$stdout.print("#{root}/doc/index.html
#{root}/doc/.htaccess
#{root}/doc/Documentum/templates
#{root}/doc/Installation/index.html
#{root}/doc/Installation/FAQ/faq.html
#{root}/doc/Installation/FAQ/faq.pdf
#{root}/doc/User/index.html
#{root}/doc/User/FAQ/faq.html
#{root}/doc/User/FAQ/faq.pdf
#{root}/doc/User/Guide/usersguide
#{root}/doc/User/Guide/Figures
#{root}/doc/User/Guide/usersguide.pdf
#{root}/doc/User/Tutorials/BioFEM
#{root}/doc/User/Tutorials/BioTensor
#{root}/doc/User/Tutorials/FusionViewer
#{root}/doc/User/Tutorials/SCIRun_Intro
#{root}/doc/User/Tutorials/BioImage/bioimage.pdf
#{root}/doc/User/Tutorials/BioImage/Figures
#{root}/doc/Developer/index.html
#{root}/doc/Developer/FAQ/faq.html
#{root}/doc/Developer/FAQ/faq.pdf
#{root}/doc/Developer/Guide/dev/Figures/ComponentWizard.gif
#{root}/doc/Developer/Guide/dev/Figures/ComponentWizard2.gif
#{root}/doc/Developer/Guide/dev/Figures/EditPort.gif
#{root}/doc/Developer/Guide/dev/Figures/msts.gif
#{root}/doc/Developer/Guide/dev/Figures/pdts.gif
#{root}/doc/Developer/Guide/srdg.pdf
#{root}/doc/Developer/Modules/srmrg.pdf
#{root}/doc/Developer/CodeViews/html
#{root}/doc/Developer/Tutorials
#{root}/doc/Utilities/Figures
#{root}/doc/Utilities/TeX/scirun-doc.sty
#{root}/doc/Utilities/TeX/scirun-doc.tex
#{root}/doc/ReleaseNotes
")

# Now do wildcards
globs = ["#{root}/doc/User/FAQ/*.gif",
  "#{root}/doc/User/Tutorials/BioImage/*.html",  
  "#{root}/doc/Installation/Guide/*/srig.pdf",
  "#{root}/doc/Installation/Guide/*/*.html",
  "#{root}/doc/Documentum/*.{html,pdf}",
  "#{root}/doc/Developer/Guide/*.html",
  "#{root}/doc/Developer/Modules/*.html",
  "#{root}/doc/Utilities/HTML/*.{css,js}",
  "#{root}/src/Dataflow/XML/*.html",
  "#{root}/src/Dataflow/Modules/*/doc/*.{jpg,gif,png}",
  "#{root}/src/Packages/*/Dataflow/XML/*.{html,jpg,gif,png}",
  "#{root}/src/Packages/*/Dataflow/TeX/*/*.{jpg,gif,png}",
  "#{root}/src/Packages/*/Dataflow/TeX/*/*/*",
  "#{root}/src/Packages/*/Dataflow/Modules/*/doc/*.{jpg,gif,png}"]

globs.each do |glob|
  Dir.glob(glob).each do |file|
    $stdout.print(file, "\n") if file != ""
  end
end

