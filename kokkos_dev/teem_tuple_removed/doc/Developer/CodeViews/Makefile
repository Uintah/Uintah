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
# Makefile for source code views.
#

.PHONY: default docs clean veryclean

SRC:=../../../src
edition:=../../edition.xml


.DELETE_ON_ERROR:

default: Doxyconfig docs

Doxyconfig: Doxyconfig.r ${edition}
	eruby $< >$@

docs:
	if ( test -d ${SRC} && ( test "$$WITH_CV" = true || test $$MAKELEVEL -eq 1 ) ) && ( test ! -d html || ruby outOfDate.rb ) ;  \
	then \
		test -d html && rm -rf html; \
		doxygen Doxyconfig; \
	fi

clean:
	-rm -f *~

veryclean:
	-rm -rf html Doxyconfig
