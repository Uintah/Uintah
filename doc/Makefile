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

.PHONY: faqs techguide userguide clean veryclean

MAKE = make
FINALTARGET = SCIRunDocs.tar.gz

default: $(FINALTARGET)

$(FINALTARGET): SCIRunDocs.tar
	rm -f SCIRunDocs.tar.gz
	gzip SCIRunDocs.tar

SCIRunDocs.tar: doc_styles.css index.html faqs techguide userguide
	rm -f SCIRunDocs.tar
	cd .. && tar cvf doc/SCIRunDocs.tar doc/UserGuide/usersguide \
                   doc/UserGuide/index.html \
		   doc/UserGuide/figures doc/UserGuide/tutorial \
		   doc/UserGuide/usersguide.ps.gz \
		   doc/UserGuide/usersguide.pdf \
                   doc/TechnicalGuide/*.html \
                   doc/faqs/*.html \
                   doc/doc_styles.css doc/index.html doc/images doc/.htaccess


faqs:
	cd faqs && $(MAKE)

techguide:
	cd TechnicalGuide && $(MAKE)

userguide:
	cd UserGuide && $(MAKE)

clean:
	rm -f SCIRunDocs.tar.gz
	rm -f SCIRunDocs.tar
	cd TechnicalGuide && $(MAKE) clean
	cd UserGuide && $(MAKE) clean
	cd faqs && $(MAKE) clean

veryclean:
	rm -f SCIRunDocs.tar.gz
	rm -f SCIRunDocs.tar
	cd TechnicalGuide && $(MAKE) veryclean
	cd UserGuide && $(MAKE) veryclean
	cd faqs && $(MAKE) veryclean
