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

.PHONY: faqs user developer installation clean veryclean

MAKE = make
FINALTARGET = SCIRunDocs.tar.gz

default: $(FINALTARGET)

$(FINALTARGET): SCIRunDocs.tar
	rm -f SCIRunDocs.tar.gz
	gzip SCIRunDocs.tar

SCIRunDocs.tar: Utilities/HTML/doc_styles.css index.html faqs developer installation user
	rm -f SCIRunDocs.tar
	cd .. && tar cvf doc/SCIRunDocs.tar doc/User/Guide/usersguide \
                   doc/User/Guide/index.html \
		   doc/User/Guide/figures doc/User/Guide/tutorial \
		   doc/User/Guide/usersguide.ps.gz \
		   doc/User/Guide/usersguide.pdf \
                   doc/Developer/Guide/*.html \
                   doc/Installation/Guide/*.html \
                   doc/FAQBook/*.html \
                   doc/Utilites/HTML/doc_styles.css doc/index.html doc/Figures doc/.htaccess


faqs:
	cd faqs && $(MAKE)

developer:
	cd Developer && $(MAKE)

installation:
	cd Installation && $(MAKE)

user:
	cd User && $(MAKE)

clean:
	rm -f SCIRunDocs.tar.gz
	rm -f SCIRunDocs.tar
	cd Installation && $(MAKE) clean
	cd Developer && $(MAKE) clean
	cd User && $(MAKE) clean
	cd faqs && $(MAKE) clean

veryclean:
	rm -f SCIRunDocs.tar.gz
	rm -f SCIRunDocs.tar
	cd Installation && $(MAKE) veryclean
	cd Developer && $(MAKE) veryclean
	cd User && $(MAKE) veryclean
	cd faqs && $(MAKE) veryclean
