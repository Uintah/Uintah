File: SCIRun/src/Packages/VS/Standalone/scripts/README.VSgetFME.p

Description:
	README file for
	VSgetFME.p
	Part of the SCIRun Virtual Soldier (VS) Package
	Relating specifically to the HotBox module.

VSgetFME.p is a short Perl executable (file mode -rwxr-xr-x)
which calls a Web browser (Mozilla)
with arguments: -remote 'openurl( ... )'

The following URL:
http://fme.biostr.washington.edu:8089/FME/index.jsp?initialConcept=$_

Effectively makes a query to the University of Washington (UW) Foundational
Model of Anatomy (FMA) via an Http Web Service (port 80), passing through the
command-line arguments to VSgetFME.p as the string (initialConcept) to
query.

The result is formatted for  display by the UW Foundational Model Explorer
(FME) back-end server.  The FME display is further navigable in Mozilla --
but currently there is no further connection between the FME and the VS/HotBox.

Author:  Stewart Dickson, Visualization Researcher
Computer Science and Mathematics Division
Oak Ridge National Laboratory
