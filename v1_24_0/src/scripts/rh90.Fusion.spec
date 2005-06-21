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

%define defname SCIRun
%define defver	1.24
%define dotver  2
%define gccver  3.2.2
%define plat	rh9.0
%define distro  Red Hat 9.0
%define debug   opt
%undefine	__check_files
%define thirdpartydotver 2
%define thirdpartyversion %{defver}.%{thirdpartydotver}
%define hdf5    hdf5-1.6.4
%define ftgl    ftgl-2.0.9


Name:		%{defname}Fusion
Version:	%{defver}.%{dotver}
Serial:		9
Release:	%{plat}
Summary:	Problem Solving Environment Software
Copyright:	University of Utah Limited
Group:		Applications
URL:		http://www.sci.utah.edu
Distribution:	%{distro}
#Icon:		%{name}.xpm
Vendor:		Scientific Computing & Imaging Institute at the University of Utah
Packager:	McKay Davis <scirun-users@sci.utah.edu>
Provides: libBLT24.so libBLTlite24.so libCore_2d.so libCore_Algorithms_DataIO.so libCore_Algorithms_GLVolumeRenderer.so libCore_Algorithms_Geometry.so libCore_Algorithms_Visualization.so libCore_Containers.so libCore_Datatypes.so libCore_Exceptions.so libCore_GLVolumeRenderer.so libCore_Geom.so libCore_GeomInterface.so libCore_Geometry.so libCore_GuiInterface.so libCore_Malloc.so libCore_Math.so libCore_OS.so libCore_Persistent.so libCore_Process.so libCore_Thread.so libCore_TkExtensions.so libCore_Util.so libDataflow_Comm.so libDataflow_Constraints.so libDataflow_Modules_DataIO.so libDataflow_Modules_Fields.so libDataflow_Modules_Math.so libDataflow_Modules_Render.so libDataflow_Modules_Visualization.so libDataflow_Network.so libDataflow_Ports.so libDataflow_Widgets.so libDataflow_XMLUtil.so libPackages_DataIO_Core_ThirdParty.so libPackages_DataIO_Dataflow_Modules_Readers.so libPackages_Fusion_Dataflow_Modules_Fields.so libPackages_Fusion_Dataflow_Modules_Render.so libPackages_Teem_Core_Datatypes.so libPackages_Teem_Dataflow_Modules_DataIO.so libPackages_Teem_Dataflow_Modules_NrrdData.so libPackages_Teem_Dataflow_Modules_Tend.so libPackages_Teem_Dataflow_Modules_Unu.so libPackages_Teem_Dataflow_Ports.so libair.so libalan.so libbane.so libbiff.so libdye.so libecho.so libell.so libgage.so libhdf5.so.0 libhest.so libhoover.so libitcl3.1.so libitk3.1.so liblimn.so libmite.so libmoss.so libnrrd.so libtcl8.3.so libteem.so libten.so libtk8.3.so libunrrdu.so libxerces-c.so.21 perl(Image::Magick) = 5.43 perl(Turtle)
Requires(rpmlib): rpmlib(CompressedFileNames) <= 3.0.4-1 rpmlib(PartialHardlinkSets) <= 4.0.4-1 rpmlib(PayloadFilesHavePrefix) <= 4.0-1 rpmlib(VersionedDependencies) <= 3.0.3-1
Requires: mdsplus  gcc-c++ >= %{gccver} /bin/bash /bin/csh /bin/sh /usr/bin/env /usr/bin/perl libBLT24.so libBLTlite24.so libCore_Algorithms_DataIO.so libCore_Algorithms_GLVolumeRenderer.so libCore_Algorithms_Geometry.so libCore_Algorithms_Visualization.so libCore_Containers.so libCore_Datatypes.so libCore_Exceptions.so libCore_GLVolumeRenderer.so libCore_Geom.so libCore_GeomInterface.so libCore_Geometry.so libCore_GuiInterface.so libCore_Malloc.so libCore_Math.so libCore_Persistent.so libCore_Thread.so libCore_TkExtensions.so libCore_Util.so libDataflow_Comm.so libDataflow_Constraints.so libDataflow_Modules_Render.so libDataflow_Network.so libDataflow_Ports.so libDataflow_Widgets.so libDataflow_XMLUtil.so libGL.so.1 libGLU.so.1 libPackages_DataIO_Core_ThirdParty.so libPackages_Teem_Core_Datatypes.so libPackages_Teem_Dataflow_Ports.so libX11.so.6 libXaw.so.7 libXext.so.6 libXi.so.6 libXmu.so.6 libXt.so.6 libbz2.so.1 libc.so.6 libc.so.6(GLIBC_2.0) libc.so.6(GLIBC_2.1) libc.so.6(GLIBC_2.1.1) libc.so.6(GLIBC_2.1.3) libc.so.6(GLIBC_2.2) libc.so.6(GLIBC_2.3) libdl.so.2 libdl.so.2(GLIBC_2.0) libdl.so.2(GLIBC_2.1) libfreetype.so.6 libgcc_s.so.1 libgcc_s.so.1(GCC_3.0) libgcc_s.so.1(GLIBC_2.0) libhdf5.so.0 libitcl3.1.so libitk3.1.so libjpeg.so.62 libm.so.6 libm.so.6(GLIBC_2.0) libnsl.so.1 libpng12.so.0 libpthread.so.0 libpthread.so.0(GLIBC_2.0) libpthread.so.0(GLIBC_2.1) libpthread.so.0(GLIBC_2.2) libpthread.so.0(GLIBC_2.3.2) libstdc++.so.5 libstdc++.so.5(CXXABI_1.2) libstdc++.so.5(GLIBCPP_3.2) libstdc++.so.5(GLIBCPP_3.2.2) libtcl8.3.so libteem.so libtiff.so.3 libtk8.3.so libxerces-c.so.21 libz.so.1 perl >= 0:5.002 perl(AutoLoader) perl(Carp) perl(DynaLoader) perl(Exporter) perl(strict) perl(vars)

conflicts: SCIRun SCIRunWithFusion SCIRunBioPSE
AutoReqProv:	NO

ExclusiveOS:	linux

source0:	Thirdparty_install.%{thirdpartyversion}.tar.gz
source1:	%{defname}.%{version}.tar.gz
source2:	Teem.PKG.%{version}.tar.gz
source3:	Fusion.PKG.%{version}.tar.gz
source4:	DataIO.PKG.%{version}.tar.gz
source5:	%{hdf5}.tar.gz
source6:	%{ftgl}.tar.gz
source7:	SCIRun-otf-files.tar.gz

%description
SCIRun is a Problem Solving Environment (PSE), and a computational steering software system. SCIRun allows a scientist or engineer to interactively steer a computation, changing parameters, recomputing, and then revisualizing--all within the same programming environment. The tightly integrated modular environment provided by SCIRun allows computational steering to be applied to the broad range of advanced scientific computations that are addressed by the SCI Institute.


%prep
rm -rf $RPM_BUILD_DIR/Thirdparty_install.*
cd $RPM_BUILD_DIR
tar xvzf %{SOURCE0}

rm -rf /usr/local/SCIRun
cd /usr/local
tar xvzf %{SOURCE1}

cd /usr/local/%{defname}/src/Packages
tar xvzf %{SOURCE2}
tar xvzf %{SOURCE3}
tar xvzf %{SOURCE4}

rm -rf $RPM_BUILD_DIR/%{hdf5}
cd $RPM_BUILD_DIR
tar -xzvf %{SOURCE5}

rm -rf  $RPM_BUILD_DIR/%{ftgl}
mkdir -p $RPM_BUILD_DIR/%{ftgl}
cd $RPM_BUILD_DIR/%{ftgl}
tar -xvzf %{SOURCE6}


%build
#cd $RPM_BUILD_DIR/%{ftgl}/FTGL/unix
#./configure --prefix=/usr/local/%{ftgl}
#make
#make install 
#cp $RPM_BUILD_DIR/%{ftgl}/FTGL/COPYING.txt /usr/local/%{ftgl}


cd $RPM_BUILD_DIR/%{hdf5} 
./configure --enable-threadsafe --with-pthread=/usr --prefix=/usr/local/%{hdf5}
make
make install
cp $RPM_BUILD_DIR/%{hdf5}/COPYING /usr/local/%{hdf5}

cd $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}
python $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}/install /usr/local/SCIRun/Thirdparty 32 1


rm -rf /usr/local/SCIRun/bin
mkdir -p /usr/local/SCIRun/bin
cd /usr/local/SCIRun/bin
/usr/local/SCIRun/src/configure --with-thirdparty="/usr/local/SCIRun/Thirdparty/%{thirdpartyversion}/Linux/gcc-%{gccver}-32bit/" --with-ftgl="/usr/local/%{ftgl}" --with-hdf5="/usr/local/%{hdf5}" --with-mdsplus="/usr/local/mdsplus" --enable-package="Fusion DataIO Teem"
cd /usr/local/SCIRun/bin/on-the-fly-libs
tar -xvzf %{SOURCE7}
cd /usr/local/SCIRun/bin/
gmake

%install
chown -R root.root /usr/local/SCIRun /usr/local/%{hdf5} /usr/local/%{ftgl}
chmod -R a+r /usr/local/SCIRun /usr/local/%{hdf5} /usr/local/%{ftgl}

%clean
rm -rf $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}
rm -rf $RPM_BUILD_DIR/%{hdf5}
rm -rf $RPM_BUILD_DIR/%{ftgl} 

%files
/usr/local/SCIRun
/usr/local/%{hdf5}
/usr/local/%{ftgl}

%changelog
