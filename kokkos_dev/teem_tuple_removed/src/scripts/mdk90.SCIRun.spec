%define defname SCIRun
%define defver	1.20
%define dotver  2
%define gccver  3.2
%define plat	mdk9.0
%define distro  Mandrake 9.0
%define debug   opt
%define thirdpartydotver 0
%define thirdpartyversion %{defver}.%{thirdpartydotver}


Name:		%{defname}
Version:	%{defver}.%{dotver}
Serial:		5
Release:	%{plat}
Summary:	Problem Solving Environment Software
Copyright:	University of Utah Limited
Group:		Applications
URL:		http://www.sci.utah.edu
Distribution:	%{distro}
#Icon:		%{name}.xpm
Vendor:		Scientific Computing & Imaging Institute at the University of Utah
Packager:	McKay Davis <rpm@sci.utah.edu>

Provides:	libair.so libbiff.so libBLT24.so libckit.so libCore_2d.so libCore_Algorithms_Geometry.so libCore_Algorithms_Visualization.so libCore_Containers.so libCore_Datatypes.so libCore_Exceptions.so libCore_Geometry.so libCore_GeomInterface.so libCore_Geom.so libCore_GLVolumeRenderer.so libCore_GuiInterface.so libCore_Malloc.so libCore_Math.so libCore_OS.so libCore_Persistent.so libCore_Process.so libCore_Thread.so libCore_TkExtensions.so libCore_Util.so libDataflow_Comm.so libDataflow_Constraints.so libDataflow_Modules_DataIO.so libDataflow_Modules_Fields.so libDataflow_Modules_Math.so libDataflow_Modules_Render.so libDataflow_Modules_Visualization.so libDataflow_Network.so libDataflow_Ports.so libDataflow_Widgets.so libDataflow_XMLUtil.so libelixir.so libell.so libesi.so libg3d.so libhest.so libitcl3.1.so libitcl.so libitk3.1.so libitk.so liblobs1d.so liblobs2d.so liblobs3d.so liblpk.so libmeschach.so libnrrd.so libPackages_BioPSE_Core_Algorithms_NumApproximation.so libPackages_BioPSE_Core_Datatypes.so libPackages_BioPSE_Dataflow_Modules_DataIO.so libPackages_BioPSE_Dataflow_Modules_Forward.so libPackages_BioPSE_Dataflow_Modules_Inverse.so libPackages_BioPSE_Dataflow_Modules_LeadField.so libPackages_BioPSE_Dataflow_Modules_Visualization.so libPackages_MatlabInterface_Core_Datatypes.so libPackages_MatlabInterface_Core_Util.so libPackages_MatlabInterface_Dataflow_Modules_DataIO.so libPackages_Teem_Core_Datatypes.so libPackages_Teem_Dataflow_Modules_DataIO.so libPackages_Teem_Dataflow_Modules_Filters.so libPackages_Teem_Dataflow_Ports.so libpirat.so libtcl8.3.so libtcl.so libtk8.3.so libtk.so libvdt.so libxerces-c1_5_2.so libxerces-c1_5.so libxerces-c.so meschach.so perl(Image::Magick) = 5.43 perl(Turtle) SCIRun

Requires:	 gcc-c++ >= %{gccver} ld-linux.so.2 libair.so libbiff.so libBLT24.so libbz2.so.1 libCore_Containers.so libCore_Datatypes.so libCore_Exceptions.so libCore_Geometry.so libCore_GeomInterface.so libCore_Geom.so libCore_GuiInterface.so libCore_Malloc.so libCore_Math.so libCore_Persistent.so libCore_Thread.so libCore_TkExtensions.so libCore_Util.so libc.so.6 libDataflow_Comm.so libDataflow_Network.so libDataflow_XMLUtil.so libdl.so.2 libell.so libfreetype.so.6 libgcc_s.so.1 libGL.so.1 libGLU.so.1 libICE.so.6 libitcl3.1.so libitcl.so libitk3.1.so libitk.so libjpeg.so.62 liblpk.so libm.so.6 libnrrd.so libnsl.so.1 libPackages_Teem_Core_Datatypes.so libpng12.so.0 libpthread.so.0 libSM.so.6 libstdc++.so.5 libtcl8.3.so libtcl.so libtiff.so.3 libtk8.3.so libtk.so libX11.so.6 libxerces-c.so libXext.so.6 libxml2.so.2 libz.so.1 libckit.so libCore_Algorithms_Geometry.so libCore_Algorithms_Visualization.so libCore_GLVolumeRenderer.so libDataflow_Constraints.so libDataflow_Modules_Render.so libDataflow_Ports.so libDataflow_Widgets.so libelixir.so libesi.so libg3d.so liblobs1d.so liblobs2d.so liblobs3d.so libmeschach.so libPackages_BioPSE_Core_Algorithms_NumApproximation.so libPackages_BioPSE_Core_Datatypes.so libPackages_MatlabInterface_Core_Util.so libPackages_Teem_Dataflow_Ports.so libpirat.so libvdt.so libXaw.so.7 libXmu.so.6 libXpm.so.4 libXt.so.6 bash perl-base sh-utils tcsh libc.so.6(GLIBC_2.0) libc.so.6(GLIBC_2.1) libc.so.6(GLIBC_2.1.1) libc.so.6(GLIBC_2.1.3) libc.so.6(GLIBC_2.2) libdl.so.2(GLIBC_2.0) libdl.so.2(GLIBC_2.1) libgcc_s.so.1(GCC_3.0) libgcc_s.so.1(GLIBC_2.0) libm.so.6(GLIBC_2.0) libpthread.so.0(GLIBC_2.0) libpthread.so.0(GLIBC_2.1) libstdc++.so.5(GLIBCPP_3.2) 
conflicts: SCIRunWithFusion
AutoReqProv:	no


#ExcludeArch:
#ExclusiveArch: linux
#ExcludeOS:
ExclusiveOS:	linux


#Prefix:	/usr/local/
#%{_tmppath}/%{name}-buildroot
#BuildRoot:     /usr/local/SCIRun


source0:	Thirdparty_install.%{thirdpartyversion}.tar.gz
source1:	%{name}.%{version}.tar.gz
source2:	cmake-1.8.1-x86-linux-files.tar
source3:	InsightToolkit-1.4.0.tar.gz
source4:	SCIRun-otf-files.tar.gz
source5:	BioTensor-otf-files.tar.gz
source6:	BioPSE.PKG.%{version}.tar.gz
source7:	MatlabInterface.PKG.%{version}.tar.gz
source8:	Teem.PKG.%{version}.tar.gz
source9:	Insight.PKG.%{version}.tar.gz
#Patch:
#nopatch

%description
SCIRun is a Problem Solving Environment (PSE), and a computational steering software system. SCIRun allows a scientist or engineer to interactively steer a computation, changing parameters, recomputing, and then revisualizing--all within the same programming environment. The tightly integrated modular environment provided by SCIRun allows computational steering to be applied to the broad range of advanced scientific computations that are addressed by the SCI Institute.


%prep
rm -rf $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}
tar -xvzf %{SOURCE0}

cd /usr/local
rm -rf /usr/local/SCIRun
tar -xvzf %{SOURCE1}

rm -rf $RPM_BUILD_DIR/cmake	
mkdir -p $RPM_BUILD_DIR/cmake
cd $RPM_BUILD_DIR/cmake
tar xvf %{SOURCE2}

rm -rf /usr/local/InsightToolkit*
cd /usr/local
tar -xvzf %{SOURCE3}
	
cd /usr/local/%{defname}/src/Packages
tar -xvzf %{SOURCE6}
tar -xvzf %{SOURCE7}
tar -xvzf %{SOURCE8}
tar -xvzf %{SOURCE9}



%build
cd /usr/local
rm -rf /usr/local/InsightToolkit-1.4.0-bin
mkdir -p /usr/local/InsightToolkit-1.4.0-bin
cd /usr/local/InsightToolkit-1.4.0-bin
$RPM_BUILD_DIR/cmake/bin/cmake /usr/local/InsightToolkit-1.4.0 -DBUILD_EXAMPLES:BOOL=OFF -DBUILD_SHARED_LIBS:BOOL=ON -DBUILD_TESTING:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=Release -DCMAKE_VERBOSE_MAKEFILE:BOOL=TRUE -DITK_USE_SYSTEM_PNG:BOOL=ON
make
make install

cd $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}
python $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}/install /usr/local/SCIRun/Thirdparty 32 1

rm -rf /usr/local/SCIRun/bin
mkdir -p /usr/local/SCIRun/bin
cd /usr/local/SCIRun/bin
export JAVA_HOME=/usr/java/jdk1.3.1_08
export PATH=${JAVA_HOME}/bin:${PATH}
/usr/local/SCIRun/src/configure --with-thirdparty="/usr/local/SCIRun/Thirdparty/%{defver}/Linux/gcc-%{gccver}-32bit/" -with-insight="/usr/local/lib/InsightToolkit"
cd /usr/local/SCIRun/bin/on-the-fly-libs
tar -xvzf %{SOURCE4}
tar -xvzf %{SOURCE5}
cd /usr/local/SCIRun/bin/
gmake


%install
chown -R root.root /usr/local/SCIRun
chmod -R a+r /usr/local/SCIRun

%clean
rm -rf $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}

%files
/usr/local/SCIRun
/usr/local/lib/InsightToolkit

%changelog
