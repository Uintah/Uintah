%define defname SCIRun
%define defver	1.20
%define dotver  2
%define gccver  3.2
%define plat	mdk9.0
%define distro  Mandrake 9.0
%define debug   opt
%define thirdpartydotver 0
%define thirdpartyversion 1.20
%define hdf5    hdf5-1.6.1


Name:		%{defname}WithFusion
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
Packager:	McKay Davis <scirun-users@sci.utah.edu>

conflicts: SCIRun
AutoReqProv:	yes

ExclusiveOS:	linux

source0:	Thirdparty_install.%{thirdpartyversion}.%{thirdpartydotver}.tar.gz
source1:	%{defname}.%{version}.tar.gz
source2:	Teem.PKG.%{version}.tar.gz
source3:	Fusion.PKG.%{version}.tar.gz
source4:	DataIO.PKG.%{version}.tar.gz
source5:	hdf5-1.6.1.tar.gz


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

rm -rf /usr/local/InsightToolkit*
cd /usr/local
tar -xvzf %{SOURCE3}
	


%build
cd $RPM_BUILD_DIR/%{hdf5} 
./configure --enable-threadsafe --with-pthread=/usr --prefix=/usr/local/hdf5/1.6.1
make
make install 

cd $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}.%{thirdpartydotver}
python $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}.%{thirdpartydotver}/install /usr/local/SCIRun/Thirdparty 32 1


rm -rf /usr/local/SCIRun/bin
mkdir -p /usr/local/SCIRun/bin
cd /usr/local/SCIRun/bin
/usr/local/SCIRun/src/configure --with-thirdparty="/usr/local/SCIRun/Thirdparty/%{defver}/Linux/gcc-%{gccver}-32bit/" --with-hdf5="/usr/local/hdf5/1.6.1" --with-mdsplus="/usr/local/mdsplus"
cd /usr/local/SCIRun/bin/
gmake

%install
chown -R root.root /usr/local/SCIRun /usr/local/hdf5/
chmod -R a+r /usr/local/SCIRun /usr/local/hdf5

%clean
rm -rf $RPM_BUILD_DIR/Thirdparty_install.%{thirdpartyversion}
rm -rf $RPM_BUILD_DIR/%{hdf5}

%files
/usr/local/SCIRun
/usr/local/hdf5

%changelog
