echo "-----------------------------------------------------------------"
echo " Build Environment Setup"
echo "-----------------------------------------------------------------"
echo

source /opt/intel/parallel_studio_xe_2017.4.056/psxevars.sh intel64
#source /opt/intel/itac/2017.3.030/bin/itacvars.sh intel64
#source /opt/rh/devtoolset-2/enable

export CC=icc
export CXX=icpc
export LINK=xiar

echo
echo "-----------------------------------------------------------------"
echo " Installing hwloc-1.10.1"
echo "-----------------------------------------------------------------"
echo

#cd $HOME/installs/tarballs
#wget https://www.open-mpi.org/software/hwloc/v1.11/downloads/hwloc-1.11.7.tar.gz
#tar xvf hwloc-1.11.7.tar.gz
#cd hwloc-1.11.7
cd $HOME/installs/tarballs/hwloc-1.11.7
make clean
make cleanreally
make reallyclean

export CC=icc
export CXX=icpc

export HWLOC_PATH_HOST=$HOME/installs/host/hwloc-1.11.7
export PATH=$HWLOC_PATH_HOST/bin:$PATH

./configure \
    --prefix=$HWLOC_PATH_HOST \
    --enable-static

make all
make install

echo
echo "-----------------------------------------------------------------"
echo " Installing Kokkos"
echo "-----------------------------------------------------------------"
echo

cd $HOME/uintah/trunk

mkdir kokkos-host-O2
mkdir -p kokkos-host-O2/TPLs
mkdir -p kokkos-host-O2/TPLs_src
mkdir -p kokkos-host-O2/TPLs_src/kokkos-build

cd kokkos-host-O2
export TPL_PATH=`pwd`/TPLs
export KOKKOS_PATH=`pwd`/TPLs_src/kokkos

cd TPLs_src
git clone https://github.com/kokkos/kokkos.git

cd kokkos-build
$KOKKOS_PATH/generate_makefile.bash \
  --prefix=$TPL_PATH \
  --kokkos-path=$KOKKOS_PATH \
  --with-openmp \
  --with-serial
#  --arch=KNL

make CC=$CC CXX=$CXX LINK=$LINK
make install CC=$CC CXX=$CXX LINK=$LINK

echo "-----------------------------------------------------------------"
echo " Installing Uintah"
echo "-----------------------------------------------------------------"
echo

cd $HOME/uintah/trunk

if [ ! -d "kokkos-host-O2" ]; then
    mkdir kokkos-host-O2
    echo
    echo "Created directory \"kokkos-host-O2\""
fi

cd kokkos-host-O2

make clean
make cleanreally
make reallyclean

../src/configure \
    --enable-64bit \
    --enable-optimize="-std=c++11 -O2 -xMIC-AVX512 -mt_mpi" \
    --enable-assertion-level=0 \
    --with-mpi-include=/opt/intel/impi/2017.3.196/include64 \
    --with-mpi-lib=/opt/intel/impi/2017.3.196/lib64 \
    LDFLAGS="-L$TPL_PATH/lib -lkokkos" \
    CXXFLAGS="-I$TPL_PATH/include -DUINTAH_ENABLE_KOKKOS" \
    CC=mpiicc \
    CXX=mpiicpc \
    F77=mpiifort

echo
echo "-----------------------------------------------------------------"
echo " Building Uintah"
echo "-----------------------------------------------------------------"
echo

make sus

echo
echo "-----------------------------------------------------------------"
echo " Moving sus Executable to tmp.kokkos-host-O2"
echo "-----------------------------------------------------------------"
echo

cd $HOME/uintah/trunk

if [ ! -d "tmp.kokkos-host-O2" ]; then
    mkdir tmp.kokkos-host-O2
    echo
    echo "Created directory \"tmp.kokkos-host-O2\""
fi

cd kokkos-host-O2/StandAlone
cp sus $HOME/uintah/trunk/tmp.kokkos-host-O2/sus.kokkos-host-O2

echo
echo "-----------------------------------------------------------------"
echo " Going Down Successfully"
echo "-----------------------------------------------------------------"
echo
