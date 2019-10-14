Environmental Flags that control the regression tester:

* TEST_COMPONENTS: MPM, Wasatch, Arches… specifies which component to run.
* WHICH_TESTS: LOCALTESTS, etc…, specifies which suite of tests to run.
Note that the test suite is component-specific.

Example:
in a bash shell
export TEST_COMPONENTS=Wasatch
export WHICH_TESTS=GPUTESTS

make runLocalRT (or gold_standards) will run the Wasatch:GPUTESTS


______________________________________________________________________

Setting up a new RT machine


Create a user, i.e. rt
and login at that user.


__________________________________

- create a ~/scripts directory and copy

  src/R_Tester/helpers/cleanup 
  src/R_Tester/toplevel/startTester

to it.

__________________________________

- Edit ~/scripts/startTester and add an additional conditional section near the top
for your machine.  For example:

if test "$MACHINE" = "cyrus.mech.utah.edu" ; then
  DEFAULT_MAIL_TO="ahumphrey@sci.utah.edu,t.harman@utah.edu"
  REPLY_TO="t.harman@utah.edu"
  COMMON_GROUP="users"
  BUILD_DIR=/raid/home/rt/"$OS"
  TEST_DATA=/raid/home/rt/"$OS"/TestData
  TEMP_DIR=/raid/home/rt/"$OS"
  SCRIPT_DIR=/raid/home/rt/scripts
  CHECKPOINTS=/raid/home/rt/CheckPoints
  MPIRUN_PATH=/usr/installed/openmpi/bin/
  export TEST_COMPONENTS="Examples"             # only run gpu tests
  
  # Where the webpage will be, and init webpage.  (SEE above notes)
  HTMLLOG=/raid/home/rt/public_html/Uintah.html
  WEBLOG=NONE
fi
__________________________________

- Create the following directories:

mkdir -p Linux/TestData/opt Linux/TestData/dbg CheckPoints public_html/Plots public_html/Uintah.html-dbg public_html/Uintah.html-dbg

__________________________________

- Create a nightly cronjob
crontab -e
# m h  dom mon dow   command
10 1 * * *      $HOME/scripts/cleanup
30 1 * * *      $HOME/scripts/startTester -j 8 -sendmail

__________________________________

- Add this to ~/.bashrc  if you're using a single GPU on a machine that has 
multiple GPUs

# needed for GPU test
export CUDA_VISIBLE_DEVICES=0


__________________________________

- The following environmental variables are useful:

    setenv RT ~/Linux/last_ran.lock
    setenv t_src     $RT/src
    setenv t_comp    $RT/src/CCA/Components
    setenv t_ice    $RT/src/CCA/Components/ICE
    setenv t_arches $RT/src/CCA/Components/Arches
    setenv t_grid   $RT/src/Core/Grid
    setenv t_core   $RT/src/Core
    setenv t_inputs $RT/src/StandAlone/inputs
    setenv t_mpmice $RT/src/CCA/Components/MPMICE
    setenv t_mpm    $RT/src/CCA/Components/MPM
    setenv t_models $RT/src/CCA/Components/Models
    setenv t_rmcrt  $RT/src/CCA/Components/Models/Radiation/RMCRT
    setenv t_dbg    $RT/dbg/
    setenv t_opt    $RT/opt/

__________________________________
-To enable core dumps (recommended for debugging) 
edit 
/etc/security/limints.conf
rt               soft    core            unlimited
rt               hard    core            unlimited

OR
(bash)  ulimit -S -c unlimited sus
(csh/tcsh) ?????


