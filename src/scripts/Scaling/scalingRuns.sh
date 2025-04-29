#!/bin/bash

usage()
{
  echo "Usage:"
  echo "  scalingRuns  [options]  - run single node scalability study"
  echo ""
  echo "Options:"
  echo "      -n --node       icx, skx, spr"
  echo "      -r, --res       128, 256, 512"
  echo "      -h, --help      Display usage"
  echo "      -c, --config    Build to use"
  exit 1
}


#-------------------------------------------------------------------

function modify_ups() {
  ups=$1
  L0_res=$2
  L0_patches=$3
  perl -pi -w -e "s/<<L0_res>>/${res_IntVector[$L0_res]}/"         "$INPUT_DIR/$ups"
  perl -pi -w -e "s/<<L0_patches>>/$L0_patches/" "$INPUT_DIR/$ups"
}

function min {
  if [ "$1" -lt "$2" ]; then
    echo "$1"
  else
    echo "$2"
  fi
}


main()
{

  #______________________________________________________________________
  # defaults


  declare skeleton_ups="advect.ups"
  declare -i threads=1
  declare -i config=1
  declare -i L0_res=128
  declare nodeType="null"

  declare -A patches=(["2"]="[2,1,1]" ["4"]="[2,2,1]" ["8"]="[2,2,2]" ["16"]="[4,2,2]" ["32"]="[4,4,2]" ["48"]="[4,4,3]" ["64"]="[4,4,4]" ["112"]="[7,4,4]" ["128"]="[8,4,4]")
  declare -A res_IntVector=(["128"]="[128,128,128]" ["256"]="[256,256,256]" ["512"]="[512,512,512]" )

  #__________________________________
  # parse arguments
  options=$( getopt --name "scalingRuns" --options="h,c:,r:,n:"  --longoptions=help,config:,res:,nodes: -- "$@" )

  if [ $# -ne 6 ] ; then
    echo "Incorrect option provided"
    usage
    exit 1
  fi

  # set is to preserve white spaces and punctuation
  eval set -- "$options"

  while true ; do
    case "$1" in
      -c|--config)
        shift
        config=$1
        ;;
      -r|--res)
        shift
        L0_res=$1
        ;;
      -n|--nodes)
        shift
        nodeType=$1
        ;;
      -h|--help)
        usage
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
  done


 if [[ $nodeType != "spr" && $nodeType != "skx" && $nodeType != "icx" && $nodeType != "arm" ]]; then
    echo " Incorrect nodeType option $nodeType"
    usage
    exit 1
  fi

 if [[ $L0_res != 128 && $L0_res != 256 && $L0_res != 512 ]]; then
    echo " Incorrect res option $L0_res"
    usage
    exit 1
  fi

  #______________________________________________________________________

  declare -x INPUT_DIR="inputs/$L0_res^3_$nodeType"
  MPIRUN_BASE="mpirun"

  case "$nodeType" in
    spr)
      CORES="112 64 32 16 8 4 2"
      ;;
    skx)
      CORES="48 32 16 8 4 2"
      ;;
    icx)
      CORES="80 64 32 16 8 4 2"
      ;;
    arm)
      CORES="128 64 32 16 8 4 2"
      ;;
     *)
      echo "ERROR: improper configuration"
      echo "       exiting..."
      exit
      ;;
  esac

  case "$config" in
    1)
      SUS="sus"
      OUTPUT_BASE="outputs_$nodeType/"
      DESC="$L0_res^3"
      export SCI_DEBUG='ComponentNodeStats:+,ExecTimes:+,ComponentStats:+,WaitTimes:+,MPIStats:+'
      ;;
    2)
      SUS="sus"
      OUTPUT_BASE="outputs_noSCI_DEBUG_$nodeType"
      DESC="$L0_res^3"
      ;;
    *)
      echo "ERROR: improper configuration"
      echo "       exiting..."
      exit
      ;;
  esac


  OUTPUT_DIR="$OUTPUT_BASE/$DESC"


  if [ -n "$INPUT_DIR" ] && [ ! -e "$INPUT_DIR" ]; then
    mkdir "$INPUT_DIR"
  fi

  if [ -n "$OUTPUT_DIR" ] && [ ! -e "$OUTPUT_DIR" ]; then
    mkdir -p "$OUTPUT_DIR"
  fi


  #______________________________________________________________________

  for cores in $CORES; do


    mpiTasks=$((cores*threads))

    # modify ups files
    ups="$mpiTasks.ups"
    L0_patches="${patches[$mpiTasks]}"

    echo "Modifying $mpiTasks, $ups $L0_patches"
    cp "$skeleton_ups" "$INPUT_DIR"/"$ups"
    modify_ups "$ups" "$L0_res" "$L0_patches"




    OUT="$OUTPUT_DIR/$mpiTasks.out"
    UPS="$INPUT_DIR"/"$ups"


    MPIRUN="$MPIRUN_BASE -np $cores"

    echo "---------------------------------"
    date
    echo "cores:      $cores"
    echo "sus:        $SUS"
    echo "ups:        $UPS"
    echo "OUT:        $OUT"
    echo "patches:    $L0_patches"
    echo "mpiTasks:   $mpiTasks"
    echo "nodes:      $SLURM_JOB_NODELIST"
    echo "SCI_DEBUG:  $SCI_DEBUG"

    if [ "$threads" == 1 ]; then
      echo "$MPIRUN $SUS $UPS  &> $OUT"
      $MPIRUN $SUS "$UPS"  &> "$OUT"
    else
      export MPI_THREAD_MULTIPLE
      echo "$MPIRUN $SUS -nthreads $threads $UPS  &> $OUT"
      $MPIRUN $SUS -nthreads $threads "$UPS"  &> "$OUT"
    fi

    echo "---------------------------------"
  done
}
#______________________________________________________________________
#______________________________________________________________________

main "$@"
