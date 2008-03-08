#!/usr/bin/perl

use XML::Simple;
use Data::Dumper;
$xml = new XML::Simple(forcearray => 1);

# read XML file
$data = $xml->XMLin("$ARGV[0]", KeepRoot => 1);

$gnuplotFile = $data->{start}->[0]->{gnuplotFile}->[0];        # if a user provides a gnuplot file

foreach $e (@{$data->{start}->[0]->{Test}}){

  $test_title   = $e->{Title}->[0];
  $compUtil_cmd = $e->{compareUtil}->[0];
  $output_file  = $test_title;
  $output_file  =~ tr/" "/"_"/;  # Replacing the spaces with underscores
  $x            = $e->{x}->[0];
  
  @uda_files = @{$e->{uda}};
  
  #__________________________________
  # bulletproofing
  my @stripped_cmd = split(/ /,$compUtil_cmd);  # remove command options
  system("which $stripped_cmd[0] >& /dev/null")==0 or die("ERROR(analyze_Analysis.pl): \tThe comparison script/command ($compUtil_cmd) could not be found\n");

  # main loop
  for ($k = 0 ; $k<=$#uda_files; $k++){ 
    
    #__________________________________
    # Run the comparison tests
    my $comp_output = "out.$x.cmp";
    print "Running:  $compUtil_cmd -o $comp_output -uda $uda_files[$k]\n";
    `rm -f $comp_output`;  
   
    @args = ("$compUtil_cmd","-o","$comp_output","-uda","$uda_files[$k]");
    system("@args")==0 or die("ERROR(analyze_Analysis.pl): @args failed: $?");
    
    $L2norm = `cat $comp_output`;
    chomp($L2norm);
    `echo $x $L2norm >> L2norm.dat`;
    system("rm $comp_output");
  }


  #______________________________________________________________________
  # Create a default gnuplot script
  if ( $gnuplotFile != "") {
    print "Using Default gnuplot Script\n";
    
    open(gpFile, "> plotScript.gp");
    print gpFile "set term png \n";
    print gpFile "set autoscale\n";
    print gpFile "set logscale x\n";
    print gpFile "set logscale y\n";
    print gpFile "set grid xtics ytics\n";
    
    print gpFile "set title \"$test_title\"\n";
    print gpFile "set output \"orderAccuracy.png\"\n";


    # generate the curvefit
    print gpFile "f1(x) = a1*x**b1                # define the function to be fit \n";
    print gpFile "a1 = 0.1; b1 = 0.01;            # initial guess for a1 and b1 \n";
    print gpFile "fit f1(x) 'L2norm.dat' using 1:2 via a1, b1 \n \n";
    
    print gpFile "set label 'Error = a * (Spatial Resolution)^b' at screen 0.2,0.4 \n";
    print gpFile "set label 'a = %3.5g',a1      at screen 0.2,0.375 \n";
    print gpFile "set label 'b = %3.5g',b1      at screen 0.2,0.35  \n\n";
    
    # comparing against a baseline
    if (-e "baseLine/L2norm.dat"){
      print gpFile "plot \'L2norm.dat\' using 1:2 t \'Current test\' with linespoints, \'baseLine/L2norm.dat\' using 1:2 t \'Base Line\' with linespoints\n";
    }else{
      print gpFile "plot 'L2norm.dat' using 1:2 t 'Current test' with linespoints, f1(x) title 'curve fit' \n";
    }
    close(gpFile);
          
    `gnuplot plotScript.gp >&gp.out`;
  } else{
    print "Now plotting Analysis with $gnuplotFile \n";
    `gnuplot $gnuplotFile >&gp.out`;
  }
}
