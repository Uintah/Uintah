#!/usr/bin/perl

# use module
use XML::Simple;
use Data::Dumper;
$xml = new XML::Simple(forcearray => 1);


# read XML file
$data = $xml->XMLin("$ARGV[0]", KeepRoot => 1);

#print Dumper($data);

# Reading the meta data and finding out the number of tests

my $i = 0;
my $j = 0;
my $tmp_err = -1;

$gnuplotFile = $data->{start}->[0]->{gnuplotFile}->[0];                # if a user provides a gnuplot file
print "analyze_results:Using gnuplotFile: $gnuplotFile \n";

foreach $e (@{$data->{start}->[0]->{Test}})
{
  $test_title  = $e->{Meta}->[0]->{Title}->[0];
  $int_command = $e->{Meta}->[0]->{Interactive}->[0];
  $output_file = $test_title;
  $output_file =~ tr/" "/"_"/;  # Replacing the spaces in the test title with underscores
  $launcher    = $e->{Meta}->[0]->{Launcher}->[0];

  @x = @{$e->{x}};
  
  @uda_files = @{$e->{udaFile}};

  for ($k = 0 ; $k<=$#uda_files; $k++)
  {   
    ####################
    # This is to take care of the number of tests 
    ####################
    if (-e ".$output_file.tmp")
    {
      $tmp_err =  `cat .$output_file.tmp`;   # This file stores the number of tests, so we decrement it every time we run a compare_mms
      chomp($tmp_err);
      $tmp_err--;
      
      `echo $tmp_err >.$output_file.tmp`;
    } 
    ############################
    # Run the comparison tests
    ############################

    print "$int_command -o $uda_files[$k].tmp -uda $uda_files[$k]\n";
    `rm -f $uda_files[$k].tmp`;    # Deleting the tmp file (if it already exists)
   
    @args = ("$int_command","-o","$uda_files[$k].tmp","-uda","$uda_files[$k]");
    system("@args")==0 or die("ERROR(analyze_results.pl): @args failed: $?");
    
    $tmp = `cat $uda_files[$k].tmp`; # This is the output from the comparison utility
    chomp($tmp);   # Removing the carriage return from the end
    
    my $prev_line = "";
    if (-e "$output_file.dat")
    {
      $prev_line = `tail -q -n1 $output_file.dat`; # This reads the last line from the global err file
      chomp($prev_line);      # Removing the carriage return 
    }

    if ($prev_line != "")  # Checking if the prev_line is empty (if the file is created for the first time this can happen)
    {
      @values = split(/ /,$prev_line);
      
      
      $R = $x[$k]/$values[0];
      $errorRatio = $values[1]/$tmp;
      
     # $order_of_accuracy = log($errorRatio)/log($R);
      $order_of_accuracy = 0;
      print "Analyze Results:\t Error ratio: $errorRatio, R: $R,\t L2Norm: $tmp \tOrder-of-accuracy: $order_of_accuracy \n\n";
      
      `echo $x[$k] $tmp $order_of_accuracy >> $output_file.dat`;
    } 
    else
    {
      print "$x[$k] $tmp $order_of_accuracy\n";
      `echo $x[$k] $tmp 0 >> $output_file.dat`;
    }
  }


  #______________________________________________________________________
  # Create a default gnuplot script
  if ( $gnuplotFile != "") {
  
    open(gpFile, ">$output_file.gp");
    print gpFile "set term png \n";
  
#    print gpFile "set ylabel \"Error\"\n";
#    print gpFile "set xlabel \"Resolution\"\n";    # The problem is, x-axis can be anything (viscosity, resolution, timestep)

    print gpFile "set autoscale\n";
    print gpFile "set logscale y\n";
    print gpFile "set grid xtics ytics\n";
    print gpFile "set y2tics\n";
    print gpFile "set title \"$test_title\"\n";
    print gpFile "set output \"err_$output_file.png\"\n";

    # comparing against a baseline
    if (-e "baseLine/$output_file.dat"){
      print gpFile "plot \'$output_file.dat\' using 1:2 t \'Current test\' with linespoints, \'baseLine/$output_file.dat\' using 1:2 t \'Base Line\' with linespoints\n";
    }   
    else{
      print gpFile "plot \'$output_file.dat\' using 1:2 t \'Current test\' with linespoints\n"; 
    }
    print gpFile "unset logscale y\n";
    print gpFile "set title \"Order of Accuracy - $test_title\"\n";
    print gpFile "set output \"order_$output_file.png\"\n";

    if (-e "baseLine/$output_file.dat"){
      print gpFile "plot \'$output_file.dat\' using 1:3 t \'Current test\' with linespoints,  \'baseLine/$output_file.dat\' using 1:3 t \'Base Line\' with linespoints\n";
    }
    else{
      print gpFile "plot \'$output_file.dat\' using 1:3 t \'Current test\' with linespoints\n";
    }

    close(gpFile);
          
    `gnuplot $output_file.gp`;
  } else{
    print "Now plotting results with $gnuplotFile \n";
    `gnuplot $gnuplotFile`;
  }
  
  
  
  #__________________________________
  # The DONE file has the basic info about the results - The two plots will be included and the test title will be added to the file
  if ($tmp_err==0)
  {
    `echo \"Title: $test_title\" >> $launcher.DONE`;
    `echo \"Results\/err_$output_file.png\" >> $launcher.DONE`;
    `echo \"Results\/order_$output_file.png\" >> $launcher.DONE`;
    
    # The results (the plots and the little table) are moved in a directory called Results
    
    `mkdir -p Results`;
    `mv $output_file.gp $output_file.dat err_$output_file.png order_$output_file.png Results/.`;
    `rm -f .$output_file.tmp`;
  }  
}
if($tmp_err==-1)
{
  `mkdir -p Results1`;
  `mv $output_file.gp $output_file.dat err_$output_file.png order_$output_file.png Results1/.`;
}
