#!/usr/bin/perl
$srcroot = $ARGV[0];

  print $srcroot,"\n";
if($srcroot eq "") {
  print " You need to specify the path to SCIRun \n";
  print " For example:  /usr/sci/projects/Uintah/tester/Linux/SCIRun.062602 \n";
  print " now exiting \n";
  exit;
}

$filename = "$srcroot" . "/src/Packages/Uintah/StandAlone/sub.mk";

  print "now working on $filename \n";
  open(IN, $filename);

  $search_str      = ".*sus.cc";
  $replacement_str = "SRCS := \$\(SRCDIR\)/sus.cc
SRCS := \$\(SRCS\) \$\(SRCDIR\)/FakeArches.cc";

   $search_str2     = ".
        Packages/Uintah/CCA/Components/Arches/Mixing .
        Packages/Uintah/CCA/Components/Arches_fortran";
  $replacement_str2 = "";

  $search_str3     = "
        Packages/Uintah/CCA/Components/Arches .
        Packages/Uintah/CCA/Components/MPMArches .";
  $replacement_str3 = "";
  
  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1      = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
       $changed1 = "true";
  }
  $changed2      = "false";
  if ($changed1 == "true" &&$text =~ s/$search_str2/$replacement_str2/g) {
       $changed2 = "true";
  }
  $changed3      = "false";
  if ($changed2 == "true" &&$text =~ s/$search_str3/$replacement_str3/g) {
       $changed3 = "true";
  }

  close IN;
  print "changed file ",$changed1," ", $changed2," ", $changed3, "\n";
  if ($changed3 eq "false") {
    print " This sub.mk file wasn't changed \n";
  }
  if ($changed3 eq "true") {
      open(OUT, "> " . $filename);
      print $filename .  "\n";
      print OUT $text;
      close OUT;
  }

#.............................
# now take care of Components sub.mk
$filename = "$srcroot" . "/src/Packages/Uintah/CCA/Components/sub.mk";

  print "now working on $filename \n";
  open(IN, $filename);

  $search_str       = "
	..SRCDIR./MPMArches .
	..SRCDIR./Arches .
	..SRCDIR./Arches/fortran .
	..SRCDIR./Arches/Mixing .";
  $replacement_str  = "";

  @lines = <IN>;
  $text = join("", @lines);
  
  $changed1 = "false";
  if ($text =~ s/$search_str/$replacement_str/g) {
    $changed1 = "true";
  }

  close IN;
  
  print "changed file ",$changed1, "\n";
  if ($changed1 eq "false") {
    print " The sub.mk file wasn't changed \n";
  }
  if ($changed1 eq "true") {
    open(OUT, "> " . $filename);    
    print $filename .  "\n";        
    print OUT $text;                
    close OUT;                      
  }

