#!/usr/bin/perl -w

my $line;
my $return_type;
my $function_name;
my $function_args;

while (<>){
    if ($_ =~ /#/){
	print $_;
	print "\n";
        next;
    }
    chomp;
    my @items = split /__ARGS/, $_;
    $function = $items[0];
    $function_args = $items[1];
    ($return_type, $function_name) = split / /, $function;
    $function_name =~ s/MPI_//;
    $function_args =~ s/\(\(//;
    $function_args =~ s/\)\)\;//;
    my @args_array = split ',', $function_args;
#    print "$function_name \n";
    my $size = scalar(@args_array);

    $callsig = "";
    $vars = "";
    $callargs = "";
    $varinit = "";

    for($i = 0; $i < $size; $i++){
	$args_array[$i] =~ s/ //g;
#	print "arg${i}: $args_array[$i] \n";
	# CALLSIG
	if ($i==0){
	    $callsig .= "$args_array[$i] arg${i}";
	}
	else{
	    $callsig .= ", $args_array[$i] arg${i}";
	}

	# VARS
	$vars .= "$args_array[$i] arg${i}; ";

	# CALLARGS
	if ($i==0){
	    $callargs .= "arg${i}";
	}
	else{
	    $callargs .= ", arg${i}";
	}
	
	#VARINIT
	if ($i==0){
	    $varinit .= "arg${i}(arg${i})";
	}
	else{
	    $varinit .= ", arg${i}(arg${i})";
	}
	
    }

    print<<ENDOFPROTO;
//-------------------------- MPI_${function_name} ------------------------
#define NAME     ${function_name}
#define TEXTNAME "MPI_${function_name}"
#define CALLSIG  $callsig
#define VARS     $vars
#define CALLARGS $callargs
#define VAR_INIT ${varinit}, retval(retval)
#define RET_TYPE $return_type
MPI_CLASS_BODY

#undef NAME
#undef TEXTNAME
#undef CALLSIG
#undef VARS
#undef CALLARGS
#undef VAR_INIT
#undef RET_TYPE

ENDOFPROTO
}
