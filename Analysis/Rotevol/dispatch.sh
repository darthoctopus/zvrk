#!/bin/bash 
#### Create a grid of stellar evolutionary tracks varied in an initial parameter

change_param() { 
	# Modifies a parameter in the current inlist. 
	# args: ($1) name of parameter 
	#       ($2) new value 
	#       ($3) filename of inlist where change should occur 
	# Additionally changes the 'inlist_0all' inlist. 
	# example command: change_param initial_mass 1.3 
	# example command: change_param log_directory 'LOGS_MS' 
	# example command: change_param do_element_diffusion .true. 
	param=$1 
	newval=$2 
	filename=$3 
	search="^\s*\!*\s*$param\s*=.+$" 
	replace="      $param = $newval" 
	sed -r -i.bak -e "s/$search/$replace/g" $filename 
	
	if [ ! "$filename" == 'inlist_0all' ]; then 
		change_param $1 $2 "inlist_0all" 
	fi 
} 

set_inlist() { 
	# Changes to a different inlist by modifying where "inlist" file points 
	# args: ($1) filename of new inlist  
	# example command: change_inlists inlist_2ms 
	newinlist=$1 
	echo "Changing to $newinlist" 
	change_param "extra_star_job_inlist2_name" "'$newinlist'" "inlist" 
	change_param "extra_controls_inlist2_name" "'$newinlist'" "inlist" 
}

job() {
	LABEL=$1
	M=$2
	Y=$3
	Z=$4
	AMLT=$5

	WORKDIR=work_$LABEL

	cp -R work $WORKDIR
	cp inlists/inlist* $WORKDIR
	cp inlists/history_columns.list $WORKDIR
	cp inlists/profile_columns.list $WORKDIR

	cd $WORKDIR
		
	change_param mixing_length_alpha $AMLT "inlist_0all"
	change_param write_profile_when_terminate ".false." "inlist_0all"
	
	# create pre-main sequence model 
	inlist="inlist_1pms" 
	set_inlist $inlist 
	change_param initial_y $Y $inlist
	change_param new_Y $Y $inlist
	change_param initial_z $Z $inlist
	change_param new_Z $Z $inlist
	change_param Zbase $Z $inlist
	change_param initial_mass $M $inlist
	./rn  2>&1
	
	# run until log g limit
	inlist="inlist_2ms" 
	set_inlist $inlist 
	change_param "log_directory" "'LOGS_$LABEL'" $inlist
	./re 2>&1

	# actually make some profiles
	inlist="inlist_3rgb" 
	set_inlist $inlist 
	change_param "log_directory" "'LOGS_$LABEL'" $inlist
	./re 2>&1

	# done
	cd ..

	# purge profile files
	mv $WORKDIR/rgb.mod rgb/$LABEL.mod
	#rm $WORKDIR/LOGS_$LABEL/profile*.data
	mv $WORKDIR/LOGS_$LABEL tracks
	rm -rf $WORKDIR
}

job $1 $2 $3 $4 $5

