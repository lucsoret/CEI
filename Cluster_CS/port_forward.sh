#!/bin/bash

if [ $# -ne 2 ] && [ $# -ne 3 ]
then
    echo "Usage : "
    echo `basename $0` " login <mode> <jobnum>"
    echo " mode = 0 : within CentraleSupelec"
    echo " mode = 1 : externally"
    echo "<jobnum> is optional; if you do not provide it, I set it to 1"
    exit
fi

login=$1
mode=$2
if [ $# -ne 3 ]; then
    job_num=1
else
    job_num=$3
fi

GREEN="\\e[1;32m" 
NORMAL="\\e[0;39m" 
RED="\\e[1;31m"
BLUE="\\e[1;34m" 
MAGENTA="\\e[1;35m"

display_info() {
    echo -e "$BLUE $1 $NORMAL"
}
display_wait() {
    echo -e "$MAGENTA $1 $NORMAL"
}
display_success() {
    echo -e "$GREEN $1 $NORMAL"
}
display_error() {
    echo -e "$RED $1 $NORMAL"
}

if [ $mode == 0 ]
then
    ssh_options=
else
    # Bounce over the proxy
    ssh_options="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
fi

get_booked_host ()
{
    if [ $mode == 0 ]
    then
	ssh $login@term2.grid "oarstat -f -j `cat job_id$job_num` " | grep assigned_hostnames | awk -F " = " '{print $NF}'
    else
	# Bounce over the proxy
	ssh_options="-o ProxyCommand=ssh -W %h:%p $login@ghome.metz.supelec.fr"
	ssh "$ssh_options" $login@term2.grid "oarstat -f -j `cat job_id$job_num` " | grep assigned_hostnames | awk -F " = " '{print $NF}'
    fi
}
test_job_state ()
{
    if [ $mode == 0 ]
    then
	ssh $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    else
	# Bounce over the proxy
	ssh "$ssh_options" $login@term2.grid "oarstat -s -j `cat job_id$job_num` " | awk -F ": " '{print $NF}' -
    fi   
}


if [ -f job_id$job_num ]; then
    display_info "The file job_id$job_num exists. I am checking the reservation is still valid"
    job_state=`test_job_state`
    if [ "$job_state" != "Running" ]; then
	display_error "   The reservation is not running yet or anymore. Please book a machine"
	exit 0
    fi
    display_success "   The reservation is still running"
    host=`get_booked_host`
    display_info "Activation port forwarding"
    if [ $mode == 0 ]
    then
	ssh -v -N -L 6006:$host:6006 $login@term2.grid
    else
	ssh "$ssh_options" -v -N -L 6006:$host:6006 $login@term2.grid
    fi
else
    display_error "   The file job_id$job_num does not exist. Please book a machine"
    exit 0
fi
