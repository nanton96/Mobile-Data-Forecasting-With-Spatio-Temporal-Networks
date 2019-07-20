
#!/bin/sh

start_epoch="$1"
epoch_interval=10
total_epoch= echo "$1 + $epoch_interval" | bc
echo $total_epoch