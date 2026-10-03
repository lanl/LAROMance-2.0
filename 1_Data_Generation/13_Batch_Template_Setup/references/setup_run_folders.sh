#!/bin/bash

# Check if the specific argument is passed (e.g., --autojob)
RUN_AUTOJOB=false

for arg in "$@"; do
    if [[ "$arg" == "--autojob" ]]; then
        RUN_AUTOJOB=true
        break
    fi  
done

# name variables
sourcedir="template_run_folder"
newdirbase="run_"
input_file="$sourcedir/samples.csv"

# move to a safe directory
cd /scratch

# check if input_file exists
if [ ! -f "$input_file" ]; then
  echo "File not found!"
  exit 1
fi

# Read .csv file with LHS samples, skipping the first line (header)
tail -n +2 "$input_file" | while IFS=, read -r TEMP VMJ2 RHOC; do
    # Increment count
    cnt=$((cnt+1))
    
    # Create new directory
    newdir=$newdirbase${cnt}
    mkdir $newdir

    # Copy files from source directory to new directory
    rsync -avz --include="*.slurm" --include="*.in" --include="*.dat" --include="*.sx" --exclude="*" $sourcedir/ $newdir/

    # # Rename the .slurm file
    # mv $newdir/run.slurm $newdir/$newdir.slurm

    # Update the SLURM file with the new directory name
    sed -i "s/#SBATCH --job-name=LApx/#SBATCH --job-name=$newdir/g" $newdir/run.slurm

    # Update the temperature in BCfile.in
    sed -i "s/TEMP_PLACE/$TEMP/g" "$newdir/BCFile.in"

    # Calculate stress radial and update in BCfile.in
    stress_hoop=$VMJ2
    time_ramp=$(echo "0.1 * $stress_hoop" | bc -l)
    stress_rate_hoop=10.0
    stress_rate_rad=5.
    sed -i "s/TIME_PLACE/$time_ramp/g" "$newdir/BCFile.in"
    sed -i "s/STRESS_HOOP_PLACE/$stress_rate_hoop/g" "$newdir/BCFile.in"    # original tube texture	
    sed -i "s/STRESS_RADIAL_PLACE/$stress_rate_rad/g" "$newdir/BCFile.in"   # original tube texture

    # Adapt single crystal (.sx) file
    # first convert to numeric format
    rhoc=$RHOC

    # replace the placeholders in the .sx file
    sed -i "s/RHOC_PLACE/$rhoc/g" "$newdir/single_crystal_model.sx"
done

# Conditionally call autojob
if [[ "$RUN_AUTOJOB" == true ]]; then
    ~/bin/autojob "$newdirbase"
fi
