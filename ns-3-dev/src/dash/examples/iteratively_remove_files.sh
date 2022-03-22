if [ $# -ne 2 ]; then
       echo "Please enter the parent folder and general string of the files to be removed (e.g. /home/someone/somewhere Ul*.txt)"
       exit 0
fi
cd "$1"
file="$2"
#file="UlInterferenceStats.txt"
echo "Listing file that will be deleted"
find . -name "$file" -type f
echo "Are you sure you want to delete these files (y/n) ? "
read inp
if [ "$inp" == "y" ]
then
   find . -name "$file" -type f -delete
fi
