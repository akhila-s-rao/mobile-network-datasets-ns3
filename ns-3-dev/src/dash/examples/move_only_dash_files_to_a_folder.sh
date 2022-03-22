if [ $# -ne 2 ]; then
       echo "Please enter the parent folder and general string of the files to be removed (e.g. /home/someone/somewhere Ul*.txt)"
       exit 0
fi
cd "$1"
file1="dash_client_logs.txt"
mkdir "$1_only_dash_logs"


find . -name "$file" -type f


   find . -name "$file" -type f

