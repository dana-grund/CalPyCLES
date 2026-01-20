
JOB_ID=$1 #1234
N_JOBS=$2 #10

FILE=./job_array_summary_$JOB_ID.txt
rm $FILE

# for all jobs print summary to file
for ARRAY_NR in $(seq 0 $((N_JOBS - 1))); do
    myjobs -j ${JOB_ID}_${ARRAY_NR} >> $FILE
done

echo "Summary of job array $JOB_ID written to $FILE"
